import fenics as fe

import logging
import numpy as np

from scipy.linalg import cholesky, eigh
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import aslinearoperator, eigsh
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, kn

from pykeops.numpy import LazyTensor

logger = logging.getLogger(__name__)
fe.set_log_level(40)


def boundary(x, on_boundary):
    return on_boundary


def matern_covariance(grid, scale=1., ell=1., nu=2):
    """
    Compute Matern covariance matrix.

    Parameters
    ----------
    grid : ndarray
        Spatial grid of shape (n_points, n_dimensions).
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    nu : float
        Smoothness parameter.
    """
    kappa = np.sqrt(2 * nu) / ell
    dist = pdist(grid, metric="euclidean")
    dist = squareform(dist)
    dist[dist == 0.0] += np.finfo(float).eps  # strict zeros result in nan

    K = scale**2 / (2**(nu - 1) * gamma(nu)) * (kappa * dist)**nu * kn(
        nu, kappa * dist)
    K[np.diag_indices_from(K)] += 1e-8
    return K


def matern_spectral_density(omega, scale=1., ell=1., nu=2):
    """
    Matern spectral density.

    Parameters
    ----------
    omega : float
        Frequency at which to evaluate the spectral density.
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    nu : float
        Smoothness parameter.
    """
    kappa = np.sqrt(2 * nu) / ell
    s = (scale**2 * np.sqrt(4 * np.pi) * gamma(nu + 1 / 2) * kappa**(2 * nu) /
         gamma(nu) * (kappa**2 + 4 * np.pi**2 * omega**2)**(-(nu + 1 / 2)))
    return s


def cov_approx_keops(grid, scale, ell, k=32):
    """
    Approximate a squared-exponential covariance matrix using KeOps.

    If a GPU device is detected then it will use this to accelerate
    computations.

    Parameters
    ----------
    grid : ndarray
        Spatial grid of shape (n_points, n_dimensions).
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    k : int
        Number of desired eigenvalues.
    """
    from pykeops.config import gpu_available
    logger.info("Approximating GP with KeOps, using GPU: %s",
                str(gpu_available))

    # compute in single, return in double
    dtype_compute = "float32"
    dtype_return = "float64"

    def sq_exp_keops(grid, scale, ell):
        x_ = grid / ell
        x_i, x_j = LazyTensor(x_[:, None, :]), LazyTensor(x_[None, :, :])
        K = (-((x_i - x_j)**2).sum(2) / 2).exp()
        K *= scale**2
        return K

    grid = grid.astype(dtype_compute)
    K_keops = sq_exp_keops(grid, scale, ell)
    assert K_keops.dtype == "float32"

    K_keops = aslinearoperator(K_keops)
    vals, vecs = eigsh(K_keops, k=k)

    return vals.astype(dtype_return), vecs.astype(dtype_return)


def sq_exp_covariance(grid, scale, ell):
    """
    Squared exponential covariance function.

    Parameters
    ----------
    grid : ndarray
        Spatial grid of shape (n_points, n_dimensions).
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    """
    dist = pdist(grid, metric="euclidean")
    dist = squareform(dist)

    K = scale**2 * np.exp(-dist**2 / (2 * ell**2))
    K[np.diag_indices_from(K)] += 1e-10
    return K


def sq_exp_spectral_density(omega, scale, ell, D=1):
    """
    Squared exponential spectral density.

    Parameters
    ----------
    omega : float
        Frequency at which to evaluate the spectral density.
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    """
    return (scale**2 * (2 * np.pi * ell**2)**(D / 2) *
            np.exp(-omega**2 * ell**2 / 2))


def cov_approx(V, k=64, scale=1., ell=1., bc="Dirichlet"):
    """
    Approximate the SqExp covariance using Hilbert-GP.

    For full details:
    Solin, A., Särkkä, S., 2020. Hilbert space methods for reduced-rank
    Gaussian process regression. Stat Comput 30, 419–446.
    https://doi.org/10.1007/s11222-019-09886-w

    Parameters
    ----------
    V : fenics.FunctionSpace
        FunctionSpace on which to compute the approximation.
    k : int, optional
        Number of modes to take in the approximation.
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    bc : str, optional
        Boundary conditions to use in the approximation. Either 'Dirichlet' or
        'Neumann'.
    """
    bc_types = ["Dirichlet", "Neumann"]
    if bc not in bc_types:
        raise ValueError("Invalid bc, expected one of {bc_types}")
    elif bc == "Dirichlet":
        bc = fe.DirichletBC(V, fe.Constant(0), boundary)
    else:
        bc = None

    # define variational problem
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    a = fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
    A = fe.PETScMatrix()
    fe.assemble(a, tensor=A)

    M = fe.PETScMatrix()
    fe.assemble(u * v * fe.dx, tensor=M)

    if bc is not None:
        bc.apply(A)
        bc.apply(M)

    M = M.mat()
    M_scipy = csr_matrix(M.getValuesCSR()[::-1], shape=M.size)
    A = A.mat()
    A_scipy = csr_matrix(A.getValuesCSR()[::-1], shape=A.size)

    # use shift-invert mode in scipy
    # faster and more stable than SLEPc
    laplace_eigenvals, eigenvecs = eigsh(A_scipy, k, M_scipy, sigma=1e-14)
    logger.info("Laplacian eigenvalues: %s", laplace_eigenvals)
    eigenvals = sq_exp_spectral_density(
        np.sqrt(laplace_eigenvals),
        scale=scale,
        ell=ell,
        D=V.tabulate_dof_coordinates().shape[1])

    # scale so eigenfunctions are orthonormal on function space
    eigenvecs_scale = eigenvecs.T @ M_scipy @ eigenvecs
    eigenvecs = eigenvecs / np.sqrt(eigenvecs_scale.diagonal())

    # indices = np.where(np.isclose(laplace_eigenvals, 1.))
    # if len(indices) > 0:
    #     indices = np.concatenate(
    #         (indices, np.where(np.isclose(laplace_eigenvals, 0.))))
    # else:
    #     indices = np.where(np.isclose(laplace_eigenvals, 0.))

    # eigenvals = np.delete(eigenvals, indices)
    # eigenvecs = np.delete(eigenvecs, indices, axis=1)

    return (eigenvals, eigenvecs)


def cov_kron_evd(K, n_modes=6):
    """
    (DEPRECATED) Compute the leading `n_modes` eigenvalue/vector pairs of
    kron(K, K).
    """
    n = K.shape[0]
    vecs = np.zeros((n**2, n_modes))
    vals_K, vecs_K = eigh(K)
    vals_cov = np.kron(vals_K, vals_K)
    idx_sorted = np.argsort(vals_cov)[(-n_modes):]
    vals = vals_cov[idx_sorted]

    idx_first_vec = idx_sorted // n
    idx_second_vec = idx_sorted % n

    for k, (i, j) in enumerate(zip(idx_first_vec, idx_second_vec)):
        vecs[:, k] = np.kron(vecs_K[i], vecs_K[j])
        np.testing.assert_almost_equal(vals_K[i] * vals_K[j], vals[k])

    return vals, vecs


def cov_fenics_evd(scale, ell, V, n_modes=6):
    """
    (DEPRECATED) Compute the EVD of a cov. matrix defined over V (assumed
    2D grid).
    """
    grid_fenics = V.tabulate_dof_coordinates()
    idx_sorted = np.lexsort((grid_fenics[:, 1], grid_fenics[:, 0]))

    # P : fenics ordering -> kronecker ordering, and
    # P.T : kronecker ordering -> fenics ordering
    P = coo_matrix(
        (np.ones_like(idx_sorted), (np.sort(idx_sorted), idx_sorted)))
    P = P.tocsr()

    grid = P @ grid_fenics
    x = np.unique(grid[:, 0])

    K = sq_exp_covariance(x[:, np.newaxis], np.sqrt(scale), ell)
    vals, vecs = cov_kron_evd(K, n_modes=n_modes)
    return vals, P.T @ vecs  # permute for proper ordering


def cov_fenics_chol(scale, ell, grid_fenics):
    """
    (DEPRECATED) Compute the Cholesky decomposition of a cov. matrix defined
    over V (assumed 2D grid).
    """
    idx_sorted = np.lexsort((grid_fenics[:, 1], grid_fenics[:, 0]))

    # P : fenics ordering -> kronecker ordering, and
    # P.T : kronecker ordering -> fenics ordering
    P = coo_matrix(
        (np.ones_like(idx_sorted), (np.sort(idx_sorted), idx_sorted)))
    P = P.tocsr()

    grid = P @ grid_fenics
    x = np.unique(grid[:, 0])

    K = sq_exp_covariance(x[:, np.newaxis], np.sqrt(scale), ell)
    K_chol = cholesky(K, lower=True)
    return np.kron(K_chol, K_chol)
