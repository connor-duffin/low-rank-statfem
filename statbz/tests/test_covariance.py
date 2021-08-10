import numpy as np
import matplotlib.pyplot as plt

import fenics as fe
from scipy.sparse import csr_matrix
from scipy.linalg import eigh

from statbz.covariance import (boundary, cov_approx, cov_approx_keops,
                               sq_exp_covariance, sq_exp_spectral_density)

mesh = fe.UnitIntervalMesh(200)
V = fe.FunctionSpace(mesh, "CG", 1)
x_grid = V.tabulate_dof_coordinates()
scale = 1.
ell = 1e-1
k = 128


def test_sq_exp():
    G = sq_exp_covariance(x_grid, scale=scale, ell=ell)
    np.testing.assert_allclose(G[0, 0], 1.)
    assert G.shape[0] == x_grid.shape[0]


def test_cov_approx_keops():
    rtol = 1e-4  # small rtol due to single precision
    norm = np.linalg.norm

    k_test = 16
    G = sq_exp_covariance(x_grid, scale=scale, ell=ell)
    G_vals, G_vecs = eigh(G)
    G_vals_keops, G_vecs_keops = cov_approx_keops(x_grid,
                                                  scale=scale,
                                                  ell=ell,
                                                  k=k_test)
    # check approx equal eigenvalues
    np.testing.assert_allclose(G_vals[-k_test:], G_vals_keops, rtol=rtol)

    # check approx equal matrices
    G_approx = G_vecs_keops @ np.diag(G_vals_keops) @ G_vecs_keops.T
    rel_error = norm(G - G_approx) / norm(G)
    assert rel_error <= rtol


def test_cov_approx():
    vals, vecs = cov_approx(V, k=k, scale=scale, ell=ell)

    # first two should be deleted for better approx.
    assert vals.shape == (128, )
    assert vecs.shape == (x_grid.shape[0], 128)

    # check ordering
    vals_sorted = np.sort(vals)[::-1]
    np.testing.assert_almost_equal(vals, vals_sorted)


def test_cov_approx_2d():
    mesh = fe.UnitSquareMesh(32, 32)
    V = fe.FunctionSpace(mesh, "CG", 1)
    x_grid = V.tabulate_dof_coordinates()
    scale, ell = 1., 1e-1
    k = 32

    bc = fe.DirichletBC(V, fe.Constant(0), boundary)

    # mass matrix used to ensure orthogonality on the weighted inner product
    # <u, v> = u M v'
    u, v = fe.TrialFunction(V), fe.TestFunction(V)
    M = fe.PETScMatrix()
    fe.assemble(u * v * fe.dx, tensor=M)
    bc.apply(M)
    M = M.mat()
    M_scipy = csr_matrix(M.getValuesCSR()[::-1], shape=M.size)

    vals, vecs = cov_approx(V, k=k, scale=scale, ell=ell)

    assert vals.shape == (32, )
    assert vecs.shape == (1089, 32)

    # check orthogonality wrt mass matrix
    np.testing.assert_almost_equal(vecs[:, 0] @ M_scipy @ vecs[:, 0], 1.)
    np.testing.assert_almost_equal(vecs[:, 0] @ M_scipy @ vecs[:, 1], 0.)


def test_cov_approx_neumann():
    mesh = fe.UnitSquareMesh(32, 32)
    V = fe.FunctionSpace(mesh, "CG", 1)
    x_grid = V.tabulate_dof_coordinates()
    scale, ell = 1., 1e-1

    # check orthogonality on the weighted inner product
    # <u, v> = u M v'
    u, v = fe.TrialFunction(V), fe.TestFunction(V)
    M = fe.PETScMatrix()
    fe.assemble(u * v * fe.dx, tensor=M)
    M = M.mat()
    M_scipy = csr_matrix(M.getValuesCSR()[::-1], shape=M.size)

    vals, vecs = cov_approx(V, k=32, scale=scale, ell=ell, bc="Neumann")

    assert vals.shape == (32, )
    assert vecs.shape == (1089, 32)

    # check orthogonality wrt mass matrix
    np.testing.assert_almost_equal(vecs[:, 0] @ M_scipy @ vecs[:, 0], 1.)
    np.testing.assert_almost_equal(vecs[:, 0] @ M_scipy @ vecs[:, 1], 0.)

    analytical_eigenvalues = np.array(
        [0, np.pi**2, np.pi**2, 2 * np.pi**2, 4 * np.pi**2, 4 * np.pi**2])
    spectral_density = sq_exp_spectral_density(np.sqrt(analytical_eigenvalues),
                                               scale,
                                               ell,
                                               D=2)
    np.testing.assert_array_almost_equal(vals[:6], spectral_density, decimal=4)


if __name__ == "__main__":
    # actual covariance matrix
    G = sq_exp_covariance(x_grid, scale, ell)

    # true eigenvalues
    laplace_true_eigenvals = np.array([(n * np.pi)**2
                                       for n in range(1, k + 1)])
    true_eigenvecs = np.array(
        [np.sqrt(2) * np.sin(n * np.pi * x_grid) for n in range(1, k + 1)])
    true_eigenvecs = true_eigenvecs[:, :, 0].T
    true_eigenvals = sq_exp_spectral_density(np.sqrt(laplace_true_eigenvals),
                                             scale=scale,
                                             ell=ell)

    # numeric approximation
    G_vals, G_vecs = cov_approx(V, k=k, scale=scale, ell=ell)
    G_approx = G_vecs @ np.diag(G_vals) @ G_vecs.T
    G_approx_true = true_eigenvecs @ np.diag(
        true_eigenvals) @ true_eigenvecs.T  # analytical

    print("Rel. l2 difference: ||G - G_approx|| / || G || = " +
          f"{np.linalg.norm(G - G_approx) / np.linalg.norm(G)}")

    fig, axs = plt.subplots(1, 3, sharey=True)
    axs[0].set_title("G approx. (numeric)")
    im = axs[0].imshow(G_approx)
    plt.colorbar(im, ax=axs[0])

    axs[1].set_title("G approx. (analytical)")
    im = axs[1].imshow(G_approx_true)
    plt.colorbar(im, ax=axs[1])

    axs[2].set_title("G truth")
    im = axs[2].imshow(G)
    plt.colorbar(im, ax=axs[2])
    plt.show()
