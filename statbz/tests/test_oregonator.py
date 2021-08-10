import pytest
import numpy as np

from dolfin import (assemble, interpolate, errornorm, Expression,
                    NonlinearVariationalSolver)
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import splu

from statbz.oregonator import Oregonator, StochasticOregonator, StatOregonator
from statbz.utils import build_observation_operator, dolfin_to_csr

np.random.seed(27)

SETTINGS = {"L": 50, "nx": 32, "dt": 1e-3, "correction": True}
PARAMS = {"f": 0.95, "q": 0.002, "eps": 0.75, "Du": 0.001, "Dv": 0.001}


@pytest.fixture
def bz_determ():
    bz_determ = Oregonator(SETTINGS, PARAMS)

    return bz_determ


@pytest.fixture
def bz_stochastic():
    bz_stochastic = StochasticOregonator(1e-3, 10., SETTINGS, PARAMS)

    return bz_stochastic


@pytest.fixture
def bz_stochastic_slow():
    settings_slow = {**SETTINGS, "error_variable": "v"}
    bz_stochastic = StochasticOregonator(1e-3, 10., settings_slow, PARAMS)

    return bz_stochastic


class TestDeterministicOregonator:
    def test_init(self, bz_determ):
        assert bz_determ.n_u_dofs == 1089
        assert bz_determ.n_v_dofs == 1089

    def test_set_uv(self, bz_determ):
        bz_determ.setup_solve("random")
        bz_determ.w.vector()[:] = 0.

        np.testing.assert_allclose(bz_determ.u, np.zeros_like(bz_determ.u))
        np.testing.assert_allclose(bz_determ.v, np.zeros_like(bz_determ.v))

        twos = 2 * np.ones_like(bz_determ.w.vector()[:])

        bz_determ.w.vector()[bz_determ.v_dofs] = 2.
        bz_determ.set_u_from_v()
        np.testing.assert_allclose(bz_determ.w.vector()[:], twos)

        bz_determ.w.vector()[:] = 0.

        bz_determ.w.vector()[bz_determ.u_dofs] = 2.
        bz_determ.set_v_from_u()
        np.testing.assert_allclose(bz_determ.w.vector()[:], twos)

    def test_concentration_correction(self, bz_determ):
        bz_determ.setup_solve("random")
        bz_determ.w.vector()[:] -= np.random.uniform(bz_determ.q,
                                                     2 * bz_determ.q)

        dofs_lower = np.where(bz_determ.u < bz_determ.q)
        assert len(dofs_lower) > 0

        bz_determ.concentration_correction()
        u = np.copy(bz_determ.u)
        np.testing.assert_allclose(u[dofs_lower], bz_determ.q)

    def test_solve(self, bz_determ):
        bz_determ.setup_solve("random")
        assert type(bz_determ.solver) == NonlinearVariationalSolver

        assert bz_determ.solver.parameters["nonlinear_solver"] == "snes"

        bz_determ.timestep()
        assert np.all(bz_determ.u >= bz_determ.q)
        assert errornorm(bz_determ.w, bz_determ.w_prev) < 1e-7

        array = np.random.normal(size=bz_determ.w.vector()[:].shape)
        bz_determ.set_w_from_vector(array)
        np.testing.assert_allclose(array, bz_determ.w.vector()[:])


class TestStochasticOregonator:
    def test_init(self, bz_stochastic):
        np.testing.assert_allclose(bz_stochastic.x_xi, bz_stochastic.x_u)

        assert bz_stochastic.xi_cov.shape == (1089, 1089)
        np.testing.assert_allclose(bz_stochastic.xi_cov_chol,
                                   np.tril(bz_stochastic.xi_cov_chol))

        np.testing.assert_allclose(bz_stochastic.xi.vector()[:], 0.)

    def test_solve(self, bz_stochastic):
        bz_stochastic.setup_solve("random")
        bz_stochastic.timestep()

        np.testing.assert_allclose(
            bz_stochastic.xi.vector()[bz_stochastic.v_dofs], 0.)

        assert np.sum(np.abs(
            bz_stochastic.xi.vector()[bz_stochastic.u_dofs])) > 0

    def test_slow(self, bz_stochastic_slow):
        """ Integration test for errors in the slow variable. """
        np.testing.assert_allclose(bz_stochastic_slow.x_xi,
                                   bz_stochastic_slow.x_v)

        assert bz_stochastic_slow.xi_cov.shape == (1089, 1089)
        np.testing.assert_allclose(bz_stochastic_slow.xi_cov_chol,
                                   np.tril(bz_stochastic_slow.xi_cov_chol))

        bz_stochastic_slow.setup_solve("random")
        bz_stochastic_slow.timestep()

        xi = np.copy(bz_stochastic_slow.xi.vector()[:])
        np.testing.assert_allclose(xi[bz_stochastic_slow.u_dofs], 0.)


class TestStatOregonator:
    def test_init(self):
        stat_bz = StatOregonator(128,
                                 128,
                                 1.,
                                 10.,
                                 settings=SETTINGS,
                                 params=PARAMS)

        assert stat_bz.n_modes == 128
        assert stat_bz.n_modes_G == 128
        assert stat_bz.L_cov.shape[1] == 128
        np.testing.assert_allclose(stat_bz.mean, 0.)
        np.testing.assert_allclose(stat_bz.L_cov, 0.)

        # check for orthornomal eigenvectors of K
        M = stat_bz._build_mass_matrix(stat_bz.V_eigen)
        M_lu = splu(M.tocsc())
        G_vec_scaled = M_lu.solve(stat_bz.G_vecs[:, 0])
        np.testing.assert_almost_equal(G_vec_scaled @ G_vec_scaled, 1.)
        np.testing.assert_almost_equal(
            stat_bz.G_vecs[:, 0] @ stat_bz.G_vecs[:, 1], 0.)

        # check that V_eigen is what it should be
        np.testing.assert_allclose(stat_bz.V_eigen.tabulate_dof_coordinates(),
                                   stat_bz.x_u)  # dofs
        np.testing.assert_allclose(stat_bz.V_eigen.tabulate_dof_coordinates(),
                                   stat_bz.x_v)
        np.testing.assert_allclose(
            stat_bz.V.sub(0).mesh().coordinates(),
            stat_bz.V_eigen.mesh().coordinates())  # mesh
        assert (stat_bz.V.sub(0).element().signature() ==
                stat_bz.V_eigen.element().signature())  # elements

        assert stat_bz.L_G_base.shape == (2178, 128)

    def test_init_approx_gp(self):
        stat_bz = StatOregonator(128,
                                 128,
                                 1.,
                                 10.,
                                 settings={**SETTINGS, "approx_gp": True},
                                 params=PARAMS)

        assert stat_bz.n_modes == 128
        assert stat_bz.n_modes_G == 128
        assert stat_bz.L_cov.shape[1] == 128

        # check for orthonormal eigenvectors wrt the L^2 inner product
        np.testing.assert_almost_equal(
            stat_bz.G_vecs[:, 0] @ stat_bz.G_vecs[:, 1], 0.)

    def test_permutation_matrix(self):
        stat_bz = StatOregonator(128,
                                 128,
                                 1.,
                                 10.,
                                 settings=SETTINGS,
                                 params=PARAMS)

        stat_bz._build_permutation_matrix(0)
        assert isinstance(stat_bz.P, csr_matrix)
        assert (stat_bz.P.T @ stat_bz.P != eye(stat_bz.n_u_dofs,
                                               format="csr")).nnz == 0

        # check the map is actually doing what we want
        test = interpolate(Expression(("x[0]", "0"), degree=8), stat_bz.V)
        test = np.copy(test.vector()[:])
        u_vals = stat_bz.x_u[:, 0].flatten()
        np.testing.assert_allclose(stat_bz.P @ u_vals, test)

        # for the different subspace
        stat_bz._build_permutation_matrix(1)
        assert isinstance(stat_bz.P, csr_matrix)
        assert (stat_bz.P.T @ stat_bz.P != eye(stat_bz.n_u_dofs,
                                               format="csr")).nnz == 0

        test = interpolate(Expression(("0", "x[0]"), degree=8), stat_bz.V)
        test = np.copy(test.vector()[:])
        u_vals = stat_bz.x_u[:, 0].flatten()
        np.testing.assert_allclose(stat_bz.P @ u_vals, test)

    def test_timestep_data(self):
        stat_bz = StatOregonator(128, 128, 1., 10.,
                                 settings=SETTINGS,
                                 params=PARAMS)
        stat_bz.setup_solve("random")

        bz_stochastic = StochasticOregonator(1e-3, 10.,
                                             settings=SETTINGS,
                                             params=PARAMS)
        bz_stochastic.setup_solve("random")
        bz_stochastic.timestep()

        y = np.copy(bz_stochastic.u)[::5]
        x = bz_stochastic.x_u[::5]
        H = build_observation_operator(x, stat_bz.V)

        stat_bz.rho = 1e-3
        stat_bz.sigma = 1e-2

        stat_bz.timestep(y, H)
        assert stat_bz.L_cov.shape == (2178, 128)
        assert stat_bz.mean.shape == (2178, )


@pytest.mark.skip()
def test_statoregonator_inference():
    """ Check marginal posterior gradients are OK.

    Skipped by default as this is slow.
    """
    import h5py

    settings = {"L": 50, "nx": 128, "dt": 1e-3, "correction": True}
    params = {"f": 0.95, "q": 0.002, "eps": 0.75, "Du": 0.001, "Dv": 0.001}

    bz = StochasticOregonator(1e-2, 10., settings, params)

    # set initial conditions from previous simulation
    warmup = h5py.File("outputs/bz-antispiral.h5", "r")
    u_init = warmup["u"][-1, :]
    v_init = warmup["v"][-1, :]
    ic = np.zeros((bz.n_dofs, ))
    ic[bz.u_dofs] = u_init
    ic[bz.v_dofs] = v_init
    bz.setup_solve(ic)

    stat_bz = StatOregonator(128,
                             128,
                             1.,
                             10.,
                             settings=settings,
                             params=params)
    stat_bz.setup_solve(ic)
    rho, sigma = np.random.uniform(0, 1e-2, size=(2, ))

    # conditioning parameters
    obs_skip = 20
    x = bz.x_u[::obs_skip]
    H = build_observation_operator(x, stat_bz.V)

    # mimic a single timestep
    stat_bz.solver.solve()
    if stat_bz.correction:
        stat_bz.concentration_correction()

    stat_bz.mean = np.copy(stat_bz.w.vector()[:])
    J_prev_scipy = dolfin_to_csr(assemble(stat_bz.J_prev))
    J_LU = splu(stat_bz.J_scipy)
    stat_bz.L_temp[:, 0:stat_bz.k] = J_LU.solve(J_prev_scipy @ stat_bz.L_cov)
    stat_bz.L_temp[:, stat_bz.k:] = J_LU.solve(stat_bz.L_G_base)

    H_L_cov = H @ stat_bz.L_temp[:, 0:stat_bz.k]
    H_L_G = H @ stat_bz.L_temp[:, stat_bz.k:]

    mean_obs = H @ stat_bz.mean
    C_obs = H_L_cov @ H_L_cov.T
    G_base_obs = H_L_G @ H_L_G.T

    bz.timestep()
    w = bz.w.vector()[:]
    y = np.copy(bz.u)[::obs_skip]
    np.testing.assert_allclose(H @ w, y)
    y += np.random.normal(scale=1e-4, size=y.shape)

    def compute_grad_error(point, direction, eps=1e-8):
        """ Rel. gradient error from a finite-diff approximation. """
        lp, grad = stat_bz.log_marginal_posterior(point, y, H, mean_obs, C_obs,
                                                  G_base_obs)
        grad = grad @ direction

        point_fwd = point + eps * direction
        point_bwd = point - eps * direction
        lp_fwd, _ = stat_bz.log_marginal_posterior(point_fwd, y, H, mean_obs,
                                                   C_obs, G_base_obs)
        lp_bwd, _ = stat_bz.log_marginal_posterior(point_bwd, y, H, mean_obs,
                                                   C_obs, G_base_obs)

        grad_fd = (lp_fwd - lp_bwd) / (2 * eps)
        return (np.abs(grad_fd - grad) / np.abs(grad))

    # check gradients are approximately equal
    rel_tol = 1e-8
    point = np.array([5e-2, 1e-3])
    direction = np.array([1., 1.])
    rel_error = compute_grad_error(point=point, direction=direction)
    print(rel_error)
    assert rel_error <= rel_tol

    lp, grad = stat_bz.log_marginal_posterior(point, y, H, mean_obs, C_obs,
                                              G_base_obs)
