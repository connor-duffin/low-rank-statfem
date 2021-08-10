import logging

import fenics as fe
import numpy as np

from scipy.linalg import cholesky, cho_solve, eigh, solve_triangular
from scipy.optimize import minimize
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import splu, eigsh
from scipy.stats import truncnorm

from statbz.covariance import cov_approx, sq_exp_covariance, cov_approx_keops
from statbz.utils import dolfin_to_csr

logger = logging.getLogger(__name__)
fe.set_log_level(20)


class InitialCondition(fe.UserExpression):
    """Initial conditions to generate spiral waves.

    From:
    Jahnke, W., Skaggs, W.E., Winfree, A.T., 1989.
    Chemical vortex dynamics in the Belousov-Zhabotinskii reaction and in the
    two-variable oregonator model.
    J. Phys. Chem. 93, 740–749.
    https://doi.org/10.1021/j100339a047
    """
    def __init__(self, f, q, L):
        self.f = f
        self.q = q
        self.L = L
        super().__init__()

    def eval(self, values, x):
        ss = self.q * (self.f + 1) / (self.f - 1)

        theta = np.arctan2(x[1] - self.L / 2, (x[0] - self.L / 2 + 1e-10))
        if theta < 0:
            theta += 2 * np.pi

        if 0 <= theta < 0.5 and (x[0] - self.L / 2) >= 0:
            values[0] = 0.8
        else:
            values[0] = ss

        values[1] = ss + theta / (8 * np.pi * self.f)

    def value_shape(self):
        return (2, )


class RandomInitialCondition(fe.UserExpression):
    def __init__(self, f, q):
        self.f = f
        self.q = q
        super().__init__()

    def eval(self, values, x):
        values[0] = np.random.uniform(self.q, 0.15)
        values[1] = np.random.uniform(self.q, 0.15)

    def value_shape(self):
        return (2, )


class Oregonator:
    def __init__(self, settings, params):
        self.L = settings["L"]
        self.nx = settings["nx"]
        self.dt = settings["dt"]
        self.correction = settings["correction"]

        if "scheme" not in settings:
            settings["scheme"] = "crank-nicolson"
        elif settings["scheme"] not in ["imex", "crank-nicolson"]:
            logger.error("%s time-stepping scheme not supported")
            raise ValueError

        self.scheme = settings["scheme"]

        self.mesh = fe.RectangleMesh(
            fe.Point(0, 0),
            fe.Point(self.L, self.L),
            self.nx,
            self.nx,
        )
        element = fe.FiniteElement("CG", fe.triangle, 1)
        self.V = fe.FunctionSpace(self.mesh,
                                  fe.MixedElement([element, element]))

        self.u_dofs = np.array(self.V.sub(0).dofmap().dofs())
        self.v_dofs = np.array(self.V.sub(1).dofmap().dofs())
        self.x_u = self.V.tabulate_dof_coordinates()[self.u_dofs, :]
        self.x_v = self.V.tabulate_dof_coordinates()[self.v_dofs, :]
        self.n_dofs = self.V.tabulate_dof_coordinates().shape[0]
        self.n_u_dofs = self.x_u.shape[0]
        self.n_v_dofs = self.v_dofs.shape[0]

        self.phi = fe.TestFunctions(self.V)
        phi_1, phi_2 = self.phi

        self.w = fe.Function(self.V)
        u, v = fe.split(self.w)

        self.w_prev = fe.Function(self.V)
        u_prev, v_prev = fe.split(self.w_prev)

        self.f, self.q, self.eps = (
            params["f"],
            params["q"],
            params["eps"],
        )
        Du, Dv = fe.Constant(params["Du"]), fe.Constant(params["Dv"])
        dt = fe.Constant(self.dt)

        def fu(u, v):
            return (u * (1 - u) - self.f * v * (u - self.q) / (u + self.q))

        def gv(u, v):
            return u - v

        # yapf: disable
        dx = fe.dx
        inner, grad = fe.inner, fe.grad
        if self.scheme == "imex":
            self.F = ((u - u_prev) * phi_1 * dx
                      + dt * inner(Du * grad(u), grad(phi_1)) * dx
                      - dt / self.eps * fu(u_prev, v_prev) * phi_1 * dx
                      + (v - v_prev) * phi_2 * dx
                      + dt * inner(Dv * grad(v), grad(phi_2)) * dx
                      - dt * gv(u_prev, v_prev) * phi_2 * dx)
        else:
            u_half = (u + u_prev) / 2
            v_half = (v + v_prev) / 2
            self.F = ((u - u_prev) * phi_1 * dx
                      + dt * inner(Du * grad(u_half), grad(phi_1)) * dx
                      - dt / self.eps * fu(u_half, v_half) * phi_1 * dx
                      + (v - v_prev) * phi_2 * dx
                      + dt * inner(Dv * grad(v_half), grad(phi_2)) * dx
                      - dt * (gv(u, v) + gv(u_prev, v_prev)) / 2 * phi_2 * dx)
        # yapf: enable
        self.J = fe.derivative(self.F, self.w)
        self.J_prev = fe.derivative(self.F, self.w_prev)

    def setup_solve(self, initial_condition="spiral"):
        """ Set the initial conditions and the abstract solver. """
        if type(initial_condition) == np.ndarray:
            self.set_w_from_vector(initial_condition)
        else:
            if initial_condition == "random":
                ic = RandomInitialCondition(self.f, self.q)
            else:
                ic = InitialCondition(self.f, self.q, self.L)

            self.w.interpolate(ic)
            self.w_prev.interpolate(ic)

        # easier to tweak with this interface
        problem = fe.NonlinearVariationalProblem(self.F, self.w, J=self.J)
        self.solver = fe.NonlinearVariationalSolver(problem)

        prm = self.solver.parameters
        prm["nonlinear_solver"] = "snes"
        prm["snes_solver"]["absolute_tolerance"] = 1e-6
        prm["snes_solver"]["relative_tolerance"] = 1e-3
        prm["snes_solver"]["line_search"] = "bt"

    def timestep(self):
        self.solver.solve()

        if self.correction:
            self.concentration_correction()

        self.w_prev.assign(self.w)

    def concentration_correction(self):
        u = self.w.vector()[self.u_dofs]
        logger.info(f"{np.sum(u < self.q)} dofs below threshold")
        w_copy = self.w.vector()[:]
        small_dofs = self.u_dofs[u < self.q]
        w_copy[small_dofs] = self.q
        self.w.vector()[:] = np.copy(w_copy)

    def set_w_from_vector(self, a):
        assert type(a) == np.ndarray
        self.w.vector()[:] = np.copy(a)
        self.w_prev.vector()[:] = np.copy(a)

    def set_w_prev(self, a=None):
        """ Set w_prev to the np.array a; if a is None then w_prev = w. """
        if a is None:
            a = self.w.vector()[:]

        self.w_prev.vector()[:] = np.copy(a)

    def set_u_from_v(self):
        v_vec = self.w.vector()[self.v_dofs]
        self.w.vector()[self.u_dofs] = np.copy(v_vec)

    def set_v_from_u(self):
        u_vec = self.w.vector()[self.u_dofs]
        self.w.vector()[self.v_dofs] = np.copy(u_vec)

    @property
    def u(self):
        w_vec = self.w.vector()
        return w_vec[self.u_dofs]

    @property
    def v(self):
        w_vec = self.w.vector()
        return w_vec[self.v_dofs]


class StochasticOregonator(Oregonator):
    def __init__(self, scale, ell, settings, params):
        super().__init__(settings=settings, params=params)

        if "error_variable" not in settings:
            settings["error_variable"] = "u"

        self.xi = fe.Function(self.V)
        self.xi.vector()[:] = 0.  # otherwise incorrect behaviour

        root_dt = fe.Constant(np.sqrt(self.dt))
        phi_1, phi_2 = self.phi

        if settings["error_variable"] == "u":
            self.x_xi = self.x_u
            self.xi_dofs = self.u_dofs

            xi_u, _ = fe.split(self.xi)
            self.F += root_dt * xi_u * phi_1 * fe.dx
        elif settings["error_variable"] == "v":
            self.x_xi = self.x_v
            self.xi_dofs = self.v_dofs

            _, xi_v = fe.split(self.xi)
            self.F += root_dt * xi_v * phi_2 * fe.dx
        else:
            logger.error("error variable not recognised")
            raise ValueError

        self.xi_cov = sq_exp_covariance(self.x_xi, scale=scale, ell=ell)
        self.xi_cov_chol = cholesky(self.xi_cov, lower=True)

    def timestep(self):
        z = np.random.normal(size=(self.xi_dofs.shape[0], ))
        xi_sample = self.xi_cov_chol @ z
        self.xi.vector()[self.xi_dofs] = xi_sample

        opposite_dofs = np.ones_like(self.xi.vector()[:], bool)
        opposite_dofs[self.xi_dofs] = 0
        np.testing.assert_allclose(self.xi.vector()[opposite_dofs], 0.)
        super().timestep()


class StatOregonator(Oregonator):
    def __init__(self, n_modes, n_modes_G, scale, ell, settings, params):
        super().__init__(settings=settings, params=params)

        self.n_modes = n_modes
        self.n_modes_G = n_modes_G

        self.mean = np.zeros((self.n_dofs, ))
        self.L_cov = np.zeros((self.n_dofs, self.n_modes))

        self.V_eigen = fe.FunctionSpace(self.mesh, "CG", 1)
        M = self._build_mass_matrix(self.V_eigen)

        if "error_variable" not in settings:
            settings["error_variable"] = "u"

        if settings["error_variable"] == "u":
            x_eigen = np.copy(self.x_u)
            self._build_permutation_matrix(0)
        else:
            x_eigen = np.copy(self.x_v)
            self._build_permutation_matrix(1)

        if "approx_gp" not in settings:
            settings["approx_gp"] = False

        if "keops_gp" not in settings:
            settings["keops_gp"] = False

        if settings["approx_gp"]:
            # approximate EVD
            self.G_vals, self.G_vecs = cov_approx(self.V_eigen,
                                                  self.n_modes_G + 1,
                                                  scale=scale,
                                                  ell=ell,
                                                  bc="Neumann")
            self.G_vecs[:] = M @ self.G_vecs
        elif settings["keops_gp"]:
            # exact EVD, single precision, GPU accelerated
            self.G_vals, self.G_vecs = cov_approx_keops(grid=x_eigen,
                                                        scale=scale,
                                                        ell=ell,
                                                        k=self.n_modes_G)
            self.G_vecs[:] = M @ self.G_vecs
        else:
            # exact EVD, dense covariance
            K = sq_exp_covariance(x_eigen, scale=scale, ell=ell)
            self.G_vals, self.G_vecs = eigsh(K,
                                             self.n_modes_G,
                                             which="LM")
            self.G_vecs[:] = M @ self.G_vecs

        logger.info("G_vals range: %e to %e", self.G_vals[-1], self.G_vals[0])

        self.L_G_base = self.P @ self.G_vecs @ diags(
            np.sqrt(self.G_vals.flatten()))
        self.L_temp = np.zeros(
            (self.n_dofs, self.n_modes + self.n_modes_G))  # store for SVD

        self.priors = {
            "rho_mean": 1,
            "rho_sd": 1,
            "sigma_mean": 0,
            "sigma_sd": 1
        }
        self.fixed_sigma = False  # backwards compatibility

    def set_hparam_inits(self, rho, sigma, fixed_sigma=False):
        self.fixed_sigma = fixed_sigma
        self.rho = rho
        self.sigma = sigma
        logger.info("Set hparams to %e, %e", rho, sigma)

    def log_marginal_posterior(self, params, y, H, mean_obs, C_obs,
                               G_base_obs):
        rho, sigma = params
        priors = self.priors

        S = C_obs + self.dt * rho**2 * G_base_obs
        S[np.diag_indices_from(S)] += sigma**2 + 1e-10  # nugget for pos.def
        S_chol = cholesky(S, lower=True)
        S_chol_inv_diff = solve_triangular(S_chol, y - mean_obs, lower=True)
        log_det = 2 * np.sum(np.log(S_chol.diagonal()))

        alpha = cho_solve((S_chol, True), y - mean_obs)
        S_inv = cho_solve((S_chol, True), np.eye(len(y)))
        S_dee_rho = 2 * self.dt * rho * G_base_obs
        S_dee_sigma = 2 * sigma * np.eye(len(y))

        lower, upper = 0., np.inf
        # yapf: disable
        lp = -(-log_det / 2
               - np.dot(S_chol_inv_diff, S_chol_inv_diff / 2)
               + truncnorm.logpdf(
                   rho,
                   (lower - priors["rho_mean"]) / priors["rho_sd"],
                   (upper - priors["rho_mean"]) / priors["rho_sd"],
                   priors["rho_mean"], priors["rho_sd"])
               + truncnorm.logpdf(
                   sigma,
                   (lower - priors["sigma_mean"]) / priors["sigma_sd"],
                   (upper - priors["sigma_mean"]) / priors["sigma_sd"],
                   priors["sigma_mean"], priors["sigma_sd"]))

        grad = -np.array([
            np.dot(alpha, S_dee_rho @ alpha) / 2
            - np.sum(S_inv.T * S_dee_rho) / 2
            - (rho - priors["rho_mean"]) / priors["rho_sd"]**2,
            np.dot(alpha, S_dee_sigma @ alpha) / 2
            - np.sum(S_inv.T * S_dee_sigma) / 2
            - (sigma - priors["sigma_mean"]) / priors["sigma_sd"]**2
        ])
        # yapf: enable
        return (lp, grad)

    def log_marginal_posterior_fixed_sigma(self, rho, y, H, mean_obs, C_obs,
                                           G_base_obs):
        sigma = self.sigma
        priors = self.priors

        S = C_obs + self.dt * rho**2 * G_base_obs
        S[np.diag_indices_from(S)] += sigma**2 + 1e-10  # nugget for pos.def
        S_chol = cholesky(S, lower=True)
        S_chol_inv_diff = solve_triangular(S_chol, y - mean_obs, lower=True)
        log_det = 2 * np.sum(np.log(S_chol.diagonal()))

        alpha = cho_solve((S_chol, True), y - mean_obs)
        S_inv = cho_solve((S_chol, True), np.eye(len(y)))
        S_dee_rho = 2 * self.dt * rho * G_base_obs

        lower, upper = 0., np.inf
        # yapf: disable
        lp = -(-log_det / 2
               - np.dot(S_chol_inv_diff, S_chol_inv_diff / 2)
               + truncnorm.logpdf(
                   rho,
                   (lower - priors["rho_mean"]) / priors["rho_sd"],
                   (upper - priors["rho_mean"]) / priors["rho_sd"],
                   priors["rho_mean"], priors["rho_sd"]))

        grad = -np.array([
            np.dot(alpha, S_dee_rho @ alpha) / 2
            - np.sum(S_inv.T * S_dee_rho) / 2
            - (rho - priors["rho_mean"]) / priors["rho_sd"]**2
        ])
        # yapf: enable
        return (lp, grad)

    def optimize_lmp(self, current_values, *args):
        """ Wrapper function for optimizing the LMP. """
        if self.fixed_sigma:
            bounds = [(1e-12, None)]
            inits = np.random.uniform(0, 1e-3, size=1)
            lmp = self.log_marginal_posterior_fixed_sigma
        else:
            bounds = 2 * [(1e-12, None)]
            inits = np.random.uniform(0, 1e-3, size=2)
            lmp = self.log_marginal_posterior

        n_optim_runs = 100
        for i in range(n_optim_runs):
            try:
                out = minimize(fun=lmp,
                               x0=inits,
                               args=(*args, ),
                               method="L-BFGS-B",
                               jac=True,
                               bounds=bounds).x
                break
            except np.linalg.LinAlgError:
                logger.info(
                    "cholesky failed -- restarting with jittered inits")
                inits = [i + np.random.uniform(0, 0.01) for i in inits]

        if self.fixed_sigma:
            return (out[0], self.sigma)
        else:
            return out

    def timestep(self, y=None, H=None, estimate_params=False):
        self.solver.solve()
        self.concentration_correction()

        self.J_scipy = dolfin_to_csr(fe.assemble(self.J))
        self.J_prev_scipy = dolfin_to_csr(fe.assemble(self.J_prev))
        self.J_LU = splu(self.J_scipy.tocsc())

        self.mean = np.copy(self.w.vector()[:])
        self.L_temp[:, 0:self.n_modes] = self.J_LU.solve(
            self.J_prev_scipy @ self.L_cov)
        self.L_temp[:, self.n_modes:] = self.J_LU.solve(self.L_G_base)

        if estimate_params is True:
            H_L_cov = H @ self.L_temp[:, 0:self.n_modes]
            H_L_G = H @ self.L_temp[:, self.n_modes:]

            mean_obs = H @ self.mean
            C_obs = H_L_cov @ H_L_cov.T
            G_base_obs = H_L_G @ H_L_G.T

            self.rho, self.sigma = self.optimize_lmp([self.rho, self.sigma], y,
                                                     H, mean_obs, C_obs,
                                                     G_base_obs)
            logger.info("MAP est: rho: %e, sigma: %e", self.rho, self.sigma)

        # scale by the known factors
        self.L_temp[:, self.n_modes:] *= np.sqrt(self.dt) * self.rho

        D, V = eigh(self.L_temp.T @ self.L_temp)
        D, V = D[::-1], V[:, ::-1]
        D_abs = np.abs(D)
        self.eff_rank = np.sum(np.sqrt(D_abs))**2 / np.sum(D_abs)
        logger.info("Effective covariance rank: %f", self.eff_rank)

        # np.dot is faster than np.matmul
        np.dot(self.L_temp, V[:, 0:self.n_modes], out=self.L_cov)
        logger.info("Norm of deleted components: %e", D[self.n_modes])
        logger.info("Prop. variance kept in reduction: %e",
                    np.sum(D[0:(self.n_modes - 1)]) / (np.sum(D)))

        if y is not None and H is not None:
            HL = H @ self.L_cov
            S = HL @ HL.T
            S[np.diag_indices_from(S)] += self.sigma**2 + 1e-10
            S_chol = cholesky(S, lower=True)

            S_inv_y = cho_solve((S_chol, True), y - H @ self.mean)
            self.mean += self.L_cov @ (HL.T @ S_inv_y)

            S_inv_HL = cho_solve((S_chol, True), HL)
            R = cholesky(np.eye(HL.shape[1]) - HL.T @ S_inv_HL, lower=True)

            self.L_cov = self.L_cov @ R
            self.w.vector()[:] = np.copy(self.mean)

        self.concentration_correction()
        self.w_prev.assign(self.w)

    def _build_permutation_matrix(self, subs=0):
        """Build permutation matrix that maps from V(subs) to V. """
        if subs == 0:
            dofs_sub = self.u_dofs
        elif subs == 1:
            dofs_sub = self.v_dofs
        else:
            logger.error("incorrect subspace for the permutation matrix")
            raise ValueError

        n_dofs_sub = len(dofs_sub)

        self.P = lil_matrix((self.n_dofs, n_dofs_sub))
        row_indices, col_indices = dofs_sub, list(range(n_dofs_sub))
        self.P[row_indices, col_indices] = 1.
        self.P = self.P.tocsr()

    def _build_mass_matrix(self, V):
        """Build the FEM mass matrix (Neumann BCs) on V. """
        u = fe.TrialFunction(V)
        v = fe.TestFunction(V)
        M = fe.assemble(u * v * fe.dx)
        return dolfin_to_csr(M)
