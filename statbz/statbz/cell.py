import logging

import fenics as fe
import numpy as np

from scipy.linalg import cholesky, cho_solve, eigh
from scipy.sparse.linalg import splu

from statbz.covariance import sq_exp_covariance
from statbz.utils import dolfin_to_csr

logger = logging.getLogger(__name__)
fe.set_log_level(40)


class InitialCondition(fe.UserExpression):
    def eval(self, value, x):
        if x <= 400 or x >= 900:
            value[0] = 0.055
            value[1] = 0.055

    def value_shape(self):
        return (2, )


class Cell:
    def __init__(self, settings, params):
        self.L = settings["L"]
        self.nx = settings["nx"]
        self.dt = settings["dt"]

        self.mesh = fe.IntervalMesh(self.nx, 0., self.L)
        element = fe.FiniteElement("CG", fe.interval, 1)
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

        self.ku, self.kv = params["ku"], params["kv"]

        ku = fe.Constant(self.ku)
        kv = fe.Constant(self.ku)
        D = fe.Constant(params["D"])
        dt = fe.Constant(self.dt)

        # yapf: disable
        dx = fe.dx
        u_half = (u + u_prev) / 2
        v_half = (v + v_prev) / 2
        self.F = ((u - u_prev) * phi_1 * dx
                  + dt * fe.inner(D * fe.grad(u_half), fe.grad(phi_1)) * dx
                  + dt * ku * u_half * phi_1 * dx
                  - dt * 2 * kv * v_half * (1 - u_half - v_half) * phi_1 * dx
                  # second system
                  + (v - v_prev) * phi_2 * dx
                  + dt * fe.inner(D * fe.grad(v_half), fe.grad(phi_2)) * dx
                  - dt * ku * u_half * phi_2 * dx
                  + dt * kv * v_half * (1 - u_half - v_half) * phi_2 * dx)
        # yapf: enable
        self.J = fe.derivative(self.F, self.w)
        self.J_prev = fe.derivative(self.F, self.w_prev)

    def setup_solve(self, ic="rectangle"):
        """ Set the initial conditions and the abstract solver. """
        ic_rectangle = InitialCondition()
        self.w.interpolate(ic_rectangle)
        self.w_prev.interpolate(ic_rectangle)

        if ic == "perturbed":
            cov = sq_exp_covariance(self.V.tabulate_dof_coordinates(), 1e-2,
                                    100)
            chol_cov = cholesky(cov, lower=True)
            z = np.random.normal(size=self.w.vector()[:].shape)
            w = np.copy(self.w.vector()[:])
            w += chol_cov @ z
            w[w < 0.] = 0.

            self.w.vector()[:] = np.copy(w)
            self.w_prev.vector()[:] = np.copy(w)

        # easier to tweak with this interface
        problem = fe.NonlinearVariationalProblem(self.F, self.w, J=self.J)
        self.solver = fe.NonlinearVariationalSolver(problem)

    def timestep(self):
        self.solver.solve()
        self.w_prev.assign(self.w)

    def set_w_from_vector(self, a):
        assert type(a) == np.ndarray
        self.w.vector()[:] = np.copy(a)
        self.w_prev.vector()[:] = np.copy(a)

    @property
    def u(self):
        w_vec = self.w.vector()
        return w_vec[self.u_dofs]

    @property
    def v(self):
        w_vec = self.w.vector()
        return w_vec[self.v_dofs]


class StatCell(Cell):
    def __init__(self, scale, ell, settings, params):
        """ Initialize with set parameters. """
        super().__init__(settings=settings, params=params)

        self.mean = np.zeros((self.n_dofs, ))
        self.cov = np.zeros((self.n_dofs, self.n_dofs))
        self.cov_hat = np.zeros((self.n_dofs, self.n_dofs))

        self._build_covariance(scale, ell)
        assert self.G.shape == (self.n_dofs, self.n_dofs)

        self.sigma_y = params["sigma_y"]

    def timestep(self, y=None, H=None):
        self.solver.solve()
        self.mean = self.w.vector()[:]

        J_scipy = dolfin_to_csr(fe.assemble(self.J))
        J_scipy_LU = splu(J_scipy.tocsc())
        J_prev_scipy = dolfin_to_csr(fe.assemble(self.J_prev))

        self.cov_hat = J_prev_scipy @ self.cov @ J_prev_scipy.T
        temp = J_scipy_LU.solve(self.cov_hat + self.G)
        self.cov = J_scipy_LU.solve(temp.T)

        if y is not None and H is not None:
            mean_obs = H @ self.mean
            HC = H @ self.cov

            S = H @ self.cov @ H.T
            S[np.diag_indices_from(S)] += self.sigma_y**2
            S_chol = cholesky(S, lower=True)

            S_inv_y = cho_solve((S_chol, True), y - mean_obs)
            S_inv_HC = cho_solve((S_chol, True), HC)

            self.mean += HC.T @ S_inv_y
            self.cov -= HC.T @ S_inv_HC

            self.w.vector()[:] = np.copy(self.mean)

        self.w_prev.assign(self.w)

    def _build_covariance(self, scale, ell):
        """ Build the covariance matrix, for iid GPs on u and v. """
        grid = self.V.tabulate_dof_coordinates()
        K = sq_exp_covariance(grid, np.sqrt(self.dt) * scale, ell)
        M = self._build_mass_matrix(self.V)

        # all dofs where there should be no correlation
        for i in self.u_dofs:
            K[i, self.v_dofs] = 0.

        for i in self.v_dofs:
            K[i, self.u_dofs] = 0.

        self.G = M @ K @ M.T

    def _build_mass_matrix(self, V):
        """ Build the FEM mass matrix on the FunctionSpace V. """
        u = fe.TrialFunction(V)
        v = fe.TestFunction(V)
        M = fe.assemble(fe.inner(u, v) * fe.dx)
        return dolfin_to_csr(M)


class StatCellLowRank(StatCell):
    def __init__(self, scale, ell, settings, params, n_modes, n_modes_G):
        super().__init__(scale=scale,
                         ell=ell,
                         settings=settings,
                         params=params)

        self.n_modes = n_modes
        self.n_modes_G = n_modes_G
        self.G_vals, self.G_vecs = eigh(self.G)
        self.G_vals, self.G_vecs = self.G_vals[::-1], self.G_vecs[:, ::-1]
        logger.info(
            "Prop. variance kept in reduction of G: %f",
            np.sum(self.G_vals[0:self.n_modes_G]) / np.sum(self.G_vals))
        self.L_G = self.G_vecs[:, 0:n_modes_G] @ np.diag(
            np.sqrt(self.G_vals[0:n_modes_G]))

        self.L_cov = np.zeros((self.n_dofs, n_modes))
        self.L_temp = np.zeros((self.n_dofs, n_modes + n_modes_G))

    def timestep(self, y=None, H=None):
        self.solver.solve()
        self.mean = self.w.vector()[:]

        J_scipy = dolfin_to_csr(fe.assemble(self.J))
        J_prev_scipy = dolfin_to_csr(fe.assemble(self.J_prev))
        J_scipy_LU = splu(J_scipy.tocsc())

        self.L_temp[:, 0:self.n_modes] = J_scipy_LU.solve(
            J_prev_scipy @ self.L_cov)
        self.L_temp[:, self.n_modes:] = J_scipy_LU.solve(self.L_G)

        # reduction step
        D, V = eigh(self.L_temp.T @ self.L_temp)
        D, V = D[::-1], V[:, ::-1]
        logger.info("Prop. variance kept in the reduction: %f",
                    np.sum(D[0:self.n_modes]) / np.sum(D))
        np.dot(self.L_temp, V[:, 0:self.n_modes], out=self.L_cov)

        if y is not None and H is not None:
            mean_obs = H @ self.mean
            HL = H @ self.L_cov

            S = HL @ HL.T
            S[np.diag_indices_from(S)] += self.sigma_y**2
            S_chol = cholesky(S, lower=True)

            S_inv_y = cho_solve((S_chol, True), y - mean_obs)
            S_inv_HL = cho_solve((S_chol, True), HL)

            self.mean += self.L_cov @ HL.T @ S_inv_y
            R = cholesky(np.eye(HL.shape[1]) - HL.T @ S_inv_HL, lower=True)
            self.L_cov = self.L_cov @ R

            self.w.vector()[:] = np.copy(self.mean)

        self.w_prev.assign(self.w)
