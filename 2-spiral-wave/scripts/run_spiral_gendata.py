"""Run the spiral example, using the low-rank Extended Kalman Filter. """
import h5py
import logging

import numpy as np
import fenics as fe

from argparse import ArgumentParser
from statbz.oregonator import Oregonator, StatOregonator
from statbz.utils import build_observation_operator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def blur(mesh, u0_vec, dt=0.1):
    """Blur u0 using the heat equation: assumes CG1 elements. """
    V = fe.FunctionSpace(mesh, "CG", 1)
    u0 = fe.Function(V)
    assert u0.vector()[:].shape == u0_vec.shape
    u0.vector()[:] = np.copy(u0_vec)

    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    grad, inner, dx = fe.grad, fe.inner, fe.dx
    F = (u - u0) * v * dx + dt * inner(grad(u), grad(v)) * dx
    a, L = fe.lhs(F), fe.rhs(F)
    u = fe.Function(V)
    fe.solve(a == L, u)

    return np.copy(u.vector()[:])


parser = ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

assert args.output_file is not None
logger.info("Saving output to %s", args.output_file)

nt = 10_000
n_modes, n_modes_g = 250, 150
rho, ell, sigma = 1e-3, 5., 1e-2
params = {"f": 2., "q": 0.002, "eps": 0.02, "Du": 1., "Dv": 0.6}
settings = {
    "L": 50,
    "nx": 128,
    "dt": 1e-3,
    "error_variable": "v",
    "correction": True,
    "scheme": "crank-nicolson"
}
logger.info(params)
logger.info(settings)

# set the variance to 1. in case of wanting to do parameter estimation
# (parameter estimation re-uses the factorization)
post = StatOregonator(n_modes, n_modes_g, 1., ell, settings, params)
prior = Oregonator(settings, params)

# set initial conditions
with h5py.File("data/bz-spiral-ic.h5", "r") as f:
    np.testing.assert_allclose(f["x"][:], post.x_u)
    ic = np.zeros((post.n_dofs, ))
    ic[post.u_dofs] = blur(post.mesh, f["u"][:])  # induce mismatch
    ic[post.v_dofs] = np.copy(f["v"][:])

    post.setup_solve(ic)
    prior.setup_solve(ic)

# set hparams to fixed values, for the paper
post.rho, post.sigma = rho, sigma

data = h5py.File(args.data_file, "r")
np.testing.assert_almost_equal(data.attrs["sigma"], sigma)

x_obs = data["x_obs"][:]
t_obs = data["t"][:]
y_obs = data["y"][:]
dgp = data["dgp"][:]
nt_obs = len(t_obs)
logger.info("data loaded from %s, observed %d of %d timesteps", args.data_file,
            nt_obs, nt)

H = build_observation_operator(x_obs, post.V, sub=1)
logger.info(H.shape)

# output processing: set up for storage
output = h5py.File(args.output_file, "w")
logger.info("saving output to %s", output)

for name, val in {**settings, **params}.items():
    output.attrs.create(name, val)

output.create_dataset("t",
                      data=np.array([(i + 1) * post.dt for i in range(nt)]))
output.create_dataset("x", data=post.x_u)
output.create_dataset("x_obs", data=x_obs)
output.create_dataset("u_dofs", data=post.u_dofs)
output.create_dataset("v_dofs", data=post.v_dofs)

w_post = output.create_dataset("w_post", shape=(nt_obs, post.n_dofs))
L_post = output.create_dataset("L_post",
                               shape=(nt_obs, post.n_dofs, post.n_modes))
params = output.create_dataset("params", shape=(nt_obs, 2))
eff_rank = output.create_dataset("eff_rank", shape=(nt_obs, ))

w_post[0, :] = np.copy(post.w_prev.vector()[:])
L_post[0, :, :] = np.copy(post.L_cov)

t = 0
i_obs = 0
for i in range(1, nt):  # run statFEM timestepping
    t += post.dt
    logger.info("Iteration %d / %d", i, nt)

    # check if data is observed
    # first check: stops errors in the final iteration
    # as i_obs is iteratied over
    if (i_obs + 1) > len(t_obs):
        y_curr = None
        H_curr = None
    # data observed at this timestep
    elif np.isclose(t, t_obs[i_obs]):
        y_curr = y_obs[i_obs, :]
        H_curr = H
    # data not observed at this timestep
    else:
        y_curr = None
        H_curr = None

    try:
        prior.timestep()
        post.timestep(y=y_curr, H=H_curr, estimate_params=False)
    except:  # bad practice
        logger.error("timestep diverged --- exiting")
        output.close()
        raise

    # save memory: store outputs ONLY if we observe data (every 5 timesteps)
    if y_curr is not None:
        u_true = dgp[i_obs, post.u_dofs]
        u_curr_prior = prior.u
        u_curr_post = post.u

        norm = np.linalg.norm
        rel_error_prior = norm(u_true - u_curr_prior) / norm(u_true)
        rel_error_post = norm(u_true - u_curr_post) / norm(u_true)

        # sanity checking
        logger.info("Prior rel error: %e", rel_error_prior)
        logger.info("Post rel error: %e", rel_error_post)

        # store outputs
        w_post[i_obs, :] = np.copy(post.w.vector()[:])
        L_post[i_obs, :, :] = np.copy(post.L_cov)
        eff_rank[i_obs] = post.eff_rank
        params[i_obs, :] = post.rho, post.sigma

        i_obs += 1  # increment for next timestep

output.close()
