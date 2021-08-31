"""Run LR-ExKF for the oscillatory example, mismatched ICs. """
import h5py
import logging

import numpy as np

from pyDOE import lhs
from argparse import ArgumentParser
from statbz.oregonator import Oregonator, StatOregonator
from statbz.utils import build_observation_operator

np.random.seed(27)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def perturbation(x1, x2, eps, L):
    return eps * (1 + np.sin(np.pi * x1 / L + np.random.normal()) *
                  np.sin(np.pi * x2 / L + np.random.normal()))


parser = ArgumentParser()
parser.add_argument("--rho", type=float)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()


RHO = args.rho
OUTPUT_FILE = args.output_file

# simulation settings (as expected)
N_OBS, NT = 512, 1000
N_MODES, N_MODES_G = 128, 64
ELL, SIGMA = 2e-4, 10., 1e-2
PARAMS = {"f": 0.95, "q": 0.002, "eps": 0.75, "Du": 0.001, "Dv": 0.001}
SETTINGS = {
    "L": 50,
    "nx": 256,
    "dt": 1e-2,
    "error_variable": "u",
    "correction": True,
    "scheme": "crank-nicolson",
    "keops_gp": True,
    "approx_gp": False
}

# set initial conditions from previous simulation
dgp = Oregonator(SETTINGS, PARAMS)
with h5py.File(f"data/bz-antispiral-ic-256.h5", "r") as f:
    x_init = np.copy(f["x"][:])
    u_init = np.copy(f["u"][:])
    v_init = np.copy(f["v"][:])

    np.testing.assert_allclose(x_init, dgp.x_u)  # sanity check

    ic = np.zeros((dgp.n_dofs, ))
    ic[dgp.u_dofs] = u_init
    ic[dgp.v_dofs] = v_init

    u_jitter = perturbation(x_init[:, 0], x_init[:, 1], 0.02, SETTINGS["L"])
    ic_jitter = np.copy(ic)
    ic_jitter[dgp.u_dofs] += u_jitter

dgp.setup_solve(ic)
prior = Oregonator(SETTINGS, PARAMS)
prior.setup_solve(ic_jitter)

# set the variance to scale=1. in case running parameter estimation
post = StatOregonator(N_MODES, N_MODES_G, 1., ELL, SETTINGS, PARAMS)
post.setup_solve(ic_jitter)
post.set_hparam_inits(RHO, SIGMA, fixed_sigma=False)

# required for the observation operator
if SETTINGS["error_variable"] == "u":
    error_subspace = 0
else:
    error_subspace = 1

# set up observation grid
x_lhs = SETTINGS["L"] * lhs(2, N_OBS, "m")
H = build_observation_operator(x_lhs, post.V, sub=error_subspace)
logger.info("H.shape (obs. operator) = %s", H.shape)

# sanity check for interpolation of obs on diff grid
temp = np.array([0., 0.])
w_obs_np = H @ (post.w.vector()[:])
for i in range(H.shape[0]):
    post.w.eval(temp, x_lhs[i, :])
    np.testing.assert_approx_equal(w_obs_np[i], temp[error_subspace])

# output storage
output = h5py.File(OUTPUT_FILE, "w")
logger.info("saving output to %s", output)

metadata = {**SETTINGS, **PARAMS}
for name, val in metadata.items():
    output.attrs.create(name, val)

output.create_dataset("x", data=post.x_u)
output.create_dataset("t",
                      data=np.array([(i + 1) * post.dt for i in range(NT)]))

u_output = output.create_dataset("u", shape=(NT, post.n_u_dofs))
var_u_output = output.create_dataset("u_var", shape=(NT, post.n_u_dofs))
u_dgp_output = output.create_dataset("u_dgp", shape=(NT, post.n_u_dofs))
u_prior_output = output.create_dataset("u_prior", shape=(NT, post.n_u_dofs))

v_output = output.create_dataset("v", shape=(NT, post.n_v_dofs))
var_v_output = output.create_dataset("v_var", shape=(NT, post.n_v_dofs))
v_dgp_output = output.create_dataset("v_dgp", shape=(NT, post.n_v_dofs))
v_prior_output = output.create_dataset("v_prior", shape=(NT, post.n_v_dofs))

k_out = 10
L_output = output.create_dataset("L", shape=(NT, post.n_dofs, k_out))

G_vals_output = output.create_dataset("G_vals", data=post.G_vals)
eff_rank_output = output.create_dataset("eff_rank", shape=(NT, ))
params_output = output.create_dataset("params", shape=(NT, 2))

y_output = output.create_dataset("y", shape=(NT, x_lhs.shape[0]))
x_obs = output.create_dataset("x_obs", data=x_lhs)

u_dofs_output = output.create_dataset("u_dofs", data=post.u_dofs)
v_dofs_output = output.create_dataset("v_dofs", data=post.v_dofs)

for i in range(NT):  # run statFEM timestepping
    logger.info("Iteration %d / %d", i + 1, NT)
    dgp.timestep()
    prior.timestep()

    # data is observed at all timesteps
    w_vec = np.copy(dgp.w.vector()[:])
    y = H @ w_vec
    y += np.random.normal(scale=SIGMA, size=y.shape)
    post.timestep(y, H, estimate_params=False)

    u_filter = np.copy(post.u)
    u_prior = np.copy(prior.u)
    u_dgp = np.copy(dgp.u)
    logger.info("L^2 norm difference between filter and DGP: %e",
                np.linalg.norm(u_filter - u_dgp) / np.linalg.norm(u_dgp))
    logger.info("L^2 norm difference between prior and DGP: %e",
                np.linalg.norm(u_prior - u_dgp) / np.linalg.norm(u_dgp))

    # store outputs from the run
    u_output[i, :] = np.copy(post.u)
    u_prior_output[i, :] = np.copy(prior.u)
    u_dgp_output[i, :] = np.copy(dgp.u)

    v_output[i, :] = np.copy(post.v)
    v_prior_output[i, :] = np.copy(prior.v)
    v_dgp_output[i, :] = np.copy(dgp.v)

    L_output[i, :, :] = post.L_cov[:, 0:k_out]
    var_temp = np.sum(post.L_cov**2, axis=1)
    var_u_output[i, :] = np.copy(var_temp.flatten()[post.u_dofs])
    var_v_output[i, :] = np.copy(var_temp.flatten()[post.v_dofs])

    eff_rank_output[i] = post.eff_rank
    params_output[i, :] = post.rho, post.sigma
    y_output[i, :] = np.copy(y)

output.close()
