import h5py
import logging

import numpy as np

from argparse import ArgumentParser
from statbz.oregonator import Oregonator, StatOregonator
from statbz.utils import build_observation_operator

np.random.seed(27)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--scheme", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

# configuration
n_obs, nt = 512, 100
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

if args.scheme is not None:
    settings.update({"scheme": args.scheme})

# model setup
prior = Oregonator(settings, params)
post = StatOregonator(n_modes, n_modes_g, 1., ell, settings, params)

with h5py.File("data/bz-spiral-ic.h5", "r") as f:
    np.testing.assert_allclose(f["x"][:], post.x_u)
    ic = np.zeros((post.n_dofs, ))
    ic[post.u_dofs] = np.copy(f["u"][:])
    ic[post.v_dofs] = np.copy(f["v"][:])

prior.setup_solve(ic)
post.setup_solve(ic)
post.rho, post.sigma = rho, sigma

skip = int(post.n_u_dofs // n_obs)
x_obs = post.x_v[::skip, :]
H = build_observation_operator(x_obs, post.V, sub=1)
logger.info(H.shape)

# timestepping
np.random.seed(27)
u_norm = np.zeros((nt, ))
eff_rank = np.zeros((nt, ))
for i in range(nt):
    logger.info("Iteration %d / %d", i + 1, nt)
    prior.timestep()

    y = H @ (prior.w.vector()[:])
    y += np.random.normal(scale=sigma, size=y.shape)
    post.timestep(y, H, estimate_params=False)

    u_norm[i] = np.linalg.norm(post.u)
    logger.info("||u|| : %e", u_norm[i])
    eff_rank[i] = post.eff_rank

    if np.any(post.u >= 1e4):
        logger.error("Filter divergence, iteration %d", i + 1)

        u_norm = u_norm[:(i + 1)]
        eff_rank = eff_rank[:(i + 1)]
        break

# store results
# output_file = f"outputs/spiral-divergence-{settings['scheme']}.h5"
with h5py.File(args.output_file, "w") as f:
    metadata = {**settings, **params}
    for name, val in metadata.items():
        f.attrs.create(name, val)

    f.create_dataset("u_norm", data=u_norm)
    f.create_dataset("eff_rank", data=eff_rank)
