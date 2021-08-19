import h5py
import logging

import numpy as np
import matplotlib.pyplot as plt

from statbz.oregonator import Oregonator
from statbz.utils import build_observation_operator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# setup the run
sigma = 1e-2
settings = {"L": 50., "nx": 128, "dt": 1e-3,
            "correction": True, "scheme": "crank-nicolson"}
params = {"f": 2, "q": 0.002, "eps": 0.02, "Du": 1., "Dv": 0.6}
logger.info(params)
logger.info(settings)
bz = Oregonator(settings, params)


with h5py.File("data/bz-spiral-ic.h5", "r") as f:
    np.testing.assert_allclose(f["x"][:], bz.x_u)
    ic = np.zeros((bz.n_dofs, ))
    ic[bz.u_dofs] = np.copy(f["u"][:])
    ic[bz.v_dofs] = np.copy(f["v"][:])

    bz.setup_solve(ic)


t = 0
nt = 10_000
nt_obs_interval = 5  # observe data every 5 timesteps
nt_obs = len([i for i in range(nt) if i % nt_obs_interval == 0])
output = h5py.File("data/bz-spiral-data.h5", "w")

x_obs = bz.x_v[::16]
nx_obs = x_obs.shape[0]  # approx 1024 locations
H = build_observation_operator(x_obs, bz.V, sub=1)

# copy runtime settings
for key, value in {**settings, **params}.items():
    output.attrs[key] = value

output.attrs["observed_component"] = "v"
output.attrs["sigma"] = sigma

output.create_dataset("x_obs", data=x_obs)
t_out = output.create_dataset("t", shape=(nt_obs, ))
y_out = output.create_dataset("y", shape=(nt_obs, nx_obs))
dgp_out = output.create_dataset("dgp", shape=(nt_obs, bz.n_dofs))

i_out = 0
logger.info("starting run")
for i in range(nt):
    t += bz.dt
    try:
        bz.timestep()
    except:
        logger.info("error in solver --- breaking timestepping and exiting")
        output.close()
        raise

    if i % nt_obs_interval == 0:
        logger.info("iteration %d of %d", i, nt)
        w_array = bz.w.vector()[:]
        noise = sigma * np.random.normal(size=())
        y = H @ w_array + noise

        dgp_out[i_out, :] = w_array
        y_out[i_out, :] = y
        t_out[i_out] = t

        i_out += 1

    if i % 100 == 0:
        x = bz.x_u
        u, v = bz.u, bz.v
        fig, ax = plt.subplots(1, 2,
                               constrained_layout=True,
                               figsize=(6.5, 3),
                               sharey=True)
        im = ax[0].tricontourf(x[:, 0], x[:, 1], u, 64)
        plt.colorbar(im, ax=ax[0])
        ax[0].set_ylabel(r"$x_2$")
        ax[0].set_xlabel(r"$x_1$")
        ax[0].set_title(r"$u(x_1, x_2, 0)$")

        im = ax[1].tricontourf(x[:, 0], x[:, 1], v, 64)
        plt.colorbar(im, ax=ax[1])
        ax[1].set_xlabel(r"$x_1$")
        ax[1].set_title(r"$v(x_1, x_2, 0)$")
        plt.savefig(f"figures/gendata/dgp-{i}.png")
        plt.close()

output.close()
