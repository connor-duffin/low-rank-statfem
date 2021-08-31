"""Run the cell example, with the low-rank Extended Kalman Filter . """
import h5py
import logging

import numpy as np

from scipy.sparse import vstack

from argparse import ArgumentParser

from statbz.cell import Cell, StatCellLowRank
from statbz.utils import build_observation_operator, write_csr_matrix_hdf5

from format_cell_data import read_cell_data

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--n_modes", type=int)
parser.add_argument("--data_file", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

settings = {"L": 1300, "nx": 200, "dt": 0.1}
params = {"D": 700, "ku": 0.025, "kv": 0.0725, "sigma_y": 1e-2}

cell = Cell(settings, params)
cell.setup_solve()

# GP hyperparameters: var = 2e-3, length = 100., k' = 32
post_lr = StatCellLowRank(2e-3, 100., settings, params, args.n_modes, 32)
post_lr.setup_solve()

dat = read_cell_data(args.data_file)

x_u = post_lr.x_u
x_v = post_lr.x_v
t_obs = dat["t"].unique()
x_obs = dat.index.values[:24]
np.testing.assert_allclose(t_obs, [0., 16., 32., 48.])

# observations come in concatenated vector
Hu = build_observation_operator(x_obs[:, np.newaxis], post_lr.V, sub=0)
Hv = build_observation_operator(x_obs[:, np.newaxis], post_lr.V, sub=1)
H = vstack([Hu, Hv])
assert H.shape == (48, 402)

t = 0.
nt = 600
idx_data = 0

# output storage setup
output = h5py.File(args.output_file, "w")

metadata = {**settings, **params}
for name, val in metadata.items():
    output.attrs.create(name, val)

output.create_dataset("x_u", data=post_lr.x_u)
output.create_dataset("x_v", data=post_lr.x_v)
output.create_dataset("t",
                      data=np.array([(i + 1) * post_lr.dt for i in range(nt)]))

u_output = output.create_dataset("u", shape=(nt, post_lr.n_u_dofs))
u_var_output = output.create_dataset("u_var", shape=(nt, post_lr.n_u_dofs))
u_prior_output = output.create_dataset("u_prior", shape=(nt, post_lr.n_u_dofs))

v_output = output.create_dataset("v", shape=(nt, post_lr.n_v_dofs))
v_var_output = output.create_dataset("v_var", shape=(nt, post_lr.n_v_dofs))
v_prior_output = output.create_dataset("v_prior", shape=(nt, post_lr.n_v_dofs))

y_output = output.create_dataset("y", shape=(nt, 2 * x_obs.shape[0]))
t_obs = output.create_dataset("t_obs", data=t_obs)
x_obs = output.create_dataset("x_obs", data=x_obs)
write_csr_matrix_hdf5(H, "H", output)

# run statFEM loop
for i in range(nt):
    t += cell.dt
    cell.timestep()

    if i == 0 or np.any(np.isclose(t, t_obs)):
        print(f"\n observing data at iter. {i}")
        # condition on data: matches created observation vector
        y_obs = np.concatenate((dat["u"][(idx_data) * 24:(idx_data + 1) * 24],
                                dat["v"][(idx_data) * 24:(idx_data + 1) * 24]))
        post_lr.timestep(y_obs, H)
        idx_data += 1
    else:
        y_obs = np.zeros((48, ))
        post_lr.timestep()  # no arguments => just a prediction step

    # store outputs
    cov = post_lr.L_cov @ post_lr.L_cov.T
    u_prior_output[i, :] = cell.u
    u_output[i, :] = post_lr.u
    u_var_output[i, :] = cov[post_lr.u_dofs, post_lr.u_dofs]

    v_prior_output[i, :] = cell.v
    v_output[i, :] = post_lr.v
    v_var_output[i, :] = cov[post_lr.v_dofs, post_lr.v_dofs]

    y_output[i, :] = y_obs

output.close()
