import h5py
import logging

import numpy as np

from scipy.sparse import vstack

from argparse import ArgumentParser

from statbz.cell import Cell, StatCell
from statbz.utils import build_observation_operator, write_csr_matrix_hdf5

from format_cell_data import read_cell_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

settings = {"L": 1300, "nx": 200, "dt": 0.1}
params = {"D": 700, "ku": 0.025, "kv": 0.0725, "sigma_y": 1e-2}

cell = Cell(settings, params)
cell.setup_solve()

stat_cell = StatCell(2e-3, 100., settings, params)
stat_cell.setup_solve()

dat = read_cell_data(args.data_file)

x_u = stat_cell.x_u
x_v = stat_cell.x_v
t_obs = dat["t"].unique()
x_obs = dat.index.values[:24]
np.testing.assert_allclose(t_obs, [0., 16., 32., 48.])

Hu = build_observation_operator(x_obs[:, np.newaxis], stat_cell.V, sub=0)
Hv = build_observation_operator(x_obs[:, np.newaxis], stat_cell.V, sub=1)
H = vstack([Hu, Hv])
assert H.shape == (48, 402)

t = 0.
nt = 600
idx_data = 0

# output setup
output = h5py.File(args.output_file, "w")

metadata = {**settings, **params}
for name, val in metadata.items():
    output.attrs.create(name, val)

output.create_dataset("x_u", data=stat_cell.x_u)
output.create_dataset("x_v", data=stat_cell.x_v)
output.create_dataset("t",
                      data=np.array([(i + 1) * stat_cell.dt
                                     for i in range(nt)]))

u_output = output.create_dataset("u", shape=(nt, stat_cell.n_u_dofs))
u_var_output = output.create_dataset("u_var", shape=(nt, stat_cell.n_u_dofs))
u_prior_output = output.create_dataset("u_prior",
                                       shape=(nt, stat_cell.n_u_dofs))

v_output = output.create_dataset("v", shape=(nt, stat_cell.n_v_dofs))
v_var_output = output.create_dataset("v_var", shape=(nt, stat_cell.n_v_dofs))
v_prior_output = output.create_dataset("v_prior",
                                       shape=(nt, stat_cell.n_v_dofs))

y_output = output.create_dataset("y", shape=(nt, 2 * x_obs.shape[0]))
t_obs = output.create_dataset("t_obs", data=t_obs)
x_obs = output.create_dataset("x_obs", data=x_obs)
write_csr_matrix_hdf5(H, "H", output)

for i in range(nt):
    t += cell.dt
    print(f"\r Time {t:.2f} / {nt * cell.dt:.2f}", end="")
    cell.timestep()

    if i == 0 or np.any(np.isclose(t, t_obs)):
        print(f"\n observing data at iter. {i}")
        # condition on data
        y_obs = np.concatenate((dat["u"][(idx_data) * 24:(idx_data + 1) * 24],
                                dat["v"][(idx_data) * 24:(idx_data + 1) * 24]))
        stat_cell.timestep(y_obs, H)
        idx_data += 1
    else:
        # march forward
        y_obs = np.zeros((48, ))
        stat_cell.timestep()

    u_prior_output[i, :] = cell.u
    u_output[i, :] = stat_cell.u
    u_var_output[i, :] = stat_cell.cov[stat_cell.u_dofs, stat_cell.u_dofs]

    v_prior_output[i, :] = cell.v
    v_output[i, :] = stat_cell.v
    v_var_output[i, :] = stat_cell.cov[stat_cell.v_dofs, stat_cell.v_dofs]

    y_output[i, :] = y_obs

output.close()
print("")
