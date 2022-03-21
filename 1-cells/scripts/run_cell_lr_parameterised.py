import h5py
import logging

import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from scipy.sparse import vstack
from multiprocessing import Pool

from statbz.cell import StatCell, StatCellLowRank
from statbz.utils import build_observation_operator, write_csr_matrix_hdf5

from format_cell_data import read_cell_data

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

norm = np.linalg.norm


def run_lr_filter(n_modes, n_modes_prior, sigma_y):
    settings = {"L": 1300, "nx": 200, "dt": 0.1}
    params = {"D": 700, "ku": 0.025, "kv": 0.0725, "sigma_y": sigma_y}

    # GP hyperparameters: var = 2e-3, length = 100., k' = 32
    post = StatCell(2e-3, 100., settings, params)
    post.setup_solve()

    post_lr = StatCellLowRank(2e-3, 100., settings, params, n_modes,
                              n_modes_prior)
    post_lr.setup_solve()

    dat = read_cell_data(args.data_file)

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

    # run statFEM loop
    for i in range(nt):
        t += post_lr.dt

        if i == 0 or np.any(np.isclose(t, t_obs)):
            print(f"\n observing data at iter. {i}")
            # condition on data: matches created observation vector
            y_obs = np.concatenate(
                (dat["u"][(idx_data) * 24:(idx_data + 1) * 24],
                 dat["v"][(idx_data) * 24:(idx_data + 1) * 24]))
            post.timestep(y_obs, H)
            post_lr.timestep(y_obs, H)
            idx_data += 1
        else:
            y_obs = np.zeros((48, ))
            post.timestep()
            post_lr.timestep()  # no arguments => just a prediction step

    mean = post.u
    var = post.cov[post.u_dofs, post.u_dofs]

    mean_lr = post_lr.u
    cov_lr = post_lr.L_cov @ post_lr.L_cov.T
    var_lr = cov_lr[post_lr.u_dofs, post_lr.u_dofs]

    rel_error_mean = norm(mean - mean_lr) / norm(mean)
    rel_error_var = norm(var - var_lr) / norm(var)

    return [rel_error_mean, rel_error_var]


if __name__ == "__main__":
    modes = [16, 32, 48, 64]
    sigmas = [1e-1, 1e-2, 1e-3, 1e-4]
    filter_settings = [(i + 16, i, j) for i in modes for j in sigmas]

    p = Pool(17)
    outputs = np.array(p.starmap(run_lr_filter, filter_settings))
    print(filter_settings)
    print(outputs)

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    axs[0].loglog(sigmas, outputs[:4, 0], "o-", label=fr"$k' = {modes[0]}$")
    axs[0].loglog(sigmas, outputs[4:8, 0], "o-", label=fr"$k' = {modes[1]}$")
    axs[0].loglog(sigmas, outputs[8:12, 0], "o-", label=fr"$k' = {modes[2]}$")
    axs[0].loglog(sigmas, outputs[12:, 0], "o-", label=fr"$k' = {modes[3]}$")

    axs[1].loglog(sigmas, outputs[:4, 1], "o-")
    axs[1].loglog(sigmas, outputs[4:8, 1], "o-")
    axs[1].loglog(sigmas, outputs[8:12, 1], "o-")
    axs[1].loglog(sigmas, outputs[12:, 1], "o-")

    for ax in axs:
        ax.set_xlabel(r"$\sigma$")

    axs[0].legend()
    axs[0].set_ylabel(r"$\Vert \mathbf{m}_n^u - \mathbf{m}_{n, LR}^u \Vert" +
                      r"/ \Vert \mathbf{m}_n^u \Vert$",
                      fontsize="small")
    axs[1].set_ylabel(
        r"$\Vert \mathrm{var}(\mathbf{u}_n) - \mathrm{var}(\mathbf{u}_{n, LR})\Vert"
        + r"/ \Vert \mathrm{var}(\mathbf{u}_n) \Vert$",
        fontsize="small")
    plt.savefig(args.output_file, dpi=666)
