import h5py

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input_file_cn", type=str)
parser.add_argument("--input_file_imex", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

with h5py.File(args.input_file_cn, "r") as f:
    u_norm_cn = f["u_norm"][:]
    eff_rank_cn = f["eff_rank"][:]

with h5py.File(args.input_file_imex, "r") as f:
    u_norm_imex = f["u_norm"][:]
    eff_rank_imex = f["eff_rank"][:]

# plot the norms
fig, axs = plt.subplots(2, 1,
                        constrained_layout=True,
                        figsize=(3.5, 4.5),
                        sharex=True)
axs = axs.flatten()
axs[0].semilogy(u_norm_cn, label="CN")
axs[0].semilogy(u_norm_imex, label="IMEX")

axins = axs[0].inset_axes([0.5, 0.25, 0.4, 0.35])
axins.semilogy(u_norm_cn[:20], "o", label="CN")
axins.semilogy(u_norm_imex[:20], label="IMEX")
axs[0].indicate_inset_zoom(axins, alpha=0.5, edgecolor="black")
axs[0].set_ylabel(r"$\Vert \mathbf{m}_u^n \Vert$")
axs[0].legend(loc="upper right")

axs[1].plot(eff_rank_cn, label="CN")
axs[1].plot(eff_rank_imex, label="IMEX")
axs[1].plot(np.ones_like(eff_rank_cn), "--", alpha=0.5, color="black")
axs[1].set_xlabel("Timestep")
axs[1].set_ylabel(r"$D_\mathrm{eff}$")
plt.savefig(args.output_dir + "norm-eff-rank.pdf")
plt.close()
