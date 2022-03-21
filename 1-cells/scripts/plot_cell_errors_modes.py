import h5py

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--n_modes", type=int, nargs="*")
parser.add_argument("--input_files_lr", type=str, nargs="*")
parser.add_argument("--prior_modes", action="store_true")
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

norm = np.linalg.norm

model = h5py.File(args.input_file, "r")
t = model["t"][:]
x_u = model["x_u"][:]
x_v = model["x_v"][:]

u = model["u"][:]
v = model["v"][:]

u_var = model["u_var"][:]
v_var = model["v_var"][:]

##########################
# errors as modes increase
lr_errors_mean = []
lr_errors_var = []
for f in args.input_files_lr:
    output_lr = h5py.File(f, "r")

    u_lr = output_lr["u"][-1, :]
    u_lr_var = output_lr["u_var"][-1, :]

    lr_errors_mean.append(norm(u[-1, :] - u_lr) / norm(u[-1, :]))
    lr_errors_var.append(norm(u_var[-1, :] - u_lr_var) / norm(u_var[-1, :]))

fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 2.5))
axs[0].plot(args.n_modes, lr_errors_mean, ".-")
axs[0].set_yscale("log")
axs[0].set_ylabel(r"$\Vert \mathbf{m}_n^u - \mathbf{m}_{n, LR}^u \Vert" +
                  r"/ \Vert \mathbf{m}_n^u \Vert$",
                  fontsize="small")

axs[1].plot(args.n_modes, lr_errors_var, ".-")
axs[1].set_yscale("log")
axs[1].set_ylabel(
    r"$\Vert \mathrm{var}(\mathbf{u}_n) - \mathrm{var}(\mathbf{u}_{n, LR})\Vert"
    + r"/ \Vert \mathrm{var}(\mathbf{u}_n) \Vert$",
    fontsize="small")

if args.prior_modes:
    axs[0].set_xlabel(r"$k'$ (prior)")
    axs[1].set_xlabel(r"$k'$ (prior)")
else:
    axs[0].set_xlabel(r"$k$")
    axs[1].set_xlabel(r"$k$")

fig.suptitle("Error at final time $t = 60$ h")
plt.savefig(args.output_file, dpi=400)
plt.close()
