import h5py

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--n_modes", type=int, nargs="*")
parser.add_argument("--input_files_lr", type=str, nargs="*")
parser.add_argument("--n_mode_plot", type=int, default=32)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

lr_file_plot = args.input_files_lr[args.n_modes.index(args.n_mode_plot)]

model = h5py.File(args.input_file, "r")
model_lr = h5py.File(lr_file_plot, "r")

t = model["t"][:]
x_u = model["x_u"][:]
x_v = model["x_v"][:]

u = model["u"][:]
v = model["v"][:]

u_lr = model_lr["u"][:]
v_lr = model_lr["v"][:]

u_prior = model["u_prior"][:]
v_prior = model["v_prior"][:]

u_var = model["u_var"][:]
v_var = model["v_var"][:]

u_lr_var = model_lr["u_var"][:]
v_lr_var = model_lr["v_var"][:]

x_obs = model["x_obs"][:]
y_obs = model["y"][:]

idx_data = [0, 159, 319, 479]

########
# curves
fig, axs = plt.subplots(2,
                        4,
                        constrained_layout=True,
                        figsize=(9, 5),  # 9:5 aspect ratio
                        sharex=True,
                        sharey=True)
for i_plot, i_array in enumerate(idx_data):
    u_curr = u_lr[i_array, :]
    v_curr = v_lr[i_array, :]

    u_prior_curr = u_prior[i_array, :]
    v_prior_curr = v_prior[i_array, :]

    u_sd = np.sqrt(u_lr_var[i_array, :])
    v_sd = np.sqrt(v_lr_var[i_array, :])

    axs[0, i_plot].set_title(f"t = {t[i_array]:.2f} (h)")
    axs[0, i_plot].plot(x_u, u_curr, label=r"$\mathbf{m}_u^n$")
    axs[0, i_plot].plot(x_obs, y_obs[i_array, :24], "+", color="tab:blue")
    axs[0, i_plot].fill_between(x_u.flatten(),
                                u_curr - 2 * u_sd,
                                u_curr + 2 * u_sd,
                                alpha=0.2)

    axs[0, i_plot].plot(x_v, v_curr, label=r"$\mathbf{m}_v^n$")
    axs[0, i_plot].plot(x_obs, y_obs[i_array, 24:48], "+", color="tab:orange")
    axs[0, i_plot].fill_between(x_v.flatten(),
                                v_curr - 2 * v_sd,
                                v_curr + 2 * v_sd,
                                alpha=0.2)

    i_array += 50
    u_curr = u_lr[i_array, :]
    v_curr = v_lr[i_array, :]

    u_prior_curr = u_prior[i_array, :]
    v_prior_curr = v_prior[i_array, :]

    u_sd = np.sqrt(u_lr_var[i_array, :])
    v_sd = np.sqrt(v_lr_var[i_array, :])

    axs[1, i_plot].set_title(f"t = {t[i_array]:.2f} (h)")
    axs[1, i_plot].plot(x_u, u_curr, label=r"$\mathbf{m}_u^n$")
    axs[1, i_plot].fill_between(x_u.flatten(),
                                u_curr - 2 * u_sd,
                                u_curr + 2 * u_sd,
                                alpha=0.2)

    axs[1, i_plot].plot(x_v, v_curr, label=r"$\mathbf{m}_v^n$")
    axs[1, i_plot].fill_between(x_v.flatten(),
                                v_curr - 2 * v_sd,
                                v_curr + 2 * v_sd,
                                alpha=0.2)

    axs[1, i_plot].plot(x_u, u_prior_curr, label=r"$\mathbf{m}_u^n$ (prior)")
    axs[1, i_plot].plot(x_v, v_prior_curr, label=r"$\mathbf{m}_v^n$ (prior)")
    axs[1, i_plot].set_xlabel(r"$x$ ($\mu$m)")

axs[0, 0].legend()
axs[1, 0].legend()
axs[0, 0].set_ylabel(r"Density")
axs[1, 0].set_ylabel(r"Density")
plt.savefig(args.output_dir + "cell-means.png", dpi=400)
plt.close()

##########
# meshplot
fig, axs = plt.subplots(1,
                        2,
                        constrained_layout=True,
                        sharex=True,
                        sharey=True,
                        figsize=(8, 5))
axs = axs.flatten()
X, T = np.meshgrid(x_u, t)
im = axs[0].pcolormesh(X, T, u)
for time in t[idx_data]:
    axs[0].axhline(time, ls="--", color="black")

plt.colorbar(im, ax=axs[0])
axs[0].set_xlabel(r"$x$ ($\mu$m)")
axs[0].set_ylabel(r"$t$ (h)")
axs[0].set_title(r"$\mathbf{m}_u(x, t)$")

im = axs[1].pcolormesh(X, T, v)
plt.colorbar(im, ax=axs[1])
for time in t[idx_data]:
    axs[1].axhline(time, ls="--", color="black")

axs[1].set_xlabel(r"$x$ ($\mu$m)")
axs[1].set_title(r"$\mathbf{m}_v(x, t)$")

plt.savefig(args.output_dir + "cell-surfaces.png", dpi=400)
plt.close()

######################
# low-rank rel. errors
norm = np.linalg.norm
rel_error_mean = norm(u - u_lr, axis=1) / norm(u, axis=1)
rel_error_var = norm(u_var - u_lr_var, axis=1) / norm(u_var, axis=1)

fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 2.5))
axs[0].plot(t, rel_error_mean)
axs[0].set_xlabel(r"$t$ (h)")
axs[0].set_ylabel(r"$\Vert \mathbf{m}_u^n - \mathbf{m}_{u, LR}^n \Vert" +
                  r"/ \Vert \mathbf{m}_u^n \Vert$",
                  fontsize="small")
axs[0].set_yscale("log")

axs[1].plot(t, rel_error_var)
axs[1].set_xlabel(r"$t$ (h)")
axs[1].set_ylabel(
    r"$\Vert \mathrm{var}(\mathbf{u}^n) - \mathrm{var}(\mathbf{u}_{LR}^n)" +
    r" \Vert / \Vert \mathrm{var}(\mathbf{u}^n) \Vert$",
    fontsize="small")
axs[1].set_yscale("log")
plt.savefig(args.output_dir + "cell-rel-error.png", dpi=400)
plt.close()

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
axs[0].set_ylabel(
    r"$\Vert \mathbf{m}_u^n - \mathbf{m}_{u, LR}^n \Vert"
    + r"/ \Vert \mathbf{m}_u^n \Vert$",
    fontsize="small")
axs[0].set_xlabel(r"$k$ (no. of modes)")

axs[1].plot(args.n_modes, lr_errors_var, ".-")
axs[1].set_yscale("log")
axs[1].set_xlabel(r"$k$ (no. of modes)")
axs[1].set_ylabel(
    r"$\Vert \mathrm{var}(\mathbf{u}^n) - \mathrm{var}(\mathbf{u}_{LR}^n)\Vert"
    + r"/ \Vert \mathrm{var}(\mathbf{u}^n) \Vert$",
    fontsize="small")
fig.suptitle("Error at final time $t = 60$ h")
plt.savefig(args.output_dir + "cell-lr-modes-errors.png", dpi=400)
plt.close()
