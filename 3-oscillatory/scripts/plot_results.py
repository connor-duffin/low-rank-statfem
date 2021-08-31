import os
import h5py

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def plot_field_panel(x, t, u, idx, filename):
    """ Plot the field u in a panel over a set of indices idx. """
    assert len(idx) == 4

    color_min = min([u[i, :].min() for i in idx])
    color_max = max([u[i, :].max() for i in idx])

    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(8, 2))
    for i in range(4):
        im = axs[i].tricontourf(x[:, 0],
                                x[:, 1],
                                u[idx[i], :],
                                64,
                                vmin=color_min,
                                vmax=color_max)
        axs[i].set_title(f"t = {t[idx[i]]:.1f}")
        axs[i].set_xlabel(r"$x$")

        if i == 0:
            axs[i].set_ylabel(r"$y$")
        elif i == 3:
            plt.colorbar(im, ax=axs[i])

    plt.savefig(filename, dpi=400)
    plt.close()


def plot_error(t, u_dgp, u_prior, u_post, v_dgp, v_prior, v_post, filename):
    assert u_dgp.shape == u_post.shape
    assert u_dgp.shape == u_prior.shape

    norm = np.linalg.norm
    u_error_prior = norm(u_dgp - u_prior, axis=1) / norm(u_dgp, axis=1)
    u_error_post = norm(u_dgp - u_post, axis=1) / norm(u_dgp, axis=1)

    v_error_prior = norm(v_dgp - v_prior, axis=1) / norm(v_dgp, axis=1)
    v_error_post = norm(v_dgp - v_post, axis=1) / norm(v_dgp, axis=1)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 3))
    ax.plot(t, u_error_prior, "--", label="$u$ (prior)")
    ax.plot(t, u_error_post, color="tab:blue", label="$u$ (post)")
    ax.plot(t, v_error_prior, "--", label="$v$ (prior)")
    ax.plot(t, v_error_post, color="tab:orange", label="$v$ (post)")
    ax.legend()
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(
        r"$\Vert \mathbf{m}_n - \mathbf{w}_{n, \mathrm{true}} \Vert /" +
        r"\Vert \mathbf{w}_{n, \mathrm{true}} \Vert$")

    plt.savefig(filename, dpi=400)
    plt.close()


def plot_params(t, params, filename):
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 3))
    axs[0].plot(t, params[:, 0], ".", label=r"Est. $\rho_n$")
    axs[0].plot(t, 1e-3 * np.ones_like(t), "--", label=r"True $\rho_n$")
    axs[0].legend()
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"Time $t$")

    axs[1].plot(t, params[:, 1], label=r"Est. $\sigma_n$")
    axs[1].plot(t, 1e-2 * np.ones_like(t), "--", label=r"True $\sigma_n$")
    axs[1].legend()
    axs[1].set_xlabel(r"Time $t$")
    plt.savefig(filename, dpi=400)
    plt.close()


def plot_params_compare(params):
    plt.plot(params[:, 0], params[:, 1], ".")
    plt.xscale("log")
    plt.yscale("log")

    plt.ylabel("rho")
    plt.xlabel("sigma")

    plt.savefig("figures/parameters-comparison.png")
    plt.close()


def plot_observations_mesh(x, x_obs, filename):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(2.5, 2.25))
    ax.triplot(x[:, 0], x[:, 1], "-", label="FEM mesh", lw=0.5)
    ax.triplot(x_obs[:, 0],
               x_obs[:, 1],
               ".",
               color="tab:orange",
               label="Obs. points")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.triplot(x[:, 0], x[:, 1], "-", lw=0.5)
    axins.triplot(x_obs[:, 0], x_obs[:, 1], ".", color="tab:orange")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    axins.set_xlim(-0.5, 6.)
    axins.set_ylim(-0.5, 6.)

    ax.indicate_inset_zoom(axins, alpha=1., edgecolor="black")
    plt.savefig(filename, dpi=400)
    plt.close()


def plot_eff_rank(t, eff_rank, filename):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 3))
    ax.plot(t, eff_rank)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$D_{\mathrm{eff}}$")
    plt.savefig(filename, dpi=400)
    plt.close()


def plot_G_vals(G_vals, filename):
    G_vals = G_vals[::-1]
    fig, ax = plt.subplots(constrained_layout=True, figsize=(3, 2.25))

    color = "tab:blue"
    ax.plot(G_vals, color=color)
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$\lambda_j$", color=color)
    ax.tick_params(axis='y', labelcolor=color)

    color = "tab:orange"
    ax_log = ax.twinx()
    ax_log.semilogy(G_vals, color=color)
    ax_log.set_ylabel(r"$\lambda_j$", color=color)
    ax_log.tick_params(axis='y', labelcolor=color)

    plt.savefig(filename, dpi=400)
    plt.close()


def plot_means_vars(x, t, u_dgp, u_prior, u_post, L, filename):
    x1, x2 = x[:, 0], x[:, 1]
    vmin = np.min([u.min() for u in [u_dgp, u_prior, u_post]])
    vmax = np.max([u.max() for u in [u_dgp, u_prior, u_post]])

    fig = plt.figure(constrained_layout=True, figsize=(15, 3.5))
    gs = fig.add_gridspec(2, 8)

    ax1 = fig.add_subplot(gs[:, 0:2])
    ax1.tricontourf(x1, x2, u_prior, 64, vmin=vmin, vmax=vmax)
    ax1.set_ylabel(r"$x_2$")
    ax1.set_xlabel(r"$x_1$")
    ax1.set_title(r"$m_{n, \mathrm{prior}}^u$")

    ax2 = fig.add_subplot(gs[:, 2:4], sharey=ax1)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.tricontourf(x1, x2, u_dgp, 64, vmin=vmin, vmax=vmax)
    ax2.set_xlabel(r"$x_1$")
    ax2.set_title(r"$u_{\mathrm{true}}^n$")

    ax3 = fig.add_subplot(gs[:, 4:6], sharey=ax1)
    plt.setp(ax3.get_yticklabels(), visible=False)
    im = ax3.tricontourf(x1, x2, u_post, 64, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax3)
    ax3.set_xlabel(r"$x_1$")
    ax3.set_title(r"$m_{n, \mathrm{post}}^u$")

    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True, useOffset=True)
    formatter.set_powerlimits((-2, 2))

    k = 0
    for i in [0, 1]:
        for j in [6, 7]:
            ax_var = fig.add_subplot(gs[i, j])

            if i == 1:
                ax_var.set_xlabel(r"$x_1$")
            else:
                plt.setp(ax_var.get_xticklabels(), visible=False)

            if j == 6:
                ax_var.set_ylabel(r"$x_2$")
            else:
                plt.setp(ax_var.get_yticklabels(), visible=False)

            im = ax_var.tricontourf(x1, x2, L[:, k], 64, cmap="coolwarm")
            plt.colorbar(im, ax=ax_var, format=formatter)
            ax_var.set_title(f"$L_{{n, [:, {k + 1}]}}$")

            k += 1

    plt.savefig(filename, dpi=400)
    plt.close()


def plot_modes(x, L, filename, k=8):
    x1, x2 = x[:, 0], x[:, 1]
    fig, ax = plt.subplots(2,
                           k // 2,
                           constrained_layout=True,
                           sharex=True,
                           sharey=True,
                           figsize=(15, 5))
    ax = ax.flatten()

    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True, useOffset=True)
    formatter.set_powerlimits((-2, 2))

    for i in range(k):
        im = ax[i].tricontourf(x1, x2, L[:, i], 64, cmap="coolwarm")
        plt.colorbar(im, ax=ax[i], format=formatter)

        if ax[i].get_subplotspec().is_last_row():
            ax[i].set_xlabel(r"$x_1$")

        if ax[i].get_subplotspec().is_first_col():
            ax[i].set_ylabel(r"$x_2$")

    plt.savefig(filename, dpi=400)
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    # args.input_file = "outputs/stoch-forcing-imex-small-dt.h5"
    # args.output_dir = "figures/stoch-forcing-imex-small-dt/"

    results = h5py.File(args.input_file, "r")
    assert results.attrs["dt"] in (0.01, 0.0001)
    assert results.attrs["scheme"] in ("imex", "crank-nicolson")
    os.makedirs(args.output_dir, exist_ok=True)

    x, t = results["x"][:], results["t"][:]
    u_var, v_var = results["u_var"][:], results["v_var"][:]
    u_dofs, v_dofs = results["u_dofs"][:], results["v_dofs"][:]

    y = results["y"][:]
    x_obs = results["x_obs"][:]
    plot_observations_mesh(x, x_obs,
                           args.output_dir + "mesh-obs-locations.png")

    params = results["params"][:]
    plot_params(t, params, args.output_dir + "est-params.pdf")

    u_post = results["u"][:]
    u_prior = results["u_prior"][:]
    u_dgp = results["u_dgp"][:]

    v_post = results["v"][:]
    v_prior = results["v_prior"][:]
    v_dgp = results["v_dgp"][:]
    plot_error(t, u_dgp, u_prior, u_post, v_dgp, v_prior, v_post,
               args.output_dir + "error.pdf")

    eff_rank = results["eff_rank"][:]
    plot_eff_rank(t, eff_rank, args.output_dir + "eff-rank.png")

    G_vals = results["G_vals"][:]
    plot_G_vals(G_vals, args.output_dir + "G-vals.pdf")
