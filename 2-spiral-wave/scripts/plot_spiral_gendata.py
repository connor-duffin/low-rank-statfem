import h5py
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def plot_ic():
    u0 = MODEL["w_post"][0, U_DOFS]
    u0_dgp = DATA["dgp"][0, U_DOFS]

    x1, x2 = X[:, 0], X[:, 1]
    vmin, vmax = np.min(u0), np.max(u0)
    fig, axs = plt.subplots(1,
                            2,
                            constrained_layout=True,
                            figsize=(5.5, 2.5),
                            sharex=True,
                            sharey=True)
    im = axs[0].tricontourf(x1, x2, u0, 64, vmin=vmin, vmax=vmax)
    axs[0].set_title(r"$\mathbf{m}_u^0$")
    axs[0].set_ylabel("$x_2$")
    axs[0].set_xlabel("$x_1$")

    im = axs[1].tricontourf(x1, x2, u0_dgp, 64, vmin=vmin, vmax=vmax)
    axs[1].set_title(r"$\mathbf{u}_{\mathrm{true}}^0$")
    axs[1].set_xlabel("$x_1$")

    fig.colorbar(im, ax=axs)
    plt.savefig(FIGURES_DIR + "spiral-initial-conditions.png", dpi=300)
    plt.close()


def plot_ic_3d():
    u0 = MODEL["w_post"][0, U_DOFS]

    x1, x2 = X[:, 0], X[:, 1]
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x1, x2, u0, antialiased=True, edgecolor="none")
    plt.savefig(FIGURES_DIR + "ic-dgp-3d.png")
    plt.close()


def plot_rel_error():
    norm = np.linalg.norm
    u_post = MODEL["w_post"][:, U_DOFS]
    u_dgp = DATA["dgp"][:, U_DOFS]

    v_post = MODEL["w_post"][:, V_DOFS]
    v_dgp = DATA["dgp"][:, V_DOFS]

    sigma = DATA.attrs["sigma"]
    rel_error_u = norm(u_post - u_dgp, axis=1) / norm(u_dgp, axis=1)
    rel_error_v = norm(v_post - v_dgp, axis=1) / norm(v_dgp, axis=1)
    assert rel_error_u.shape[0] == u_post.shape[0]
    assert rel_error_v.shape[0] == v_post.shape[0]

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 3))
    axs[0].plot(T, rel_error_u)
    axs[0].plot(T, sigma * np.ones_like(rel_error_u), "--")
    axs[0].set_xlabel("Time $t$")
    axs[0].set_ylabel(
        r"$ \Vert \mathbf{m}_n^u - \mathbf{u}_{\mathrm{true}}^n \Vert"
        + r"/ \Vert \mathbf{m}_u^n \Vert$")

    axs[1].plot(T, rel_error_v)
    axs[1].plot(T, sigma * np.ones_like(rel_error_v), "--")
    axs[1].set_xlabel("Time $t$")
    axs[1].set_ylabel(
        r"$ \Vert \mathbf{m}_v^n - \mathbf{v}_{\mathrm{true}}^n \Vert"
        + r"/ \Vert \mathbf{m}_v^n \Vert $")
    plt.savefig(FIGURES_DIR + "spiral-rel-error.png", dpi=300)
    plt.close()


def plot_eff_rank():
    eff_rank = MODEL["eff_rank"][:]

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 3))
    ax.plot(T, eff_rank)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$D_{\mathrm{eff}}$")
    plt.savefig(FIGURES_DIR + "spiral-eff-rank.png", dpi=300)
    plt.close()


def plot_post_mean_data(idx):
    ui_post = MODEL["w_post"][idx, U_DOFS]
    vi_post = MODEL["w_post"][idx, V_DOFS]
    yi = DATA["y"][idx, :]
    x_obs = DATA["x_obs"][:]
    fig, axs = plt.subplots(1,
                            3,
                            constrained_layout=True,
                            figsize=(6.5, 2),
                            sharey=True)
    im = axs[0].tricontourf(X[:, 0], X[:, 1], ui_post, 64)
    plt.colorbar(im, ax=axs[0])
    axs[0].set_ylabel(r"$x_2$")
    axs[0].set_xlabel(r"$x_1$")
    axs[0].set_title(r"$\mathbf{m}_u^n$")

    im = axs[1].tricontourf(X[:, 0], X[:, 1], vi_post, 64)
    plt.colorbar(im, ax=axs[1])
    axs[1].set_xlabel(r"$x_1$")
    axs[1].set_title(r"$\mathbf{m}_v^n$")

    im = axs[2].scatter(x_obs[:, 0], x_obs[:, 1], c=yi, marker=".")
    plt.colorbar(im, ax=axs[2])
    axs[2].set_xlabel(r"$x_1$")
    axs[2].set_title(r"$\mathbf{y}_n$")
    plt.savefig(FIGURES_DIR + f"spiral-post-data-means-{idx}.png", dpi=300)
    plt.close()


def plot_post_var(idx):
    L_post = MODEL["L_post"][idx, :, :]
    var = np.sum(L_post**2, axis=1)
    x1, x2 = X[:, 0], X[:, 1]

    var_u = var[U_DOFS]
    var_v = var[V_DOFS]

    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True, useOffset=True)
    formatter.set_powerlimits((-2, 2))

    fig, axs = plt.subplots(1,
                            2,
                            constrained_layout=True,
                            figsize=(5.5, 2),
                            sharex=True,
                            sharey=True)
    im = axs[0].tricontourf(x1, x2, var_u, 64)
    axs[0].set_title(r"$\mathrm{var}(\mathbf{u}^n)$")
    axs[0].set_ylabel("$x_2$")
    axs[0].set_xlabel("$x_1$")
    fig.colorbar(im, ax=axs[0], format=formatter)

    im = axs[1].tricontourf(x1, x2, var_v, 64)
    axs[1].set_title(r"$\mathrm{var}(\mathbf{v}^n)$")
    axs[1].set_xlabel("$x_1$")
    fig.colorbar(im, ax=axs[1], format=formatter)

    plt.savefig(FIGURES_DIR + f"spiral-post-var-{idx}.png", dpi=400)
    plt.close()


def plot_modes(idx, colorbar=False, n_modes=8):
    L_ui_post = MODEL["L_post"][idx, U_DOFS, 0:n_modes]
    x1, x2 = X[:, 0], X[:, 1]
    fig, axs = plt.subplots(2,
                            n_modes // 2,
                            constrained_layout=True,
                            figsize=(11.5, 4),
                            sharex=True,
                            sharey=True)
    axs = axs.flatten()

    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True, useOffset=True)
    formatter.set_powerlimits((-2, 2))

    for i in range(n_modes):
        im = axs[i].tricontourf(x1, x2, L_ui_post[:, i], 64, cmap="coolwarm")
        axs[i].set_title(fr"$\mathbf{{L}}_{{n, [:, {i + 1}]}}$")
        if colorbar:
            plt.colorbar(im, ax=axs[i], format=formatter)

    plt.savefig(FIGURES_DIR + f"spiral-post-modes-{idx}.png", dpi=300)
    plt.close()


def plot_observations_mesh():
    fig, ax = plt.subplots(constrained_layout=True, figsize=(3.5, 3))
    ax.triplot(X[:, 0], X[:, 1], "-", label="FEM mesh")
    ax.triplot(X_OBS[:, 0],
               X_OBS[:, 1],
               ".",
               color="orange",
               label="Obs. locations")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    # inset axes
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.triplot(X[:, 0], X[:, 1], "-")
    axins.triplot(X_OBS[:, 0], X_OBS[:, 1], ".", color="orange")
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    axins.set_xlim(-0.5, 6.)
    axins.set_ylim(-0.5, 6.)

    ax.indicate_inset_zoom(axins, alpha=1., edgecolor="black")
    plt.savefig(FIGURES_DIR + "spiral-observations-mesh.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    # set global vars
    DATA = h5py.File(args.data_file, "r")
    MODEL = h5py.File(args.input_file, "r")
    FIGURES_DIR = args.output_dir
    pathlib.Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    T = DATA["t"][:]
    X = MODEL["x"][:]
    X_OBS = MODEL["x_obs"][:]

    U_DOFS = MODEL["u_dofs"][:]
    V_DOFS = MODEL["v_dofs"][:]

    # plot the things
    plot_ic()
    plot_eff_rank()
    plot_rel_error()

    idx_plot = len(T) // 2
    plot_post_mean_data(idx_plot)
    plot_post_var(idx_plot)
    plot_modes(idx_plot, colorbar=True, n_modes=8)

    # plot_ic_3d(model, data)
    # plot_observations_mesh()

    # for i in range(len(t)):
