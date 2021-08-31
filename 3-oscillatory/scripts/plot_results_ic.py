import h5py

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def texp(string, dollar_surround=True):
    if "e" in string:
        string_split = string.split('e')
        exponent = string_split[-1]

        exponent_sign = exponent[0]
        exponent_value = exponent[1:].lstrip("0")
        if exponent_sign == "+":
            exponent_sign = ""

        exponent = f"{{{exponent_sign + exponent_value}}}"

        string_split.insert(1, r" \times 10^")
        string_split[-1] = exponent
        string = ''.join(string_split)

    if dollar_surround:
        return fr"${string}$"
    else:
        return string


def plot_error(t, prior_error, filename, *args):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 3))
    ax.plot(t, prior_error, color="grey", label=r"$m_n^u$ (prior)")
    for error in args:
        ax.plot(t, error)
        # ax.legend(loc="upper right")

    axins = ax.inset_axes([0.5, 0.25, 0.4, 0.35])
    for error in args:
        axins.plot(t, error)

    axins.set_xticks([0., 1., 2.])
    axins.set_xlim(-0.1, 2.)
    # axins.set_yscale("log")
    # axins.set_yticks([])
    # axins.set_xticklabels("")
    # axins.set_yticklabels("")

    ax.indicate_inset_zoom(axins, alpha=1., edgecolor="black")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(
        r"$\Vert \mathbf{m}_n^u - \mathbf{u}_{n, \mathrm{true}} \Vert" +
        r"/ \Vert \mathbf{u}_{n, \mathrm{true}} \Vert$"
    )

    plt.savefig(filename, dpi=400)
    plt.close()


def plot_params(t, params_est, filename):
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(7.5, 3))
    axs[0].plot(t, params_est[:, 0], ".", label=r"Est. $\rho$")
    axs[0].plot(t, 1e-2 * np.ones_like(t), label=r"Fixed $\rho = 10^{-2}$")
    axs[0].plot(t, 1e-3 * np.ones_like(t), label=r"Fixed $\rho = 10^{-3}$")
    axs[0].legend()
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"Time $t$")

    axs[1].plot(t, params_est[:, 1], label=r"Est. $\sigma_n$")
    axs[1].plot(t, 1e-2 * np.ones_like(t), label=r"Fixed $\sigma_n = 10^{-2}$")
    axs[1].legend()
    axs[1].set_xlabel(r"Time $t$")
    plt.savefig(filename, dpi=400)
    plt.close()


def plot_eff_rank(t, filename, *args):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 3))

    for rho, eff_rank in args:
        ax.plot(t, eff_rank, label=fr"$\rho = {rho}$")

    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$D_{\mathrm{eff}}$")
    ax.legend()
    plt.savefig(filename, dpi=400)
    plt.close()


def plot_modes(t, k, L, filename, rho):
    x1, x2 = x[:, 0], x[:, 1]
    fig, ax = plt.subplots(2,
                           k // 2,
                           constrained_layout=True,
                           sharex=True,
                           sharey=True,
                           figsize=(6, 4))

    for a in ax[:, 0]:
        a.set_ylabel(r"$x_2$")

    for a in ax[1, :]:
        a.set_xlabel(r"$x_1$")

    ax = ax.flatten()

    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True, useOffset=True)
    formatter.set_powerlimits((-2, 2))

    for i in range(k):
        im = ax[i].tricontourf(x1, x2, L[:, i], 64, cmap="coolwarm")
        plt.colorbar(im, ax=ax[i], format=formatter)
        ax[i].set_title(f"$\\mathbf{{L}}_{{n, [:, {i + 1}]}}$")

    rho_formatted = texp(f"{rho:.0e}", dollar_surround=False)
    fig.suptitle(fr"Leading {k} modes, $\rho = {rho_formatted}$")
    plt.savefig(filename, dpi=400)
    plt.close()


def plot_u0(u0, u0_true, filename):
    mins = []
    maxs = []
    for u in [u0, u0_true]:
        mins.append(u.min())
        maxs.append(u.max())

    color_min = np.min(mins)
    color_max = np.max(maxs)

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(5.5, 2.5))
    im = axs[0].tricontourf(x[:, 0],
                            x[:, 1],
                            u0,
                            64,
                            vmin=color_min,
                            vmax=color_max)
    axs[0].set_title(r"$\mathbf{m}_0^u$")
    axs[0].set_xlabel(r"$x_1$")
    axs[0].set_ylabel(r"$x_2$")

    axs[1].tricontourf(x[:, 0],
                       x[:, 1],
                       u0_true,
                       64,
                       vmin=color_min,
                       vmax=color_max)
    axs[1].set_title(r"$\mathbf{u}_{0, \mathrm{true}}$")
    axs[1].set_xlabel(r"$x_1$")
    plt.colorbar(im, ax=axs[1])
    plt.savefig(filename, dpi=400)
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_files", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    assert len(args.input_files) == len(args.labels)
    results = [h5py.File(f, "r") for f in args.input_files]

    # read in metadata
    x, t = results[0]["x"][:], results[0]["t"][:]
    rhos = [result["params"][0, 0] for result in results]
    rhos_formatted = [
        texp(f"{rho:.0e}", dollar_surround=False) for rho in rhos
    ]

    # plot effective rank
    eff_ranks = [(rho, result["eff_rank"][:])
                 for rho, result in zip(rhos_formatted, results)]
    plot_eff_rank(t, args.output_dir + "eff-rank.pdf", *eff_ranks)

    # plot errors
    norm = np.linalg.norm
    post_errors = []

    for result in results:
        u_dgp = result["u_dgp"][:]
        u_post = result["u"][:]

        post_errors.append(norm(u_dgp - u_post, axis=1) / norm(u_dgp, axis=1))

    u_dgp = results[0]["u_dgp"][:]
    u_prior = results[0]["u_prior"][:]
    plot_u0(u_prior[0, :], u_dgp[0, :], args.output_dir + "icu.png")

    prior_error = norm(u_dgp - u_prior, axis=1) / norm(u_dgp, axis=1)
    plot_error(t, prior_error, args.output_dir + "error.pdf", *post_errors)

    # plot modes
    for result, label in zip(results, args.labels):
        L = result["L"][:]
        u_dofs = result["u_dofs"][:]
        plot_modes(t[-1],
                   4,
                   L[-1, u_dofs, :],
                   args.output_dir + f"modes-{label}.png",
                   rho=result["params"][0, 0])
