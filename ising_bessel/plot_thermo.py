"""Free energy, internal energy and entropy of the 2D Ising model (square lattice).

Post-processing script. It reads a tab-separated file with three columns
    K,  g_proposed,  g_onsager
where K = J/(k_B T) and g = -beta f = (1/N) ln Z, then:

  * differentiates a smoothing spline of g_proposed to obtain
        u/J  = -dg/dK        and        s/k_B = g - K dg/dK,
  * computes Onsager's exact u/J and s/k_B from the closed-form expression,
  * plots the comparison.

The input file is produced by generate_data.py.

Usage:
    python plot_thermo.py                      # reads data/energy.txt
    python plot_thermo.py data/energy.txt --show
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline
from scipy.special import ellipk

# Smoothing factor of the spline (sum of squared residuals allowed).
# Tuned by hand: smaller values follow numerical noise in g and amplify it
# in the derivative; larger values oversmooth the region near K_c.
SMOOTHING = 2e-7
SPLINE_DEGREE = 4  # degree 4 so that the first derivative is still smooth
N_GRID = 4001      # resolution of the fine grid for the exact solution
EPS = 1e-6         # avoids division by zero at K = 0


def load_data(path):
    """Return K, g_proposed, g_onsager sorted by increasing K."""
    data = np.loadtxt(path, delimiter="\t")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1], data[order, 2]


def energy_and_entropy_from_spline(K, g, smoothing=SMOOTHING, degree=SPLINE_DEGREE):
    """u/J and s/k_B from a smoothing spline of g(K)."""
    spline = UnivariateSpline(K, g, s=smoothing, k=degree)
    dg_dK = spline.derivative()(K)
    u = -dg_dK
    s = spline(K) - K * dg_dK
    return u, s


def onsager_exact(K_eval):
    """Exact u/J and s/k_B (Onsager), integrating dg/dK = -u from K = 0.

    K_eval must span K = 0, since g(0) = ln 2 is the integration constant.
    """
    K_eval = np.asarray(K_eval, dtype=float)
    if not (K_eval.min() <= 0.0 <= K_eval.max()):
        raise ValueError("K range must include K = 0 (g(0) = ln 2 is the reference).")

    # Fine grid that contains K = 0 exactly.
    K_fine = np.unique(np.concatenate([np.linspace(K_eval.min(), K_eval.max(), N_GRID), [0.0]]))
    i0 = np.searchsorted(K_fine, 0.0)

    K_safe = np.where(np.abs(K_fine) < EPS, EPS, K_fine)
    C, S = np.cosh(2 * K_safe), np.sinh(2 * K_safe)
    coth2K, tanh2K = C / S, S / C
    m = np.minimum((2 * S / C**2) ** 2, 1.0)  # elliptic parameter, clipped against rounding
    with np.errstate(invalid="ignore"):
        u_fine = -coth2K * (1 + (2 / np.pi) * (2 * tanh2K**2 - 1) * ellipk(m))
    u_fine[i0] = 0.0  # exact limit at K = 0

    # 0 * inf at K = K_c exactly: fill with linear interpolation of neighbours.
    bad = ~np.isfinite(u_fine)
    if bad.any():
        u_fine[bad] = np.interp(K_fine[bad], K_fine[~bad], u_fine[~bad])

    raw = cumulative_trapezoid(u_fine, K_fine, initial=0.0)
    g_fine = np.log(2) - (raw - raw[i0])  # g(0) = ln 2, dg/dK = -u
    s_fine = K_fine * u_fine + g_fine

    return np.interp(K_eval, K_fine, u_fine), np.interp(K_eval, K_fine, s_fine)


def make_figure(K, g_proposed, g_onsager, u_proposed, s_proposed, u_exact, s_exact):
    fig, axes = plt.subplots(3, 1, figsize=(4, 6))
    marks = dict(marker="o", linestyle="none", markersize=5, markevery=(2, 4))

    axes[0].plot(K, g_proposed, linewidth=1.5, label="Bessel function solution")
    axes[0].plot(K, g_onsager, label="Onsager's solution", **marks)
    axes[0].set_xlim(-1.0, 1.0)
    axes[0].set_ylabel(r"$-\beta f$", fontsize=14)
    axes[0].legend()

    axes[1].plot(K, u_proposed, linewidth=1.5)
    axes[1].plot(K, u_exact, color="C1", fillstyle="none", **marks)
    axes[1].set_xlim(-1.0, 1.0)
    axes[1].set_ylabel(r"$u/J$", fontsize=14)

    axes[2].plot(K, s_proposed, linewidth=1.5)
    axes[2].plot(K, s_exact, color="C1", fillstyle="none", **marks)
    axes[2].set_xlim(0.0, 1.0)
    axes[2].axhline(np.log(2), ls=":", color="gray", linewidth=0.8)
    axes[2].set_ylabel(r"$s/k_B$", fontsize=14)
    axes[2].set_xlabel(r"$K$", fontsize=14)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_style = Path(__file__).with_name("plot_style.mplstyle")
    parser.add_argument("data", type=Path, nargs="?", default=Path("data/energy.txt"),
                        help="tab-separated file: K, g_proposed, g_onsager (default data/energy.txt)")
    parser.add_argument("--style", default=str(default_style) if default_style.exists() else None,
                        help="matplotlib style file (default: plot_style.mplstyle next to this script, if present)")
    parser.add_argument("--out", default="figures/plot_thermo_stacked", help="output path without extension")
    parser.add_argument("--show", action="store_true", help="open the figure window")
    args = parser.parse_args()

    if args.style:
        plt.style.use(args.style)

    K, g_proposed, g_onsager = load_data(args.data)
    u_proposed, s_proposed = energy_and_entropy_from_spline(K, g_proposed)
    u_exact, s_exact = onsager_exact(K)

    fig = make_figure(K, g_proposed, g_onsager, u_proposed, s_proposed, u_exact, s_exact)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")            # vector, for the paper
    fig.savefig(f"{args.out}.png", bbox_inches="tight", dpi=300)  # raster, for quick viewing
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
