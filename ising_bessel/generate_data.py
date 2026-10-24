"""Generate the data for the 2D Ising (square lattice) comparison figure.

For each K = J/(k_B T) it computes g = -beta f = (1/N) ln Z in two independent ways:

  g_proposed : through a Bessel-function integral B(k) (integrand containing
               J0), composed with
                   ln Z / N = ln 2 + (1/2) [ ln|sinh(2K)/2| + B(k(K)) ],
                   k(K)     = 2 cosh(2K) coth(2K).
  g_onsager  : Onsager's original double integral,
                   ln Z / N = ln 2 + 1/(2 pi^2) * int_0^pi int_0^pi
                              ln[ cosh^2(2K) - sinh(2K)(cos x - cos y) ] dx dy.

Since g(K) = g(-K), only K >= 0 is computed and then mirrored.

The Bessel integral B(k) is cut at t = T. The integrand contains the slowly decaying
oscillatory terms (2 cos t - cos kt)/t, whose remainder beyond T is known in
closed form, -2 Ci(T) + Ci(|k| T) (Ci = cosine integral). It is added by default;
use --no-tail to reproduce the plain truncation.

Output: tab-separated file with columns K, g_proposed, g_onsager (header lines
start with '#', so numpy.loadtxt ignores them).

Usage:
    python generate_data.py
    python generate_data.py --npoints 401 --T 2000 --workers 4 --out data/energy.txt
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import j0, sici

T_CUTOFF = 5000.0  # upper limit of the Bessel integral
BLOCK = np.pi      # integration panel width (matches the period of cos t)
K_SMALL = 1e-6     # below this, K is treated as 0 and g = ln 2


# ---------------------------------------------------------------- Bessel side
def bessel_integrand(t, k):
    if t == 0.0:
        return 0.0
    return (np.cos(t) * (2.0 - j0(2.0 * t / k) ** 2) - np.cos(k * t)) / t


def tail_correction(k, T):
    """Closed form of int_T^inf (2 cos t - cos kt)/t dt = -2 Ci(T) + Ci(|k| T)."""
    return -2.0 * sici(T)[1] + sici(abs(k) * T)[1]


def bessel_integral(k, T=T_CUTOFF, tail=True, epsabs=1e-10, epsrel=1e-10):
    """Bessel-function integral B(k), panel by panel, cut at t = T."""
    k = float(k)
    if k == 0.0:
        raise ValueError("k must be non-zero (the integrand contains J0(2t/k)).")
    edges = np.append(np.arange(0.0, T, BLOCK), T)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        value, _ = quad(bessel_integrand, a, b, args=(k,), epsabs=epsabs, epsrel=epsrel, limit=100)
        total += value
    if tail:
        total += tail_correction(k, T)
    return total


def k_of_K(K):
    return 2.0 * np.cosh(2.0 * K) / np.tanh(2.0 * K)


def ln_z_bessel(K, T=T_CUTOFF, tail=True):
    if abs(K) < K_SMALL:
        return np.log(2.0)
    bessel_term = bessel_integral(k_of_K(K), T=T, tail=tail)
    return np.log(2.0) + 0.5 * (np.log(abs(0.5 * np.sinh(2.0 * K))) + bessel_term)


# --------------------------------------------------------------- Onsager side
def _onsager_integrand(y, x, K):
    arg = np.cosh(2.0 * K) ** 2 - np.sinh(2.0 * K) * (np.cos(x) - np.cos(y))
    return np.log(max(arg, 1e-300))


def ln_z_onsager(K, epsabs=1e-9, epsrel=1e-9):
    """Onsager's double integral evaluated numerically."""
    def inner(x):
        value, _ = quad(_onsager_integrand, 0.0, np.pi, args=(x, K),
                        epsabs=epsabs, epsrel=epsrel, limit=200)
        return value

    outer, _ = quad(inner, 0.0, np.pi, epsabs=epsabs, epsrel=epsrel, limit=200)
    return np.log(2.0) + outer / (2.0 * np.pi**2)


# ----------------------------------------------------------------------- main
def _compute_point(K, T, tail):
    return ln_z_bessel(K, T=T, tail=tail), ln_z_onsager(K)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kmax", type=float, default=1.0, help="K runs from -kmax to kmax (default 1.0)")
    parser.add_argument("--npoints", type=int, default=201, help="odd number of K points (default 201)")
    parser.add_argument("--T", type=float, default=T_CUTOFF, help="cutoff of the Bessel integral (default 5000)")
    parser.add_argument("--no-tail", action="store_true", help="do not add the analytic tail correction")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    parser.add_argument("--out", type=Path, default=Path("data/energy.txt"), help="output file")
    args = parser.parse_args()

    if args.npoints % 2 == 0:
        parser.error("--npoints must be odd so that K = 0 belongs to the grid")

    tail = not args.no_tail
    K_half = np.linspace(0.0, args.kmax, (args.npoints + 1) // 2)  # K >= 0

    start = time.time()
    func = partial(_compute_point, T=args.T, tail=tail)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, os.cpu_count() or 1)) as pool:
            results = list(pool.map(func, K_half))
    else:
        results = [func(K) for K in K_half]
    g_half = np.array(results)  # shape (n_half, 2)

    # mirror: g(-K) = g(K)
    K = np.concatenate([-K_half[:0:-1], K_half])
    g = np.vstack([g_half[:0:-1], g_half])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"2D Ising model, square lattice: g = -beta f vs K = J/(k_B T)\n"
              f"generated by generate_data.py: T = {args.T:g}, tail correction = {tail}, kmax = {args.kmax:g}\n"
              f"K\tg_proposed\tg_onsager")
    np.savetxt(args.out, np.column_stack([K, g]), delimiter="\t", fmt="%.15g", header=header, comments="# ")

    diff = np.abs(g[:, 0] - g[:, 1])
    print(f"wrote {args.out} ({len(K)} points) in {time.time() - start:.1f} s")
    print(f"max |g_proposed - g_onsager| = {diff.max():.3e} at K = {K[diff.argmax()]:.4f}")


if __name__ == "__main__":
    main()
