# 2D Ising model from a Bessel-function integral: data and figure

Code that produces the comparison figure between the free energy, internal
energy and entropy of the 2D Ising model (square lattice) obtained from a
Bessel-function integral representation and Onsager's exact solution.


> Archived version of this code: [Zenodo DOI]

## Contents

| File | Role |
|------|------|
| `generate_data.py` | Computes g = -βf = (1/N) ln Z on a grid of K = J/(k_B T), in two independent ways: through a Bessel-function (J0) integral representation composed with the Ising solution and through Onsager's original double integral. Writes `data/energy.txt`. |
| `plot_thermo.py` | Reads `data/energy.txt`, obtains u/J = -dg/dK and s/k_B = g - K dg/dK from a smoothing spline of the Bessel-based g, compares with Onsager's closed-form u and s, and saves the figure. |
| `plot_style.mplstyle` | Matplotlib style used in the figure (optional). |
| `data/energy.txt` | Output of `generate_data.py` used in the paper (columns: K, g_proposed, g_onsager). |
| `figures/` | Output of `plot_thermo.py`. |

## Requirements

Python 3 with the packages in `requirements.txt`. Tested with Python [3.10.12],
NumPy [1.26.4], SciPy [1.14], Matplotlib [3.5.1].

```bash
pip install -r requirements.txt
```

## Reproducing the figure

```bash
python generate_data.py --workers 4   # about a minute; writes data/energy.txt
python plot_thermo.py                 # writes figures/plot_thermo_stacked.{pdf,png}
```

If you only want the figure, skip the first step: `data/energy.txt` is included.

`generate_data.py` prints the largest difference between the two g columns, as
a consistency check. Useful options: `--npoints`, `--kmax`, `--T` (cutoff of
the Bessel integral), `--no-tail` (plain truncation, without the analytic tail
correction), `--out`.

## Numerical notes

- The Bessel integral is cut at t = T (default 5000). The slowly decaying terms
  (2 cos t - cos kt)/t leave a remainder of order 1/T, which is added in
  closed form as -2 Ci(T) + Ci(|k|T). Without it the error in g is about 2e-4
  at T = 5000; with it, about 1e-8 away from K_c.
- u and s of the Bessel-based solution come from differentiating a smoothing
  spline of g (`SMOOTHING` and `SPLINE_DEGREE` at the top of `plot_thermo.py`),
  so their accuracy is limited by that choice.
- g(K) is even, so only K >= 0 is computed and then mirrored.


