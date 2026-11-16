# Deterministic bulk laws — Figure 2

Figure 2 (`fig:deterministic-bulk-laws`) shows finite-$n$ deterministic laws and their limiting laws for the infinite spectra $\sigma_j=j^{-1.2}$ and $\sigma_j=[j(1+\log j)^{1.2}]^{-1}$. The displayed sample sizes are 8, 64, and 1,024. These curves do not involve Gaussian Monte Carlo trials.

```sh
python main.py                  # Plot the archived curves
python main.py --mode full      # Recompute curves, refinement checks, and plot
```

## Boundary calculation

Write $a_n=n\sigma_n$. The power-law coordinate is $x/a_n$; the critical-law coordinate is $(x-b_n)/a_n$, with $b_n=\sum_{j>n}\sigma_j$. The solver parametrizes the inverse Stieltjes variable as $u=t+ih$ and solves

$$\frac1n\sum_j\frac{(\sigma_j/\sigma_n)^2}{(t-\sigma_j/\sigma_n)^2+h^2}=1.$$

The lower edge is the negative real root of the same condition with $h=0$. The first population-tail moment is separated analytically before evaluating the inverse map; only terms with quadratic decay remain in the tail quadrature. This avoids cancellation and a prohibitively large explicit cutoff for critical decay.

The published refinement uses an exact head $2\max(512,4n)$, 12-point Gauss–Legendre rules over 44 logarithmic panels, and a critical centering sum through 131,072 with an Euler–Maclaurin remainder. It is checked against a coarser head and quadrature. A graded mesh resolves the square-root lower edge. The CDF comes from a complex primitive rather than numerical accumulation of the plotted density. The limiting curves are evaluated analytically in `shared/limiting_laws.py`.

`compute.py` checks the boundary equation, CDF monotonicity, edge expansion, density integral against the CDF primitive, and changes under refinement. `curves.npz` is the publication curve archive; `validation.json` records the original diagnostics. The fresh computation rewrites both files. Figure 4 reuses this boundary solver, with its own directional weights and zero atoms.
