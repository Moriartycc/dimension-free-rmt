# Interpolation endpoint errors — Figure 9

Figure 9 (`fig:interpolation-endpoint-errors`) compares Backward, Linear, and Forward interpolation in a 2×6 layout. Power-law panels are on the left and critical-law panels on the right. Rows are $L^1$ and $L^2$ errors across independent trials. They are **endpoint residual norms**, not norms over interpolation time $k$.

```sh
python main.py                  # Archived 400-trial residuals and final layout
python main.py --mode full      # Preparation, checks, all cells; CUDA cluster workload
```

## Population and grid

Here $n=400$, $d=65,536$, and $\alpha=\beta=0.2$. The laws are exactly zero beyond $d$. Define $a_n=n\sigma_n$ and $b_n=\sum_{j=n+1}^d\sigma_j$. We evaluate $z=a_n\zeta$ for power decay and $z=b_n+a_n\zeta$ for critical decay.

There are 25 real coordinates, spaced by 0.25 on $[1,7]$ for power and $[-2,4]$ for critical. Seventeen geometric imaginary heights span 0.005 to 1. This gives 425 cells per law, concentrated near the real spectrum. The colored real-axis segments show the deterministic support projected into the same coordinates; a segment can extend beyond the displayed interval.

## Residuals and pairing

Let $w_0$ be the endpoint of the fitted backward path, $w_0^{\rm lin}$ the population linear-path starting point, and $\mu_n$ the actual forward endpoint. In the paper's indexing, $G_n$ uses $n-1$ observations, while $G_1$ uses none. For each of the 400 evaluation trials, the three normalized complex residuals are

$$\frac{G_n(z,0)-G_1(z,w_0)}{1+|G_1(z,w_0^{\rm lin})|},\qquad
\frac{G_n(z,0)-G_1(z,w_0^{\rm lin})}{1+|G_1(z,w_0^{\rm lin})|},\qquad
\frac{G_n(z,\mu_n)-G_1(z,w_0^{\rm lin})}{1+|G_1(z,w_0^{\rm lin})|}.$$

For residuals $r_1,\ldots,r_{400}$, plot $400^{-1}\sum|r_j|$ and $(400^{-1}\sum|r_j|^2)^{1/2}$. This differs from the absolute mean complex drift $|400^{-1}\sum r_j|$, which can cancel. The same evaluation Gaussian draws are reused across all methods and grid cells. Backward expectation fitting uses a separate independent set of 1,024 draws; those draws are not counted as evaluation trials.

Every panel uses the same logarithmic color normalization, determined jointly from both norms and both laws. Brighter colors mean larger errors. Light hatching and an outline identify the method with the lowest observed norm at that cell within the same law and row. This comparison is descriptive, not a significance test.

## Resolvent computation and checks

The first 4,096 population coordinates are retained explicitly. The remaining sampled coordinates through 65,536 are compressed into 16 real weighted random Gram moments. A geometric matrix-polynomial expansion evaluates their contribution to the resolvent and its derivative. The compressed tail retains its sampled randomness; it is not replaced by an expectation. There is no tail beyond $d$.

Large population eigenvalues are deflated before the sample-space linear solve. Path calculations use complex128. Forward steps use fixed-point iteration and a root-solver fallback, with convergence residuals and expansion ratios recorded. The population starting point is continued from large imaginary height to the target grid. The preflight compares backward fits of size 256, 512, and 1,024 and checks compressed versus direct full-coordinate resolvents at representative path indices and low-height grid points. It also checks the zero-observation trace against a direct finite sum. Production refuses to start if the preflight fails.

## Execution and data

```sh
python prepare.py power
python run.py preflight power
python run.py production power --column 0
# Columns 0,...,24 may run independently; repeat for critical.
python aggregate.py
python plot.py
```

Preparation writes hashed training/evaluation inputs. `--row-start` and `--row-stop` allow nonoverlapping subdivisions of a column. Keep the same prepared data for all jobs to preserve pairing. The preparation and production paths are GPU-memory intensive; their batch sizes are explicit constants in `model.py`. No local cluster account, hostname, or credentials are embedded.

`completed/{power,critical}/heatmap_data.npz` retains the publication's complex residuals, support and population endpoints. `endpoint_norms.npz` is the compact plotting summary. Raw indexed inputs and cell checkpoints are excluded from Git. `aggregate.py` reconstructs the summaries from a completed fresh run, verifies finiteness and dimensions, and derives the shared color limits.
