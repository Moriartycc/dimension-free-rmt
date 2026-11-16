# Ridge-type filters — Figures 7 and 11

Each figure is a 3×4 panel: rows are variance $`\mathsf{V}_n`$, squared bias $`\mathsf{B}_n`$, and total prediction risk $`\mathsf{R}_n=\mathsf{V}_n+\mathsf{B}_n`$; columns are the four laws in the root README. Figure 7 uses $`q=16`$; Appendix Figure 11 uses $`q=24`$. Panels a/b are $`k=1`$ absolute/relative, and panel c is $`k=2`$ relative. Results for $`k=4`$ are retained and can be plotted, but are not current paper figures.

```sh
python main.py --q 16 --k 1 --side absolute
python main.py --q 16 --k 2 --side relative
python main.py --q 24 --k 4 --side relative
python main.py --mode theory
python main.py --mode full             # 400 CUDA trials; expensive
```

All laws have $`d=2^{24}`$, noise standard deviation $`\tau=0.01`$, and the same source definition. Empirical sample sizes are 16, 32, 64, 128, 192, 256, 384, and 512. Theoretical curves use every fourth integer from 16 through 512. Penalties are

```math
\lambda\in\{-0.6,0,0.5,1,1.5,2\}\kappa_n,
```

where $`\kappa_n`$ is the predicted lower edge of **$`XX^*/n`$**, not the effective regularization root. `c` in stored arrays is simply $`\lambda/\kappa_n`$. The filter is $`\varphi_{k,\lambda}(x)=[1-(\lambda/(x+\lambda))^k]/x`$. At $`\lambda=0`$ it is ridgeless for every $`k`$. No penalty is optimized after observing the data.

## Empirical conditional risks

The Gaussian worker accumulates $`G=Z\Sigma Z^\top`$ and $`H=Z\Sigma^2Z^\top`$ in coordinate blocks, along with $`Z\Sigma\theta`$ and $`Z\Sigma^2\theta`$. All products and sums use float64, with TF32 disabled. If $`G/n=U\mathrm{diag}(e)U^\top`$, store

```math
L=U^\top(H/n)U,\quad y=U^\top Z\Sigma\theta/\sqrt n,\quad r=U^\top Z\Sigma^2\theta/\sqrt n.
```

For a filter vector $`f_i=\varphi(e_i)`$ and $`v=f\odot y`$,

```math
V=\frac{\tau^2}{n}\sum_i f_i^2L_{ii},\qquad
B=\|\Sigma\theta\|^2-2r^\top v+v^\top Lv.
```

This evaluates response-noise expectations analytically for each design. The 400 trials sample the design randomness. All laws, penalties, iteration counts, and nested sample sizes are paired within a trial. The saved source contractions support **q=16 and q=24 only**; a new empirical source generally requires more sampling or additional saved contractions.

The worker records the event $`\lambda_{\min}(XX^*/n)\ge0.8\kappa_n`$ and distance to the negative pole. All raw risks are retained; no trial is discarded on the basis of that event. It checks NumPy versus PyTorch products and, for the first replicate, independently replays the coordinates to verify squared bias in feature space. Aggregation verifies seed uniqueness, $`R=V+B`$, source-independent variance, and the equality of all ridgeless iteration counts.

## Predictions and uncertainty

`shared/ridge_prediction.py` solves the scalar effective-regularization equation on the physical branch and recursively inverts its local power series through order seven. The coefficients give the derivatives needed for $`k=1,2,4`$. Population quadrature is checked under a larger exact head and finer integration. This avoids finite differences of high-order derivatives near negative ridge.

Absolute markers show trial means and pointwise $`1.96\,\mathrm{SD}/\sqrt{400}`$ intervals. Relative plots divide each component's mean by its own mean ridgeless baseline at the same $`n`$, law, and source: $`V/V_0`$, $`B/B_0`$, and $`R/R_0`$. They are ratios of means, not means of trialwise ratios. Paired ratio standard errors use the influence values

```math
\frac{A_r-\widehat\rho B_r}{\overline B},\qquad\widehat\rho=\overline A/\overline B.
```

Thus the ridgeless ratio is identically one with zero estimated uncertainty. Theoretical ratios use their own theoretical ridgeless baselines.

## Data and cluster execution

`ridge_aggregated.npz` retains the individual trial risks. `data/theory.npz` contains the deterministic curves. `summarize.py` constructs the component means, paired errors, and ratios used by `plot.py`. `edges.npz` supplies the deterministic penalty scale to the sampler. Per-design sample-space sufficient statistics are much larger and are generated in the Git-ignored `samples/` directory.

```sh
python compute.py
python worker.py --start 0 --trials 10
# Dispatch nonoverlapping starts 0,10,...,390 as separate CUDA jobs.
python aggregate.py
python summarize.py
python plot.py --q 16 --k 1 --side absolute
```

The worker skips completed trial files. Do not mix configurations in one `samples/` directory. After the complete run, gradient flow can reuse these same files without generating another Gaussian design.
