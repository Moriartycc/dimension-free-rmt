# Gradient flow — Figures 8 and 12

Figure 8 (`fig:numerical-gradient-flow`) has two rows, $`q=16,24`$, and the same four law columns as the ridge figures. Figure 12 (`fig:gradient-flow-source-transitions`) is a **theory-only** supplement with $`q=28,32,36,40`$. Both use $`d=2^{24}`$ and noise standard deviation 0.01.

```sh
python main.py                         # Figure 8, archived empirical results
python main.py --appendix              # Figure 12, archived theoretical results
python main.py --mode theory           # Recompute Figure 8 predictions
python main.py --mode theory --appendix
python main.py --mode full             # Reuse the complete ridge samples/ directory
```

## Time, risk, and empirical reuse

The plotted time is $`\widetilde t=t\tau_1(n)/n`$, with $`\tau_1(n)=\sum_{j=n+1}^d\sigma_j`$. Displayed times are the squares of 91 equally spaced numbers in $`[0,2]`$, giving denser coverage near zero. The filter is $`(1-e^{-tx})/x`$, evaluated using `expm1` to avoid early-time cancellation.

Empirical sample sizes are 64, 128, 192, 256, 384, and 512, with the same 400 Gaussian designs and q=16/24 contractions as ridge. `postprocess.py` reads the ridge sample-space eigenvalues, weighted Gram matrices, and source contractions. It evaluates conditional variance and bias for every time; no response noise or design is resampled. It checks zero initial variance, initial bias equal to signal energy, monotone variance, and agreement of the infinite-time limit with the saved ridgeless results.

For each sample size, trial risks are divided by the **mean empirical ridgeless risk**. Theory uses its own theoretical baseline. The figures do not add irreducible test-response noise to the prediction risk.

## Meaning of the surfaces

In Figure 8, the colored surface is the empirical mean, the thin dark mesh is the theoretical prediction, and the violet shell represents pointwise 10–90% trial quantiles. It is trial variability, not a confidence interval for the mean. Each paired trial is interpolated in $`n`$ using PCHIP for display before the pointwise quantiles are computed. The denser displayed surface is not evidence of extra simulated sample sizes.

Nested translucent sheets inside the envelope increase visibility near the mean. Their opacity is a visual taper, not an estimated probability density. The faint floor projection represents theoretical risk. Heights outside the displayed limits are clipped as polygons rather than flattened onto a ceiling. The main figure displays heights 0.5–1.5; the supplement uses 0–1.5 to expose the low-risk branches.

The red curves show predicted minimizing times. Surface occlusion is evaluated in projected triangle coordinates: hidden portions are drawn with thinner, lighter red strokes so the whole path remains traceable. This changes visibility only; it does not change the optimum values.

## Theory and optimum search

The theory is computed at $`n=64,96,128,192,256,320,384,448,512`$. `shared/risks.py` integrates the directional and weighted real-boundary measures, including zero atoms and the continuous diagonal of divided differences. Total masses are normalized before integration. The ridgeless limit is checked against the independent scalar ridge formula, and selected times are recomputed on a doubled boundary mesh.

The optimum search is separate from the displayed grid. For Figure 8 it uses 151 logarithmically spaced normalized times in $`[10^{-12},10^3]`$. Each detected derivative sign change is refined by Brent's method in log time. The candidate risks at all detected stationary points are compared with $`t=0`$ and $`t=\infty`$. It does not assume a unique stationary point. Figure 12 uses 161 search times in $`[10^{-10},10^4]`$; relative improvements over ridgeless of at most $`10^{-8}`$ are displayed as numerical ties.

Triangles and dashed connectors indicate jumps between sampled minimizing branches. Finite-to-finite jumps connect neighboring sample sizes. A jump to infinity is drawn parallel to the time axis to a schematic infinity endpoint; this segment is not a finite optimum at the plotted extension time. These searches provide numerical evidence of branch changes and possible phase transitions. They do not prove absence of unresolved stationary points or establish a transition threshold between sampled values of $`n`$.

## Files and sources

`gf_aggregated.npz` retains the 400-trial V/B/R arrays for the main figure. `theory/` contains its dense-time predictions and minimizing candidates; `supplement/` contains theory for q=28–40. `compute.py` and `compute_supplement.py` generate them, and the two plot scripts render the final layouts. The archived arrays suffice for plotting both figures on CPU.

Changing q for theory reweights the directional measures. It does **not** provide an empirical trial envelope at a new source. Figure 12 therefore contains no empirical surface or uncertainty band. A fresh empirical main figure requires the complete ridge `samples/` cache; its compact aggregate risks alone cannot reconstruct all gradient-flow times.
