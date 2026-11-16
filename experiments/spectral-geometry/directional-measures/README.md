# Directional measures — Figure 4

Figure 4 (`fig:directional-measures`) uses the same infinite population laws and boundary map as Figure 2. Its four columns are power/critical at $`n=8`$, followed by power/critical at $`n=64`$. Within each panel the source ranks are $`j=1,n/4,n/2,n`$.

```sh
python main.py
python main.py --mode full
```

The final presentation uses a linear eigenvalue axis and logarithmic density axis. The bars above each panel give the atom at zero of the corresponding directional probability measure $`\mu_{\omega,j}`$. The continuous density alone generally has mass less than one. The weighted measure $`\mu_\omega`$ is also plotted, **without probability normalization**: its mass is $`\mathrm{Tr}(\Sigma^2)/n`$ and it has no zero atom.

`compute_directional.py` imports the common boundary calculation from the sibling deterministic-bulk-laws experiment, locates the support bands, evaluates each directional density, and retains the atoms separately. Its checks include directional total mass, first moments, and the weighted mass identity. The plot uses `curves.npz` and `validation.json`; it does not infer zero masses by histogramming a narrow bin near the origin.

This experiment is deterministic. There are no empirical error bars and no finite $`d=2^{24}`$ truncation here. See the [bulk-law README](../deterministic-bulk-laws/README.md) for tail quadrature and boundary refinement.
