# Spectral-law validation — Figure 6

Figure 6 (`fig:numerical-spectral-laws`) compares empirical PDFs/CDFs with deterministic predictions. Columns are power $\alpha=0.2$, critical $\beta=0.2$, power $\alpha=0.05$, and critical $\beta=0.05$. Each curve uses 2,000 independent Gaussian matrices at $n=2,8,32,128$, with covariance exactly truncated at $d=2^{24}$.

```sh
python main.py                          # Saved publication arrays
python main.py --mode theory            # Recompute deterministic curves
python main.py --mode full --device cpu # Full 2,000-trial computation
# Use --device cuda with an appropriate PyTorch installation.
```

## Simulation and coordinates

The worker accumulates $G=Z\Sigma Z^\top$ in float64 using coordinate blocks of at most 65,536, then takes the eigenvalues of the symmetrized matrix. **Every coordinate through $d$ is sampled.** There is no random-tail compression or expectation replacement. The four laws use the same Gaussian block within a replicate. Sample sizes 2 and 8 are paired in one independent run family, and 32 and 128 in another. The original small-size run additionally sampled 4 and 16, which are omitted from the publication plot; a fresh run retains only the selected sizes and need not be bitwise identical.

The data stores eigenvalues of the raw Gram matrix $XX^*$, not $XX^*/n$. The figure uses $x/a_n$ for power and $(x-b_n)/a_n$ for critical decay, where $a_n=n\sigma_n$ and $b_n=\sum_{j=n+1}^d\sigma_j$. This is equivalent to dividing normalized-Gram eigenvalues by $\sigma_n$, with the corresponding critical shift. Mixing these conventions introduces a factor of $n$.

The display applies $\log_{10}(u-c)$ to the horizontal coordinate: $c=3$ for power 0.2, $c=7$ for power 0.05, and $c=-2$ for both critical laws. Tick labels remain in the original scaled coordinate $u$. The PDF is a density **with respect to $u$**, not a density with respect to its logarithm; no Jacobian is applied solely for changing the axis display.

## Histograms and error bars

Histogram bins have width 0.25 for power 0.05 and 0.2 otherwise. Each trial contributes its normalized histogram and empirical CDF. We average those functions across matrices. Density normalization includes all eigenvalues, including those outside the plotting window.

The independent unit is a matrix, not an individual eigenvalue. Pointwise error bars are $1.96$ times the across-matrix sample standard deviation divided by $\sqrt{2000}$. The same rule is used for histogram bins and CDF ordinates, with lower/upper bars clipped at the natural probability boundaries where needed. These are Monte Carlo intervals for the average empirical curve, not variability bands for individual spectra.

`data/spectra.npz` contains all publication spectra. `data/predictions/` contains the boundary predictions, and `data/scales.json` the finite-population shifts/scales. `plot.py` also regenerates standard-error arrays and window-mass diagnostics. `compute.py` uses the shared finite-population solver and checks measure masses.

## Parallel execution

Disjoint `--start/--stop` intervals can run independently. Use the same output folder and seed within a run family, with nonoverlapping intervals:

```sh
python worker.py --device cpu --sizes 32 128 --start 0 --stop 50 --output samples/large
python worker.py --device cpu --sizes 2 8 --seed 202611160000 --start 0 --stop 50 --output samples/small
# After all intervals through 2,000 finish:
python aggregate.py
python plot.py
```

`aggregate.py` requires all expected trials and validates dimensions and seeds. A reduced `--dimension` is available only for smoke tests and cannot be aggregated into the publication experiment. Set `SLURM_CPUS_PER_TASK` to the thread count for a worker; default is four. Choose concurrency according to the local scheduler allocation rather than hard-coding an account or cluster hostname.
