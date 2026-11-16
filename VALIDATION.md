# Validation of the public-code organization

The initial repository was checked with Python 3.11 and the numerical versions in `requirements-tested.txt`.

- All 13 numerical figure/panel commands in `figures.json` completed using only the repository and its installed dependencies.
- Regenerated PNGs were compared with the approved original figure outputs.
- The small numerical checks verify finite-population moments against direct sums, ridgeless identities for k=1/2/4, boundary-measure masses, gradient-flow initial and ridgeless limits, independent Gaussian sampling, and figure/configuration links.
- The spectral worker smoke test uses a dimension that is not a multiple of the coordinate-block size, checking its last partial block.
- The cleaned ridge summary code was rerun from the archived 400-trial arrays.
- Python undefined-name and unused-import checks passed. Public source files contain no local machine paths, cluster accounts, authentication helpers, or imported pilot directories.

The original 2,000- and 400-trial production runs were not repeated during this packaging check. Their aggregate data is preserved. Full CUDA sampling, CIFAR feature extraction, and fresh full-grid interpolation runs remain expensive workflows; their numerical preflight checks are retained in the code. The standalone TikZ schematics are provided as source separately from the numerical plotting checks.
