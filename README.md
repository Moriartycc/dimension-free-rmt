# Dimension-free spectral laws and the risk of spectral filters

Code for the numerical experiments in the paper **Dimension-free spectral laws and the risk of spectral filters**.

Author: Chen Cheng.

The repository contains the following experiments reported in the paper:

1. [Gram spectra](experiments/spectral-geometry/gram-spectra/README.md): empirical spectra for proportional, polynomial-decay, and CIFAR-10 feature models (Figure 1).
2. [Deterministic bulk laws](experiments/spectral-geometry/deterministic-bulk-laws/README.md): finite-sample predictions and their limiting laws (Figure 2).
3. [Directional measures](experiments/spectral-geometry/directional-measures/README.md): spectral measures and weighted measures (Figure 4).
4. [Spectral-law validation](experiments/numerical-validation/spectral-laws/README.md): empirical PDFs and CDFs against theoretical predictions (Figure 6).
5. [Ridge-type filters](experiments/numerical-validation/ridge-filters/README.md): variance, bias, and risk for ridge and iterated ridge (Figures 7 and 11).
6. [Gradient flow](experiments/numerical-validation/gradient-flow/README.md): risk surfaces and optimal stopping times (Figures 8 and 12).
7. [Interpolation errors](experiments/interpolation/path-errors/README.md): backward, linear, and forward interpolation on the complex plane (Figure 9).

**Start with the saved results.** Every numerical figure in the paper can be regenerated on a CPU without repeating the simulations. Fresh simulations are separate commands: the validation experiments explicitly sample up to $d=2^{24}$ Gaussian coordinates and are substantially more expensive.

## Quick start

Use Python 3.11 or later, from the repository root:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tests/numerical_checks.py
python reproduce.py 6
```

`python reproduce.py all` regenerates all 13 numerical figures/panels in the table below. Each command writes PDF and PNG files in its experiment's `output/` directory. Published outputs are included, so the figures can also be viewed before installing Python. `requirements-tested.txt` records the versions used to check this distribution; install that file for the tested environment. Matplotlib's bundled Computer Modern fonts are used; the numerical plots do not require TeX.

## Figure mapping

Numbers refer to the paper PDF checked on **November 16, 2026**. The manuscript labels and descriptive filenames remain useful if figures are renumbered. [figures.json](figures.json) is the machine-readable mapping used by `reproduce.py`.

| Paper | Experiment and setup | Published figure | Command |
|---|---|---|---|
| 1 | [Gram spectra](experiments/spectral-geometry/gram-spectra/README.md): proportional, polynomial, CIFAR-10 | [PDF](experiments/spectral-geometry/gram-spectra/output/intro_gram_histograms.pdf) | `python reproduce.py 1` |
| 2 | [Deterministic bulk laws](experiments/spectral-geometry/deterministic-bulk-laws/README.md) and their infinite-sample limits | [PDF](experiments/spectral-geometry/deterministic-bulk-laws/output/deterministic_bulk_laws.pdf) | `python reproduce.py 2` |
| 3 | Two-ray contour for gradient flow; analytical illustration | [TikZ source](illustrations/analytic-filter-support.tex) | See [illustrations](illustrations/README.md) |
| 4 | [Directional measures](experiments/spectral-geometry/directional-measures/README.md), including their atoms and weighted measure | [PDF](experiments/spectral-geometry/directional-measures/output/directional_measures_semilogy.pdf) | `python reproduce.py 4` |
| 5 | Two-circle contours for ridge; analytical illustration | [TikZ source](illustrations/ridge-filter-contours.tex) | See [illustrations](illustrations/README.md) |
| 6 | [Spectral validation](experiments/numerical-validation/spectral-laws/README.md): PDFs and CDFs, $n=2,8,32,128$ | [PDF](experiments/numerical-validation/spectral-laws/output/spectral_comparison_2x4.pdf) | `python reproduce.py 6` |
| 7a | [Ridge](experiments/numerical-validation/ridge-filters/README.md), $q=16,k=1$, absolute V/B/R | [PDF](experiments/numerical-validation/ridge-filters/output/ridge_q16_k1_absolute_paper.pdf) | `python reproduce.py 7a` |
| 7b | Ridge, $q=16,k=1$, relative V/B/R | [PDF](experiments/numerical-validation/ridge-filters/output/ridge_q16_k1_relative_paper.pdf) | `python reproduce.py 7b` |
| 7c | Iterated ridge, $q=16,k=2$, relative V/B/R | [PDF](experiments/numerical-validation/ridge-filters/output/ridge_q16_k2_relative_paper.pdf) | `python reproduce.py 7c` |
| 8 | [Gradient flow](experiments/numerical-validation/gradient-flow/README.md), $q=16,24$, empirical surfaces and theoretical meshes | [PDF](experiments/numerical-validation/gradient-flow/output/gf_empirical_surfaces_2x4_envelope.pdf) | `python reproduce.py 8` |
| 9 | [Interpolation errors](experiments/interpolation/path-errors/README.md), backward/linear/forward, $L^1,L^2$ | [PDF](experiments/interpolation/path-errors/output/endpoint_l1_l2_combined.pdf) | `python reproduce.py 9` |
| 10 | Two-phase backward-path argument; analytical illustration | [TikZ source](illustrations/two-phase-backward-path.tex) | See [illustrations](illustrations/README.md) |
| 11a | Appendix F.1: ridge, $q=24,k=1$, absolute V/B/R | [PDF](experiments/numerical-validation/ridge-filters/output/ridge_q24_k1_absolute_paper.pdf) | `python reproduce.py 11a` |
| 11b | Appendix F.1: ridge, $q=24,k=1$, relative V/B/R | [PDF](experiments/numerical-validation/ridge-filters/output/ridge_q24_k1_relative_paper.pdf) | `python reproduce.py 11b` |
| 11c | Appendix F.1: iterated ridge, $q=24,k=2$, relative V/B/R | [PDF](experiments/numerical-validation/ridge-filters/output/ridge_q24_k2_relative_paper.pdf) | `python reproduce.py 11c` |
| 12 | Appendix F.2: theory-only gradient flow, $q=28,32,36,40$ | [PDF](experiments/numerical-validation/gradient-flow/output/gf_theory_q28_q32_q36_q40_4x4.pdf) | `python reproduce.py 12` |

## Organization

```text
shared/                         Covariance laws, boundary solver, risks, plotting
experiments/
  spectral-geometry/
    gram-spectra/               Figure 1
    deterministic-bulk-laws/    Figure 2
    directional-measures/      Figure 4
  interpolation/
    path-errors/               Figure 9
  numerical-validation/
    spectral-laws/             Figure 6
    ridge-filters/              Figures 7 and 11
    gradient-flow/              Figures 8 and 12
illustrations/                  Figures 3, 5, and 10 (TikZ)
tests/                         Small numerical identity and consistency checks
```

Each experiment has a README, a portable `main.py`, a shell runner, a configuration record, and publication outputs. The JSON configuration records describe the paper runs; they are **not arbitrary parameter overrides**. The scientific routines expose their supported parameters directly. The shared code avoids importing any unpublished pilot directory.

## Three ways to run

From any experiment directory:

```sh
python main.py --mode plot     # Saved numerical results; the default
python main.py --mode smoke    # Small numerical checks, no production data overwritten
python main.py --mode full    # Fresh computation; read the experiment README first
```

`bash run_experiment.sh` is equivalent to `python main.py`. Ridge accepts `--q 16|24`, `--k 1|2|4`, and `--side absolute|relative`. Gradient flow uses `--appendix` for Figure 12. `--mode theory` computes predictions without new Gaussian designs for spectral laws, ridge, and gradient flow. The full interpolation experiment includes preparation and numerical preflight checks.

Fresh ridge simulations need a CUDA-enabled PyTorch installation. The spectral worker supports CPU or CUDA. Gradient-flow theory can run on CPU or CUDA, and empirical gradient flow is evaluated from the saved ridge sufficient statistics. CIFAR feature extraction requires the additional requirements and input files described in its README. Full simulation defaults should be treated as cluster workloads, not installation tests.

## Numerical setup

For Figures 6–8 and 11–12, the covariance is diagonal and exactly truncated at $d=2^{24}$:

$$
\sigma_j=j^{-1-\alpha}\quad\text{or}\quad
\sigma_j=\frac{1}{j(1+\log j)^{1+\beta}},\qquad 1\le j\le d.
$$

The four columns are power $\alpha=0.2$, critical $\beta=0.2$, power $\alpha=0.05$, and critical $\beta=0.05$. No mass beyond $d$ is added. For the filter experiments,

$$
\beta_j=\sqrt{\sigma_j}\theta_j,\qquad
\theta_j=\frac{q^{j-1}/(j-1)!}{\left(\sum_{r=0}^{d-1}(q^r/r!)^2\right)^{1/2}},
\qquad \tau=0.01.
$$

The common factor $e^{-q}$ cancels when normalizing the Poisson coefficients. This is an $\ell^2$-normalized source, not a probability vector. `gammaln` and subtraction of the maximum log weight prevent overflow. The source is evaluated on an explicit head whose omitted energy is checked; the covariance itself retains all $d$ coordinates.

The settings for the earlier illustrations are intentionally different. Figures 2 and 4 use infinite laws with analytic tail corrections. Figure 1 uses proportional Gaussian features, an infinite polynomial law with an expectation approximation beyond its explicit random head, and fixed CIFAR features. Figure 9 uses a separate exactly truncated $d=2^{16}$ experiment. Their details are given in the corresponding READMEs.

## Numerical methods beyond the paper

**Population sums and support.** `shared/covariance.py` sums the first 4,096 coordinates exactly and integrates the remaining *finite* population in logarithmic coordinate using 12-point Gauss–Legendre quadrature on panels of width at most 0.75. The first moment is retained separately through analytic/Euler–Maclaurin tail differences. The refinement check uses a 16,384-coordinate head, 20-point rules, and half-width panels. This approximation is used for deterministic predictions, not for the Gaussian simulation.

`shared/spectral_solver.py` parametrizes the real boundary by the inverse Stieltjes variable $u$. It locates support endpoints from the derivative condition, searches the explicit population gaps for separated bands, and solves for the imaginary height on each band. The mesh is graded near edges. It stores the Gram density/CDF, directional densities and zero atoms, and the weighted measure. Mass and first-moment identities provide independent quadrature checks. The CDF is computed from an analytic primitive of the inverse map rather than by accumulating a coarse histogram.

**Exact finite-dimensional sampling.** The validation workers generate Gaussian coordinate blocks and accumulate $Z\Sigma Z^\top$ in float64. They do not form an $n\times d$ design at once and do not replace its tail by an expectation. The ridge worker also accumulates $Z\Sigma^2 Z^\top$ and two source contractions. An eigendecomposition in sample space then gives conditional variance and squared bias. Response noise is integrated analytically; trial variation is variation over the random designs. The same Gaussian blocks are used across laws and nested sample sizes within a replicate. Independent replicates use independent seeds.

**Filter predictions.** Ridge predictions use a scalar effective-regularization root and formal inverse-series coefficients through order seven, sufficient for $k=4$. This avoids numerical differentiation near negative penalties. Gradient-flow predictions integrate the real-boundary measures with stable divided exponentials and explicit diagonal limits. Their mass normalization is enforced before integration, and the ridgeless limit is checked against the independent scalar ridge formula. See the [ridge](experiments/numerical-validation/ridge-filters/README.md) and [gradient-flow](experiments/numerical-validation/gradient-flow/README.md) READMEs for formulas, uncertainty, and optimum searches.

**What “reproduce” means.** Plot mode uses the archived numerical arrays underlying the publication, so no sampling variation is introduced. Full mode generates statistically equivalent experiments; hardware, library versions, random-number chunking, and the original staged accumulation of the Figure 1 samples can prevent bitwise equality. Checkpointed workers resume completed trial files. Keep different parameter settings in separate run directories. Large per-trial sufficient statistics and CIFAR images/features are excluded from Git; the aggregate arrays required for every published numerical plot are included.

The repository contains the final experiment paths and numerical checks. Abandoned tuning sweeps, alternate figure layouts, login scripts, cluster account settings, and manuscript working notes are omitted.

## Stable manuscript labels and configuration records

| Figure | Manuscript label | Settings |
|---|---|---|
| 1 | `fig:intro-gram-spectra` | [JSON](experiments/spectral-geometry/gram-spectra/configs/main.json) |
| 2 | `fig:deterministic-bulk-laws` | [JSON](experiments/spectral-geometry/deterministic-bulk-laws/configs/main.json) |
| 4 | `fig:directional-measures` | [JSON](experiments/spectral-geometry/directional-measures/configs/main.json) |
| 6 | `fig:numerical-spectral-laws` | [JSON](experiments/numerical-validation/spectral-laws/configs/main.json) |
| 7a | `fig:ridge-q16-k1-absolute` | [JSON](experiments/numerical-validation/ridge-filters/configs/main.json) |
| 7b | `fig:ridge-q16-k1-relative` | [JSON](experiments/numerical-validation/ridge-filters/configs/main.json) |
| 7c | `fig:ridge-q16-k2-relative` | [JSON](experiments/numerical-validation/ridge-filters/configs/main.json) |
| 8 | `fig:numerical-gradient-flow` | [JSON](experiments/numerical-validation/gradient-flow/configs/main.json) |
| 9 | `fig:interpolation-endpoint-errors` | [JSON](experiments/interpolation/path-errors/configs/main.json) |
| 11a | `fig:ridge-q24-k1-absolute` | [JSON](experiments/numerical-validation/ridge-filters/configs/appendix.json) |
| 11b | `fig:ridge-q24-k1-relative` | [JSON](experiments/numerical-validation/ridge-filters/configs/appendix.json) |
| 11c | `fig:ridge-q24-k2-relative` | [JSON](experiments/numerical-validation/ridge-filters/configs/appendix.json) |
| 12 | `fig:gradient-flow-source-transitions` | [JSON](experiments/numerical-validation/gradient-flow/configs/appendix.json) |
