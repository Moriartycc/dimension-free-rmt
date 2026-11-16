# Gram spectra — Figure 1

Figure 1 (`fig:intro-gram-spectra`) compares normalized Gram matrices $`XX^*/n`$ in three rows and four sample sizes $`n=128,256,512,1024`$. Each panel pools the eigenvalues of 2,000 independently sampled matrices. The histogram height is logarithmic; the eigenvalue axis is linear. Four shades distinguish the pooled empirical percentile ranges 0–90%, 90–95%, 95–99%, and 99–100%. Quantiles are computed from the pooled spectra, not from trial-to-trial uncertainty.

```sh
python main.py                         # Regenerate from figure_data.npz
python main.py --mode smoke             # Small numerical checks
python -m pip install -r requirements.txt
python main.py --mode full --features /absolute/path/to/features
```

The final command extracts frozen features and runs all three rows. Its input directory must contain `cifar-10-python.tar.gz` (the official CIFAR-10 Python archive, MD5 `c58f30108f718f92721af3b95e74349a`) and `resnet50-0676ba61.pth` (the torchvision ImageNet V1 ResNet-50 checkpoint). Obtain these from the dataset publisher and torchvision's official distribution; they are not included in this repository. The extraction script checks the archive digest and checkpoint digest prefix before reading them. Images and feature arrays remain local.

## Three rows

1. **Proportional:** independent standard Gaussian entries, $`d=n/4`$. Compute the $`d`$ positive eigenvalues of $`X^*X/n`$ and append the $`n-d`$ exact zeros. Thus the empirical measure includes a zero atom of mass $`3/4`$. The reference histogram integrates the Marchenko–Pastur continuous density over each bin and includes the atom in the first bin.
2. **Polynomial:** $`\sigma_j=j^{-1.2}`$. Explicitly sample $`D_n=\max(16384,256n)`$ coordinates and add $`\sum_{j>D_n}\sigma_j/n`$ to each Gram eigenvalue. This is an expectation approximation to the infinite random tail, distinct from the exact finite-coordinate validation in Figure 6. The tail sum uses the Hurwitz zeta function. Products use float32 with TF32 disabled; accumulation and eigendecomposition use float64. A paired float64-product check is run on the first new replicate at each size. The rescaled insets use $`\sigma_n`$ and the deterministic prediction.
3. **CIFAR-10:** a fixed ImageNet V1 ResNet-50 in evaluation mode, with its final classifier replaced by the identity, gives 2,048-dimensional pooled features. Use the checkpoint's prescribed preprocessing. Center both training and test vectors by the 50,000-image training mean; divide them by the square root of the leading training-covariance eigenvalue, using divisor 49,999. Each trial samples $`n`$ distinct images from the 10,000-image test pool, independently across trials. The feature map and training reference are fixed across trials. Sampling without replacement within a trial does not prevent overlap between trials.

For CIFAR, $`\widehat\sigma_n`$ is the $`n`$th eigenvalue of the normalized **training feature covariance**. It is not the minimum eigenvalue of each sampled test Gram matrix, so an inset eigenvalue divided by $`\widehat\sigma_n`$ need not be bounded below by one. Reference statistics are also computed on two training halves as a diagnostic.

## Files and reproducibility

`figure_data.npz` contains the actual publication spectra, reference spectra, training mean, and test-image indices. `simulation.py` computes the proportional and feature rows; `polynomial_samples.py` computes the Gaussian polynomial row; `extract_features.py` handles the fixed network. `data/deterministic_*.npz` holds the inset predictions. `plot.py` only reads these arrays and renders the final layout.

The archived polynomial run was extended in stages. A clean new simulation uses independent seeds throughout and therefore matches its statistical setup rather than every original random draw. Use plot mode to recover the archived figure exactly at the data level. Histogram density is normalized by the full number of eigenvalues, including mass outside the displayed window; it is not renormalized after cropping.
