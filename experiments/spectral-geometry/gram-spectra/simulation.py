"""Build the introduction figure from cached Gaussian spectra and frozen features."""

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.integrate import quad

SIZES = (128, 256, 512, 1024)
REPS = 2000
POLY = HERE / "data"
ROWS = ("Proportional", "Polynomial", "CIFAR-10")


def reference_statistics(cache):
    # Compute the two reference halves in blocks, then combine their scatters.
    batches = [
        np.load(cache / f"resnet50_data_batch_{i}_features.npy", mmap_mode="r")
        for i in range(1, 6)
    ]
    assert all(batch.shape == (10000, 2048) for batch in batches)
    chunks = [part for batch in batches for part in np.array_split(batch, 2)]
    means, scatters = [], []
    for parts in (chunks[:5], chunks[5:]):
        mean = sum(part.sum(axis=0, dtype=np.float64) for part in parts) / 25000
        scatter = np.zeros((2048, 2048), dtype=np.float64)
        for part in parts:
            centered = part - mean
            scatter += centered.T @ centered
        means.append(mean)
        scatters.append(scatter)
    difference = means[0] - means[1]
    full = (
        scatters[0] + scatters[1] + 12500 * np.outer(difference, difference)
    ) / 49999
    raw_ref = np.linalg.eigvalsh(full)[::-1]
    raw_halves = np.stack(
        [np.linalg.eigvalsh(scatter / 24999)[::-1] for scatter in scatters]
    )
    return (means[0] + means[1]) / 2, raw_ref, raw_halves


def sample_empirical_rows(cache):
    mean, raw_ref, raw_halves = reference_statistics(cache)
    test = np.load(cache / "resnet50_test_batch_features.npy").astype(np.float64)
    assert test.shape == (10000, 2048)
    test -= mean
    test /= np.sqrt(raw_ref[0])
    ref, halves = raw_ref / raw_ref[0], raw_halves / raw_ref[0]
    print("Leading training covariance eigenvalue", raw_ref[0], flush=True)
    # Cache dot products once; each sampled Gram is a principal submatrix.
    test_gram = test @ test.T
    squared_norms = np.sum(test**2, axis=1)
    np.testing.assert_allclose(test_gram.diagonal(), squared_norms, rtol=1e-12)
    metadata = dict(
        reference_spectrum=ref,
        half_reference_spectra=halves,
        raw_reference_spectrum=raw_ref,
        training_mean=mean,
        feature_multiplier=1 / np.sqrt(raw_ref[0]),
        gaussian_d_over_n=0.25,
        replicate_count=REPS,
        gram_matrix="XX*/n",
        eigenvalue_divisor_by_n=np.array(SIZES),
    )
    spectra = {}
    for n in SIZES:
        # Distinct random streams for each size and trial. The reference set is fixed.
        streams = np.random.SeedSequence([20261110, 50, n]).spawn(REPS)
        groups = np.stack(
            [
                np.random.default_rng(stream).choice(len(test), n, replace=False)
                for stream in streams
            ]
        )
        metadata[f"cifar_test_indices_n{n}"] = groups
        rng = np.random.default_rng(20261110 + n)
        gaussian_values = []
        for _ in range(REPS):
            d = n // 4
            gaussian = rng.standard_normal((n, d))
            positive = np.linalg.eigvalsh(gaussian.T @ gaussian / n)
            assert positive.min() > 0
            gaussian_values.append(np.pad(positive, (n - d, 0)))
        spectra[0, n] = np.stack(gaussian_values)
        cifar_values = []
        for indices in groups:
            values = np.linalg.eigvalsh(test_gram[np.ix_(indices, indices)] / n)
            assert (
                values[0]
                > values[-1] * max(n, test.shape[1]) * np.finfo(np.float64).eps
            )
            np.testing.assert_allclose(
                values.sum(), squared_norms[indices].sum() / n, rtol=1e-10
            )
            cifar_values.append(values)
        spectra[2, n] = np.stack(cifar_values)
        print("Verified spectra at n =", n, flush=True)
    return spectra, metadata


def finish_experiment(spectra, metadata):
    from polynomial_samples import polynomial_spectra

    for n in SIZES:
        spectra[1, n] = polynomial_spectra(n, REPS)
    for (row, n), values in spectra.items():
        assert (
            values.shape == (REPS, n)
            and np.isfinite(values).all()
            and values.min() >= -1e-10
        )
    tmp = HERE / "figure_data.tmp.npz"
    np.savez_compressed(
        tmp, **metadata, **{f"row{row + 1}_n{n}": v for (row, n), v in spectra.items()}
    )
    tmp.replace(HERE / "figure_data.npz")
    return render(
        spectra, metadata["reference_spectrum"], metadata["half_reference_spectra"]
    )


def density(values, edges):
    counts = np.histogram(values, edges)[0]
    assert counts.sum() + (values > edges[-1]).sum() == values.size
    return counts / (values.size * np.diff(edges))


def mp_bins(edges):
    ratio = 0.25
    left, right = (1 - np.sqrt(ratio)) ** 2, (1 + np.sqrt(ratio)) ** 2

    def pdf(x):
        return np.sqrt((right - x) * (x - left)) / (2 * np.pi * x)

    masses = [
        quad(pdf, max(a, left), min(b, right))[0] if max(a, left) < min(b, right) else 0
        for a, b in zip(edges[:-1], edges[1:])
    ]
    assert edges[0] == 0 and abs(sum(masses) - ratio) < 1e-7
    masses[0] += 1 - ratio
    assert abs(sum(masses) - 1) < 1e-7
    return masses / np.diff(edges)


def render(spectra, ref, halves):
    palettes = [
        ("#3480ab", "#69a2c1", "#99c1d8", "#c5dde9"),
        ("#b96730", "#d68f5b", "#e8b58e", "#f6d6bb"),
        ("#34856b", "#66a68e", "#99c8b5", "#c6e3d6"),
    ]
    main_limits = (3, 3, 3)
    inset_limits = {1: 24, 2: 65}
    inset_bins = {1: 48, 2: 120}
    records = []
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 3.8))
    fig.subplots_adjust(
        left=0.117, right=0.985, top=0.91, bottom=0.206, wspace=0.15, hspace=0.17
    )
    dash = (0, (5, 3))

    def draw_hist(ax, heights, edges, palette, cutoffs):
        # Place the display baseline below even a single observation in the largest panel.
        floor = 10.0 ** np.floor(
            np.log10(0.5 / (REPS * max(SIZES) * (edges[1] - edges[0])))
        )
        ax.set_yscale("log")
        # Split the fill at each quantile while preserving the original bin height.
        boundaries = np.union1d(edges, np.clip(cutoffs, edges[0], edges[-1]))
        left, right = boundaries[:-1], boundaries[1:]
        segment_heights = heights[np.searchsorted(edges, left, side="right") - 1]
        bands = np.searchsorted(cutoffs, left, side="right")
        colors = np.asarray(palette)[bands]
        occupied = segment_heights > 0
        ax.bar(
            left[occupied],
            segment_heights[occupied] - floor,
            width=(right - left)[occupied],
            bottom=floor,
            align="edge",
            color=colors[occupied],
            linewidth=0,
            antialiased=False,
        )
        ax.set_ylim(bottom=floor)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(edges[0], edges[-1])
        ax.tick_params(which="minor", length=0)

    def draw_reference(ax, heights, edges):
        ax.stairs(
            np.where(heights > 0, heights, np.nan),
            edges,
            baseline=None,
            color="#222222",
            linestyle=dash,
            linewidth=1.0,
        )

    for row in range(3):
        edges = np.linspace(0, main_limits[row], 121)
        heights = [density(spectra[row, n], edges) for n in SIZES]
        upper = 1.8 * max(h.max() for h in heights)
        inset_axes = []
        inset_max = 0.0
        for col, n in enumerate(SIZES):
            ax = axes[row, col]
            values = spectra[row, n]
            cutoffs = np.quantile(values, [0.90, 0.95, 0.99], method="inverted_cdf")
            draw_hist(ax, heights[col], edges, palettes[row], cutoffs)
            ax.set_ylim(top=upper)
            ax.set_xticks([0, main_limits[row] / 2, main_limits[row]])
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.grid(axis="y", which="major", alpha=0.13)
            ax.tick_params(length=2.8, pad=2, labelleft=col == 0, labelbottom=row == 2)
            overflow = float((values > edges[-1]).mean())
            if overflow:
                ax.text(
                    0.99,
                    1.01,
                    f"{100 * overflow:.1f}% →",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                )
            if row == 0:
                draw_reference(ax, mp_bins(edges), edges)
                ax.set_title(rf"$n={n}$", pad=4)
            q = np.quantile(values, [0.25, 0.5, 0.75, 0.9])
            record = dict(
                row=ROWS[row],
                n=n,
                quartiles=q[:3].tolist(),
                q90=float(cutoffs[0]),
                pooled_q90_q95_q99=cutoffs.tolist(),
                largest_observed=float(values.max()),
                smallest_observed=float(values.min()),
                main_right=edges[-1],
                main_off_axis_mass=overflow,
                zero_mass=float(np.mean(values == 0)),
            )
            if row:
                rank = n
                scale = n**-1.2 if row == 1 else ref[rank - 1]
                inset = ax.inset_axes([0.51, 0.53, 0.46, 0.40])
                inset_axes.append(inset)
                limits = np.linspace(0, inset_limits[row], inset_bins[row] + 1)
                scaled = values / scale
                h = density(scaled, limits)
                draw_hist(inset, h, limits, palettes[row], cutoffs / scale)
                inset_max = max(inset_max, h.max())
                if row == 1:
                    x, _, cdf = np.load(POLY / f"deterministic_{n}.npz")["curve"]
                    theory = np.diff(
                        np.interp(limits, x, cdf, left=0, right=1)
                    ) / np.diff(limits)
                    draw_reference(inset, theory, limits)
                    inset_max = max(inset_max, theory.max())
                symbol = r"\sigma_n" if row == 1 else (r"\widehat\sigma_n")
                inset.set_xlabel(rf"$\lambda/{symbol}$", fontsize=8.0, labelpad=-1.5)
                inset.set_xticks([0, 12, 24] if row == 1 else [0, 30, 60])
                inset.tick_params(labelsize=6.7, length=1.5, pad=1, labelleft=col == 0)
                inset.patch.set_alpha(0.98)
                inset.set_yticks([1e-3, 1e-1])
                record.update(
                    inset_scale=float(scale),
                    inset_right=float(limits[-1]),
                    inset_off_axis_mass=float((scaled > limits[-1]).mean()),
                    inset_quartiles=(q[:3] / scale).tolist(),
                    inset_pooled_q90_q95_q99=(cutoffs / scale).tolist(),
                )
                if row == 2:
                    record.update(
                        reference_rank=rank,
                        half_reference_scales=halves[:, rank - 1].tolist(),
                    )
                    if n < ref.size:
                        tail = ref[n:].sum() / n
                        record.update(
                            tail_scale=float(tail),
                            tail_scaled_quartiles=(q[:3] / tail).tolist(),
                        )
            records.append(record)
        for inset in inset_axes:
            inset.set_ylim(top=1.5 * inset_max)
        axes[row, 0].set_ylabel(ROWS[row], fontsize=10.5, labelpad=3)
    fig.supxlabel(r"Eigenvalue of $XX^*/n$", y=0.083, fontsize=11.5)
    legend_handles = [
        Patch(facecolor=color) for color in ("#696969", "#929292", "#b7b7b7", "#dcdcdc")
    ]
    legend_handles.append(Line2D([], [], color="#222222", ls=dash, lw=1))
    fig.legend(
        legend_handles,
        ["0–90%", "90–95%", "95–99%", "99–100%", "Theoretical prediction"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=5,
        frameon=False,
        prop={"family": ["cmr10", "STIXGeneral"], "size": 9},
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.12,
    )
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "output" / f"intro_gram_histograms.{ext}", dpi=300)
    (HERE / "diagnostics.json").write_text(
        json.dumps(records, indent=2), encoding="utf8"
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    finish_experiment(*sample_empirical_rows(parser.parse_args().cache))

__all__ = ["HERE", "SIZES", "np", "render"]
