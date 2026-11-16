from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))
"""Completed 2x4 empirical/theoretical comparison, with matrix-level uncertainty."""
from pathlib import Path
import numpy as np
import json
from common import CASES

HERE = Path(__file__).resolve().parent / "data"
NS = (2, 8, 32, 128)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

PRED = HERE / "predictions"
COLORS = ["#be5d24", "#8c68a8", "#388d80", "#186a9c"]
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def main():
    samples = np.load(HERE / "spectra.npz")
    scales = json.loads((HERE / "scales.json").read_text())
    # Match Figure 4's native 7.2-inch width and font sizes, so manuscript scaling agrees.
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex="col", sharey="row")
    report = {}
    uncertainty = {}
    for col, (law, p) in enumerate(CASES):
        bounds = (
            ((3.2, 18) if p == 0.2 else (7.2, 30)) if law == "power" else (-1.2, 12)
        )
        center = (3 if p == 0.2 else 7) if law == "power" else -2
        width = 0.25 if law == "power" and p == 0.05 else 0.2
        bin_left = (3 if p == 0.2 else 7) if law == "power" else bounds[0]
        bins = np.arange(bin_left, bounds[1] + width / 2, width)
        # Sample CDF curves and their markers uniformly in the displayed coordinate.
        grid = np.geomspace(bounds[0] - center, bounds[1] - center, 1201) + center
        for ni, n in enumerate(NS):
            key = f"{law}_{p:g}_n{n}"
            s = scales[f"{law}_{p:g}"]["by_n"][str(n)]
            empirical = (samples[key] - s["b"]) / s["a"]
            assert empirical.shape == (2000, n)
            pooled = np.sort(empirical.ravel())
            hist = np.histogram(pooled, bins)[0] / len(pooled) / np.diff(bins)
            cdf = np.searchsorted(pooled, grid, side="right") / len(pooled)
            # Independent units are matrices, not their correlated eigenvalues.
            trial_hist = np.array(
                [np.histogram(v, bins)[0] / n / np.diff(bins) for v in empirical]
            )
            trial_cdf = np.array(
                [np.searchsorted(v, grid, side="right") / n for v in empirical]
            )
            hist_se = trial_hist.std(axis=0, ddof=1) / np.sqrt(len(empirical))
            cdf_se = trial_cdf.std(axis=0, ddof=1) / np.sqrt(len(empirical))
            assert np.allclose(hist, trial_hist.mean(axis=0))
            assert np.allclose(cdf, trial_cdf.mean(axis=0))
            uncertainty[key + "_hist_se"] = hist_se
            uncertainty[key + "_cdf_se"] = cdf_se
            uncertainty[key + "_bins"] = bins
            uncertainty[key + "_grid"] = grid
            data = np.load(PRED / f"{key}.npz")
            c = COLORS[ni]
            axs[0, col].stairs(
                hist, bins, color=c, lw=0.8, ls=(0, (2, 2)), alpha=0.65, zorder=2
            )
            axs[0, col].errorbar(
                (bins[:-1] + bins[1:]) / 2,
                hist,
                yerr=np.stack([np.minimum(hist, 1.96 * hist_se), 1.96 * hist_se]),
                fmt="none",
                ecolor=c,
                elinewidth=0.6,
                capsize=1.2,
                capthick=0.6,
                alpha=0.7,
                zorder=4,
            )
            for lo, hi in data["support"]:
                use = (data["x"] >= lo) & (data["x"] <= hi)
                axs[0, col].plot(
                    data["xr"][use], data["rho"][use], color=c, lw=1.35, zorder=3
                )
            predicted = np.interp(grid, data["xr"], data["cdf"], left=0, right=1)
            axs[1, col].plot(grid, cdf, color=c, lw=1.15, ls=(0, (2, 2)), alpha=0.7)
            axs[1, col].plot(grid, predicted, color=c, lw=1.35)
            # Sparse hollow symbols distinguish empirical CDFs even where they coincide.
            ii = np.arange(35 + ni * 15, len(grid), 150)
            axs[1, col].errorbar(
                grid[ii],
                cdf[ii],
                yerr=np.stack(
                    [
                        np.minimum(cdf[ii], 1.96 * cdf_se[ii]),
                        np.minimum(1 - cdf[ii], 1.96 * cdf_se[ii]),
                    ]
                ),
                fmt="o",
                ms=2.5,
                mec=c,
                mfc="white",
                mew=0.65,
                ecolor=c,
                elinewidth=0.7,
                capsize=1.8,
                capthick=0.7,
                zorder=4,
            )
            report[key] = dict(
                bin_width=width,
                replicates=2000,
                uncertainty="Pointwise normal-approximation 95% Monte Carlo intervals; trial-level SD/sqrt(2000)",
                max_histogram_standard_error=float(hist_se.max()),
                max_cdf_standard_error=float(cdf_se.max()),
                mass_below_window=float(np.mean(pooled < bounds[0])),
                mass_above_window=float(np.mean(pooled > bounds[1])),
                cdf_max_error_in_window=float(np.max(np.abs(cdf - predicted))),
            )
        axs[0, col].set_title(
            ("Power, " + rf"$\alpha={p:g}$")
            if law == "power"
            else ("Critical, " + rf"$\beta={p:g}$"),
            pad=7,
        )
        axs[1, col].set_xlabel(
            r"$x/a_n$" if law == "power" else r"$(x-b_n)/a_n$", labelpad=3
        )
        for row in range(2):
            ax = axs[row, col]
            ax.set_xscale(
                "function",
                functions=(
                    lambda u, shift=center: np.log10(
                        np.maximum(np.asarray(u) - shift, np.finfo(float).tiny)
                    ),
                    lambda v, shift=center: 10.0 ** np.asarray(v) + shift,
                ),
            )
            if law == "power":
                ticks = [3.5, 4, 5, 8, 18] if p == 0.2 else [7.5, 8, 10, 15, 30]
            else:
                ticks = [-1, 0, 1, 3, 6, 12]
            ax.set_xlim(*bounds)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis="y", alpha=0.13, lw=0.6)
            ax.tick_params(length=3, pad=2)
    axs[0, 0].set_ylabel("PDF", labelpad=5)
    axs[1, 0].set_ylabel("CDF", labelpad=5)
    axs[0, 0].set_ylim(0, 0.6)
    axs[0, 0].set_yticks([0, 0.2, 0.4, 0.6])
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].set_yticks([0, 0.25, 0.5, 0.75, 1])
    handles = [
        Line2D([], [], color=c, lw=1.6, label=rf"$n={n}$") for c, n in zip(COLORS, NS)
    ]
    methods = [
        Line2D([], [], color=".3", lw=1.4, label="Theoretical\nprediction"),
        axs[1, 0].errorbar(
            [],
            [],
            yerr=[],
            color=".5",
            lw=0.9,
            ls="--",
            marker="o",
            mfc="white",
            ms=3,
            capsize=2,
            label=r"Empirical",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.515, 0.006),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=2.3,
        columnspacing=1.5,
    )
    axs[0, 0].legend(
        handles=methods,
        loc="upper right",
        frameon=False,
        fontsize=7.8,
        handlelength=1.8,
        borderaxespad=0.35,
    )
    fig.subplots_adjust(
        left=0.085, right=0.99, bottom=0.18, top=0.91, wspace=0.25, hspace=0.27
    )
    for ext in ("png", "pdf"):
        fig.savefig(HERE.parent / "output" / f"spectral_comparison_2x4.{ext}", dpi=220)
    plt.close(fig)
    np.savez_compressed(HERE / "monte_carlo_standard_errors.npz", **uncertainty)
    (HERE / "plot_errorbar_diagnostics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
