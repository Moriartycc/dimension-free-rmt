from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))
import numpy as np
from common import CASES
from plotting import plt, Line2D, MaxNLocator

HERE = Path(__file__).resolve().parent
COLORS = ["#2477a6", "#171717", "#d48c32", "#c45b50", "#8967ab", "#3f9279"]


def lambda_label(c):
    if c == 0:
        return r"$\lambda=0$ (ridgeless)"
    coefficient = "" if c == 1 else f"{c:g}"
    return rf"$\lambda={coefficient}\kappa_n$"


"""Separate q=16/24 V/B/R figures, matched to Figure 4 typography."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--q", type=int, choices=(16, 24), default=16)
    parser.add_argument("--side", choices=("absolute", "relative"), required=True)
    args = parser.parse_args()
    out = HERE / "output"
    a = np.load(HERE / "data" / f"ridge_q{args.q}_combined_summary.npz")
    ki = int(np.flatnonzero(a["k"] == args.k)[0])
    relative = args.side == "relative"
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axs = plt.subplots(3, 4, figsize=(7.2, 4.2), sharex=True)
    for col, (law, p) in enumerate(CASES):
        name = f"{law}_{p:g}"
        pred = a[name + ("_prediction_ratio" if relative else "_prediction")][
            :, :, ki, :
        ]
        mean = a[name + ("_ratio" if relative else "_mean")][:, :, ki, :]
        se = a[name + ("_ratio_se" if relative else "_se")][:, :, ki, :]
        for row, metric in enumerate(("V", "B", "R")):
            ax = axs[row, col]
            for ci, c in enumerate(a["c"]):
                ax.plot(
                    a["theory_n"],
                    pred[:, ci, row],
                    color=COLORS[ci],
                    lw=1.65 if c == 0 else 1.15,
                    zorder=2,
                )
                ax.errorbar(
                    a["n"],
                    mean[:, ci, row],
                    yerr=1.96 * se[:, ci, row],
                    fmt="o",
                    ms=2.5,
                    mfc="white",
                    mec=COLORS[ci],
                    mew=0.65,
                    ecolor=COLORS[ci],
                    elinewidth=0.65,
                    capsize=1.5,
                    capthick=0.6,
                    clip_on=False,
                    zorder=3,
                )
            ax.set_xlim(16, 512)
            ax.set_xticks([16, 256, 512])
            ax.tick_params(pad=2, length=2.5)
            ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
            ax.grid(axis="y", alpha=0.13, lw=0.6)
            if not relative:
                ax.ticklabel_format(
                    axis="y", style="sci", scilimits=(0, 0), useMathText=True
                )
                ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(
                    ("Power, " + rf"$\alpha={p:g}$")
                    if law == "power"
                    else ("Critical, " + rf"$\beta={p:g}$"),
                    y=1.17,
                    pad=2,
                )
            if row == 2:
                ax.set_xlabel("$n$", labelpad=2)
            if col == 0:
                label = rf"\mathsf{{{metric}}}_n[\mathrm{{\varphi}}_{{k,\lambda}}]"
                if relative:
                    label += rf"/\mathsf{{{metric}}}_0"
                ax.set_ylabel("$" + label + "$", labelpad=4)
    empirical = axs[0, 0].errorbar(
        [],
        [],
        yerr=[],
        fmt="o",
        ms=2.5,
        mfc="white",
        mec=".3",
        ecolor=".3",
        capsize=1.5,
        label="Empirical",
    )
    fig.legend(
        handles=[
            Line2D([], [], color=".3", lw=1.2, label="Theoretical prediction"),
            empirical,
        ],
        fontsize=9,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.075),
        ncol=2,
        handlelength=1.8,
    )
    handles = [
        Line2D(
            [], [], color=COLORS[i], lw=1.65 if c == 0 else 1.2, label=lambda_label(c)
        )
        for i, c in enumerate(a["c"])
    ]
    penalty_legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.014),
        ncol=6,
        frameon=False,
        fontsize=9,
        columnspacing=0.65,
        handlelength=1.25,
        handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.12, right=0.99, bottom=0.23, top=0.87, wspace=0.48, hspace=0.38
    )
    fig.canvas.draw()
    legend_bounds = penalty_legend.get_window_extent().transformed(
        fig.transFigure.inverted()
    )
    assert legend_bounds.x0 >= 0 and legend_bounds.x1 <= 1, legend_bounds
    stem = f"ridge_q{args.q}_k{args.k}_{args.side}_paper"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
