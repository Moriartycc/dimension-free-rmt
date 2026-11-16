from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))
"""Render the two deterministic bulk panels from saved curves."""
from compute import HERE, WINDOW, np
from plotting import plt, Line2D
from matplotlib.ticker import MaxNLocator

DISPLAY = {8: "#443983", 64: "#23888d", 1024: "#719b34"}


def plot_curves(ax, curves, law, width):
    for n, color in DISPLAY.items():
        x, y, _ = curves[f"{law}_{n}"]
        ax.plot(x, y, color=color, lw=width)
    x, y, _ = curves[f"{law}_limit"]
    ax.plot(x, y, color="#191919", lw=width, ls=(0, (4, 3)))


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
    fig.subplots_adjust(left=0.081, right=0.986, bottom=0.31, top=0.855, wspace=0.24)
    with np.load(HERE / "curves.npz") as curves:
        for ax, law in zip(axes, ("power", "critical")):
            plot_curves(ax, curves, law, 1.25)
            ax.set_xlim(*WINDOW[law])
            peak = max(curves[f"{law}_{key}"][1].max() for key in [*DISPLAY, "limit"])
            ax.set_ylim(0, 1.08 * peak)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis="y", alpha=0.13)
            ax.tick_params(length=3, pad=2)
            ax.set_xlabel(
                r"$\lambda/a_n$" if law == "power" else r"$(\lambda-b_n)/a_n$",
                labelpad=2,
            )
            inset = ax.inset_axes([0.55, 0.49, 0.42, 0.41])
            plot_curves(inset, curves, law, 0.95)
            if law == "power":
                inset.set_xlim(4.188, 4.278)
                inset.set_ylim(0, 0.10)
                inset.set_xticks([4.20, 4.24, 4.27])
                inset.set_yticks([0, 0.08])
            else:
                inset.set_xlim(-1.04, -0.55)
                inset.set_ylim(0, 0.22)
                inset.set_xticks([-1.0, -0.8, -0.6])
                inset.set_yticks([0, 0.1, 0.2])
            inset.tick_params(labelsize=6.7, length=2, pad=1)
            inset.set_title("Lower edge", fontsize=7.5, pad=2)
        axes[0].set_title(r"Power law $\quad\sigma_j=j^{-1.2}$", fontsize=11, pad=5)
        axes[1].set_title(
            r"Critical law $\quad\sigma_j=j^{-1}(1+\log j)^{-1.2}$", fontsize=11, pad=5
        )
        axes[0].set_ylabel("Density", labelpad=3)
    handles = [
        Line2D([], [], color=c, lw=1.5, label=rf"$n={n}$") for n, c in DISPLAY.items()
    ]
    handles.append(
        Line2D([], [], color="#191919", ls=(0, (4, 3)), lw=1.4, label=r"$\mu_\star$")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=2,
        columnspacing=1.5,
    )
    for ext in ("pdf", "png"):
        fig.savefig((HERE / "output") / f"deterministic_bulk_laws.{ext}", dpi=300)


if __name__ == "__main__":
    main()
