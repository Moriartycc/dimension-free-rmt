from pathlib import Path
import sys

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "shared").is_dir()))
sys.path.insert(0, str(ROOT / "shared"))
"Compact review figure for the four directional measures per panel."
from compute_directional import HERE, PANELS, np, json
from plotting import plt, Line2D
from matplotlib.ticker import NullLocator

COLORS = ["#443983", "#23888d", "#719b34", "#191919"]
LABELS = ["$j=1$", "$j=n/4$", "$j=n/2$", "$j=n$"]
WEIGHTED_COLOR = "#c14f19"
WEIGHTED_STYLE = (0, (5, 1.6, 1, 1.6))


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.2, top=0.67, wspace=0.26)
    checks = json.loads((HERE / "validation.json").read_text())
    with np.load(HERE / "curves.npz") as data:
        for ax, (law, n) in zip(axes, PANELS):
            key = f"{law}_{n}"
            for i in range(len(checks[key]["support"])):
                t, *densities = data[f"{key}_band{i}"]
                for k, y in enumerate(densities):
                    ax.plot(
                        t,
                        y,
                        color=COLORS[k],
                        lw=1.25,
                        ls=(0, (4, 2)) if k == 3 else "-",
                    )
                ax.plot(
                    t,
                    data[f"{key}_weighted{i}"],
                    color=WEIGHTED_COLOR,
                    ls=WEIGHTED_STYLE,
                    lw=1.5,
                    zorder=4,
                )
            ax.set_title(
                ("Power" if law == "power" else "Critical") + f", $n={n}$",
                fontsize=10.5,
                y=1.49,
            )
            ax.set_xscale("linear")
            ax.set_xlim((0.3, 2.6) if n == 8 else (0.025, 1.55))
            ticks = [0.4, 1, 2] if n == 8 else [0.03, 0.1, 0.3, 1]
            ticks = [0.5, 1, 1.5, 2, 2.5] if n == 8 else [0.5, 1, 1.5]
            ax.set_xticks(ticks, [f"{val:g}" for val in ticks])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_yscale("log", nonpositive="mask")
            ax.set_ylim(0.0001, 12)
            ax.set_yticks([0.0001, 0.01, 1, 10])
            ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(length=2.5, pad=2)
            ax.grid(axis="y", alpha=0.13)
            inset = ax.inset_axes([0.3, 1.13, 0.53, 0.22])
            inset.bar(np.arange(4), data[key + "_atom"], color=COLORS, width=0.69)
            inset.set_ylim(0, 1)
            inset.set_yticks([0, 1])
            inset.set_xticks(range(4), [str(j) for j in data[key + "_rank"]])
            inset.tick_params(labelsize=6.4, length=1.5, pad=1)
            inset.set_title("Mass at $0$", fontsize=7.1, pad=1)
            inset.set_facecolor("white")
    axes[0].set_ylabel("Density", labelpad=2)
    handles = [
        Line2D(
            [], [], color=c, lw=1.4, ls=(0, (4, 2)) if i == 3 else "-", label=LABELS[i]
        )
        for i, c in enumerate(COLORS)
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=WEIGHTED_COLOR,
            lw=1.5,
            ls=WEIGHTED_STYLE,
            label="$\\mu_\\omega$",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=1.8,
        columnspacing=1.5,
    )
    stem = "directional_measures_loglog"
    stem = "directional_measures_semilogy"
    for ext in ("png", "pdf"):
        fig.savefig(HERE / "output" / f"{stem}.{ext}", dpi=300)


if __name__ == "__main__":
    main()
