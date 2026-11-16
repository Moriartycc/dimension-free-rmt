"""Paper-width two-by-six heat maps; unchanged residuals and normalization."""

from plot_helpers import HERE
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext
from plot_helpers import outline_region
from matplotlib.ticker import NullLocator


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "hatch.linewidth": 0.3,
        }
    )
    data = {
        law: dict(np.load(HERE / "completed" / law / "endpoint_norms.npz"))
        for law in ("power", "critical")
    }
    limits = json.loads((HERE / "completed" / "l1_l2_diagnostics.json").read_text())[
        "shared_color_limits"
    ]
    norm = LogNorm(*limits)
    fig = plt.figure(figsize=(10.4, 4.3))
    gs = fig.add_gridspec(
        2,
        7,
        width_ratios=[1, 1, 1, 0.20, 1, 1, 1],
        left=0.085,
        right=0.91,
        bottom=0.21,
        top=0.82,
        wspace=0.13,
        hspace=0.23,
    )
    axes = np.empty((2, 6), dtype=object)
    for group, law in enumerate(("power", "critical")):
        d = data[law]
        x, y = d["x"], d["y"]
        xe = np.r_[x - 0.125, x[-1] + 0.125]
        factor = np.sqrt(y[1] / y[0])
        ye = np.r_[y / factor, y[-1] * factor]
        for row, key in enumerate(("l1", "l2")):
            for method in range(3):
                col = group * 3 + method
                ax = fig.add_subplot(gs[row, group * 4 + method])
                axes[row, col] = ax
                values = d[key]
                mesh = ax.pcolormesh(
                    xe,
                    ye,
                    values[method],
                    norm=norm,
                    cmap="viridis",
                    shading="flat",
                    rasterized=True,
                )
                outline_region(ax, values[method] == values.min(0), xe, ye)
                ax.set_yscale("log")
                ax.set_ylim(ye[0], ye[-1])
                ax.set_xlim(xe[0], xe[-1])
                ax.set_yticks([0.005, 0.01, 0.1, 1], ["0.005", "0.01", "0.1", "1"])
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_xticks([x[0], x[12], x[-1]])
                ax.tick_params(length=2, pad=1.6)
                if col != 0:
                    ax.tick_params(axis="y", labelleft=False)
                if row == 0:
                    ax.set_title(["Backward", "Linear", "Forward"][method], pad=5)
                    ax.tick_params(axis="x", labelbottom=False)
                else:
                    ax.set_xlabel(r"$\mathrm{Re}\,\zeta$", labelpad=2)
                if col == 0:
                    ax.set_ylabel(r"$\mathrm{Im}\,\zeta$", labelpad=2)
                for left, right in d["support"]:
                    lo, hi = max(left, xe[0]), min(right, xe[-1])
                    if lo >= hi:
                        continue
                    ax.plot(
                        [lo, hi],
                        [0, 0],
                        transform=ax.get_xaxis_transform(),
                        color="#bd2870",
                        lw=2.5,
                        solid_capstyle="butt",
                        clip_on=False,
                        zorder=6,
                    )
                    if right > xe[-1]:
                        ax.plot(
                            hi,
                            0,
                            marker=">",
                            ms=4,
                            color="#bd2870",
                            transform=ax.get_xaxis_transform(),
                            clip_on=False,
                            zorder=6,
                        )
        left = axes[0, group * 3].get_position().x0
        right = axes[0, group * 3 + 2].get_position().x1
        title = (
            r"Power law, $\alpha=0.2$"
            if law == "power"
            else r"Critical law, $\beta=0.2$"
        )
        fig.text((left + right) / 2, 0.95, title, ha="center", va="center", fontsize=14)
    for row, label in enumerate((r"$L^1$ error", r"$L^2$ error")):
        box = axes[row, 0].get_position()
        fig.text(
            0.015,
            (box.y0 + box.y1) / 2,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12.5,
        )
    bottom = axes[1, 5].get_position().y0
    top = axes[0, 5].get_position().y1
    bar = fig.colorbar(
        mesh,
        cax=fig.add_axes([0.929, bottom, 0.012, top - bottom]),
        format=LogFormatterMathtext(),
        ticks=[1e-3, 1e-2, 1e-1, 1],
    )
    bar.ax.tick_params(labelsize=10.5, length=2, pad=2)
    bar.set_label("Normalized error", fontsize=11, labelpad=4)
    fig.legend(
        [
            Patch(
                facecolor="#435d85",
                edgecolor=(1, 1, 1, 0.7),
                hatch="///",
                linewidth=0.6,
            ),
            Line2D([], [], color="#bd2870", lw=2.5),
        ],
        ["Lowest error among the three methods", "Predicted support on the real axis"],
        loc="lower center",
        bbox_to_anchor=(0.51, 0.015),
        frameon=False,
        fontsize=11,
        ncol=2,
        columnspacing=2,
    )
    for ext in ("png", "pdf"):
        fig.savefig(HERE / "output" / f"endpoint_l1_l2_combined.{ext}", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
