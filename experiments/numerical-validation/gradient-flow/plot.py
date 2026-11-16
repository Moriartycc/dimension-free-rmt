from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))
"""Compact GF theory surfaces and the paired 400-design empirical cloud."""
from pathlib import Path
import sys, argparse

HERE = Path(__file__).resolve().parent
import surface_plotting as base

np, plt = (base.np, base.plt)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import PchipInterpolator
from depth_visibility import apply_surface_visibility


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, choices=(16, 24))
    args = parser.parse_args()
    qs = (args.q,) if args.q else (16, 24)
    obs = np.load(HERE / "gf_aggregated.npz")
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "font.size": 18.5,
            "axes.labelsize": 20.5,
            "xtick.labelsize": 16.4,
            "ytick.labelsize": 16.4,
        }
    )
    fig = plt.figure(figsize=(14.8, 6.8 if len(qs) == 2 else 3.9))
    norm = Normalize(0.5, 1.5)
    cmap = plt.get_cmap("YlGnBu_r")
    ng = np.linspace(64, 512, 75)
    depth_report = []
    for col, (law, p) in enumerate(base.CASES):
        name = f"{law}_{p:g}"
        ds = [dict(np.load(HERE / "theory" / f"{name}_n{n}.npz")) for n in base.NS]
        s = ds[0]["s"]
        assert np.allclose(s, obs["s"])
        ss, nn = np.meshgrid(s, ng)
        title = (
            f"Power, $\\alpha={p:g}$" if law == "power" else f"Critical, $\\beta={p:g}$"
        )
        fig.text(
            0.142 + col * 0.223, 0.985, title, ha="center", va="top", fontsize=21.5
        )
        for row, q in enumerate(qs):
            qi = int(np.flatnonzero(obs["q"] == q)[0])
            rect = (
                [0.027 + col * 0.223, 0.55 - row * 0.43, 0.223, 0.38]
                if len(qs) == 2
                else [0.027 + col * 0.223, 0.25, 0.223, 0.62]
            )
            ax = fig.add_axes(rect, projection="3d", computed_zorder=False)
            raw = np.stack([d["ratio"][:, qi] for d in ds])
            zz = PchipInterpolator(base.NS, raw, axis=0)(ng)
            risk = obs[name][:, :, :, 2, qi]
            den = obs[name + "_infinity"][:, :, 2, qi].mean(0)
            cloud = risk / den[None, :, None]
            floor_colors = cmap(norm(zz))
            floor_opacity = 0.2
            floor_colors[..., :3] = (
                floor_opacity * floor_colors[..., :3] + 1 - floor_opacity
            )
            floor = ax.plot_surface(
                ss,
                nn,
                np.full_like(zz, 0.5),
                facecolors=floor_colors,
                rstride=1,
                cstride=1,
                shade=False,
                linewidth=0,
                antialiased=False,
                rasterized=True,
                zorder=1,
            )
            floor.set_edgecolor("none")
            base.clipped_surface(ax, ss, nn, zz, cmap, norm)
            occluding_surface = ax.collections[-1]
            occluding_surface.set_zorder(2)
            ax.collections[-1].remove()
            ensemble = PchipInterpolator(obs["n"], cloud, axis=1)(ng)
            low, high = np.quantile(ensemble, [0.1, 0.9], axis=0)
            center = ensemble.mean(0)
            assert np.all(low <= center + 1e-12) and np.all(center <= high + 1e-12)
            levels = ((1.0, 0.15), (0.7, 0.23), (0.4, 0.36), (0.15, 0.5))
            cloud_color = "#8055ad"

            def cloud_cmap(value):
                return cloud_color

            for layer, (distance, opacity) in enumerate(levels):
                sheet = center + distance * (low - center)
                base.clipped_surface(ax, ss, nn, sheet, cloud_cmap, norm)
                ax.collections[-1].set_alpha(opacity)
                ax.collections[-1].set_zorder(2 + 0.15 * layer)
            base.clipped_surface(ax, ss, nn, center, cmap, norm)
            occluding_surface = ax.collections[-1]
            occluding_surface.set_zorder(4.2)
            for layer, (distance, opacity) in enumerate(reversed(levels)):
                sheet = center + distance * (high - center)
                base.clipped_surface(ax, ss, nn, sheet, cloud_cmap, norm)
                ax.collections[-1].set_alpha(opacity)
                ax.collections[-1].set_zorder(3.2 + 0.15 * layer)
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            for ni in (0, -1):
                for bound in (low, high):
                    for d0, d1, opacity in (
                        (0, 0.2, 0.85),
                        (0.2, 0.4, 0.65),
                        (0.4, 0.6, 0.45),
                        (0.6, 0.8, 0.3),
                        (0.8, 1, 0.18),
                    ):
                        inner = center + d0 * (bound - center)
                        outer = center + d1 * (bound - center)
                        faces = []
                        for j in range(len(s) - 1):
                            values = np.r_[inner[ni, j : j + 2], outer[ni, j : j + 2]]
                            if values.min() < 0.5 or values.max() > 1.5:
                                continue
                            faces.append(
                                [
                                    (s[j], ng[ni], inner[ni, j]),
                                    (s[j + 1], ng[ni], inner[ni, j + 1]),
                                    (s[j + 1], ng[ni], outer[ni, j + 1]),
                                    (s[j], ng[ni], outer[ni, j]),
                                ]
                            )
                        ax.add_collection3d(
                            Poly3DCollection(
                                faces,
                                facecolor=cloud_color,
                                edgecolor="none",
                                alpha=opacity,
                                rasterized=True,
                                zorder=4.5,
                            )
                        )
                for bound in (low, high):
                    edge = bound[ni].copy()
                    edge[(edge < 0.5) | (edge > 1.5)] = np.nan
                    ax.plot(
                        s,
                        np.full_like(s, ng[ni]),
                        edge,
                        color=cloud_color,
                        lw=0.9,
                        alpha=0.8,
                        zorder=5,
                    )
            for bound in (low, high):
                edge = bound[:, -1].copy()
                edge[(edge < 0.5) | (edge > 1.5)] = np.nan
                ax.plot(
                    np.full_like(ng, 4),
                    ng,
                    edge,
                    color=cloud_color,
                    lw=0.9,
                    alpha=0.8,
                    zorder=5,
                )
            shown = zz.copy()
            shown[(shown > 1.5) | (shown < 0.5)] = np.nan
            for idx in range(0, len(ng), 6):
                ax.plot(
                    s,
                    ng[idx] * np.ones_like(s),
                    shown[idx],
                    color="#18334c",
                    lw=0.35,
                    alpha=0.7,
                    zorder=6,
                )
            for idx in range(0, len(s), 4):
                ax.plot(
                    s[idx] * np.ones_like(ng),
                    ng,
                    shown[:, idx],
                    color="#18334c",
                    lw=0.35,
                    alpha=0.7,
                    zorder=6,
                )
            for nedge in (64, 512):
                ax.plot(
                    [0, 4],
                    [nedge, nedge],
                    [1, 1],
                    color=".45",
                    lw=0.85,
                    ls="--",
                    zorder=3,
                )
            opt = np.array([d["optimum_s"][qi] for d in ds])
            optz = np.array([d["optimum_ratio"][qi] for d in ds])
            finite = np.isfinite(opt) & (opt > 0)
            if finite.sum() >= 2:
                nv = np.linspace(base.NS[finite].min(), base.NS[finite].max(), 180)
                sv = np.exp(PchipInterpolator(base.NS[finite], np.log(opt[finite]))(nv))
                zv = PchipInterpolator(base.NS[finite], optz[finite])(nv)
                sv[(sv > 4) | (zv < 0.5) | (zv > 1.5)] = np.nan
                ax.plot(sv, nv, zv, color="white", lw=3.4, zorder=8)
                ax.plot(sv, nv, zv, color="#b3154f", lw=1.8, zorder=9)
            if np.all(np.isinf(opt)):
                ax.text2D(
                    0.18,
                    0.96,
                    "$\\widetilde t_{\\mathrm{opt}}=\\infty$",
                    transform=ax.transAxes,
                    fontsize=16.4,
                    color="black",
                )
            ax.set_xlim(0, 4)
            ax.set_ylim(512, 64)
            ax.set_zlim(0.5, 1.5)
            ax.set_xticks([0, 2, 4])
            ax.set_yticks([64, 256, 512])
            ax.set_zticks([0.5, 1, 1.5])
            ax.set_xlabel("")
            ax.set_ylabel("$n$", labelpad=10)
            ax.text2D(
                0.45,
                0.09,
                "$\\widetilde t$",
                transform=ax.transAxes,
                ha="center",
                va="top",
                rotation=0,
                fontsize=20.5,
            )
            ax.tick_params(axis="x", pad=-2)
            ax.tick_params(axis="y", pad=4)
            ax.tick_params(axis="z", pad=0, labelsize=16.4)
            ax.view_init(elev=24, azim=-58)
            ax.set_proj_type("ortho")
            ax.set_box_aspect((1.35, 1, 0.9), zoom=1.07)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False
                axis.pane.set_edgecolor("#eeeeee")
                axis._axinfo["grid"].update(color=(0.6, 0.6, 0.6, 0.2), linewidth=0.45)
                axis.line.set_color("#b0b0b0")
            counts = apply_surface_visibility(ax, occluding_surface, curves_only=True)
            depth_report.append(dict(law=law, p=p, q=q, **counts))
    for row, q in enumerate(qs):
        fig.text(
            0.014,
            0.735 - row * 0.43 if len(qs) == 2 else 0.57,
            f"$q={q}$",
            rotation=90,
            ha="center",
            va="center",
            fontsize=21.5,
        )
    cax = fig.add_axes(
        [0.927, 0.24 if len(qs) == 2 else 0.31, 0.01, 0.56 if len(qs) == 2 else 0.48]
    )
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.set_label("$R_{\\mathrm{GF}}/R_0$", fontsize=20.5, labelpad=4)
    bar.set_ticks([0.5, 0.75, 1, 1.25, 1.5])
    bar.ax.tick_params(labelsize=16.4, length=2)
    handles = [
        Patch(facecolor=cmap(norm(1)), label="Theoretical prediction"),
        Line2D([], [], color="#b3154f", lw=2, label="Predicted optimum"),
        Line2D(
            [],
            [],
            color=".15",
            marker="o",
            mfc="white",
            ls="none",
            ms=5,
            label="Empirical mean",
        ),
        Line2D(
            [],
            [],
            color=".45",
            marker="o",
            alpha=0.5,
            ls="none",
            ms=5,
            label="Trial cloud",
        ),
    ]
    handles = [
        Patch(facecolor=cmap(norm(1)), label="Empirical mean surface"),
        Line2D([], [], color="#18334c", lw=1, label="Theoretical mesh"),
        Line2D([], [], color="#b3154f", lw=2, label="Predicted optimum"),
        Patch(facecolor="#8055ad", alpha=0.65, label="10-90% trial envelope"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.49, 0.015),
        frameon=False,
        ncol=4,
        fontsize=18.5,
        handlelength=1.7,
        columnspacing=1.2,
    )
    stem = (
        "gf_empirical_surfaces_2x4"
        if args.q is None
        else f"gf_empirical_surfaces_q{args.q}_1x4"
    )
    stem += "_envelope"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=220)
    import json

    (out / f"{stem}_depth_checks.json").write_text(json.dumps(depth_report, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
