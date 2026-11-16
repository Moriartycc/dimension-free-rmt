from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))
"""Stacked theory-only GF surfaces for q=28,32,36,40."""
from pathlib import Path
import argparse

HERE = Path(__file__).resolve().parent
import surface_plotting as base

np, plt = base.np, base.plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import PchipInterpolator
from mpl_toolkits.mplot3d import proj3d
from scipy.optimize import brentq
from depth_visibility import apply_surface_visibility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, choices=(28, 32, 36, 40))
    args = ap.parse_args()
    qs = (args.q,) if args.q else (28, 32, 36, 40)
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    base.ZMIN = 0.0
    base.ZMAX = 1.5
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
    height = 0.952 + 2.924 * len(qs)
    fig = plt.figure(figsize=(14.8, height))
    norm = Normalize(0, 1.5)
    cmap = plt.get_cmap("YlGnBu_r")
    ng = np.linspace(64, 512, 75)
    depth_report = []
    for col, (law, p) in enumerate(base.CASES):
        ds = [
            dict(np.load(HERE / "supplement" / f"{law}_{p:g}_n{n}.npz"))
            for n in base.NS
        ]
        s = ds[0]["s"]
        ss, nn = np.meshgrid(s, ng)
        title = (
            rf"Power, $\alpha={p:g}$" if law == "power" else rf"Critical, $\beta={p:g}$"
        )
        fig.text(
            0.142 + col * 0.223,
            1 - 0.102 / height,
            title,
            ha="center",
            va="top",
            fontsize=21.5,
        )
        for row, q in enumerate(qs):
            qi = int(np.flatnonzero(ds[0]["q"] == q)[0])
            rect = [
                0.027 + col * 0.223,
                (0.816 + (len(qs) - 1 - row) * 2.924) / height,
                0.223,
                2.584 / height,
            ]
            ax = fig.add_axes(rect, projection="3d", computed_zorder=False)
            zz = PchipInterpolator(
                base.NS, np.stack([d["ratio"][:, qi] for d in ds]), axis=0
            )(ng)
            floor_colors = cmap(norm(zz))
            floor_colors[..., :3] = 0.2 * floor_colors[..., :3] + 0.8
            floor = ax.plot_surface(
                ss,
                nn,
                np.zeros_like(zz),
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
            surface = ax.collections[-1]
            surface.set_zorder(2)
            shown = zz.copy()
            shown[(shown > 1.5) | (shown < 0)] = np.nan
            for idx in range(0, len(ng), 6):
                ax.plot(
                    s,
                    np.full_like(s, ng[idx]),
                    shown[idx],
                    color="#18334c",
                    lw=0.3,
                    alpha=0.4,
                    zorder=3,
                )
            for idx in range(0, len(s), 4):
                ax.plot(
                    np.full_like(ng, s[idx]),
                    ng,
                    shown[:, idx],
                    color="#18334c",
                    lw=0.3,
                    alpha=0.4,
                    zorder=3,
                )
            for nedge in (64, 512):
                ax.plot(
                    [0, 4],
                    [nedge, nedge],
                    [1, 1],
                    color=".45",
                    ls="--",
                    lw=0.8,
                    zorder=3,
                )
            opt = np.array([d["optimum_s"][qi] for d in ds])
            optz = np.array([d["optimum_ratio"][qi] for d in ds])
            finite = np.isfinite(opt) & (opt > 0) & (1 - optz > 1e-8)
            # Smooth within branches; dashed red connectors mark branch changes.
            groups = []
            group = []
            for i in range(len(base.NS)):
                if not finite[i]:
                    if group:
                        groups.append(group)
                        group = []
                    continue
                if group and abs(np.log(opt[i] / opt[group[-1]])) > np.log(50):
                    groups.append(group)
                    group = []
                group.append(i)
            if group:
                groups.append(group)
            branch_ends = []
            for group in groups:
                if len(group) < 2:
                    continue
                time_curve = PchipInterpolator(base.NS[group], np.log(opt[group]))
                risk_curve = PchipInterpolator(base.NS[group], optz[group])
                nv = np.linspace(base.NS[group[0]], base.NS[group[-1]], 180)
                crossings = []
                for j, k in zip(group[:-1], group[1:]):
                    if (opt[j] - 4) * (opt[k] - 4) < 0:
                        crossings.append(
                            brentq(
                                lambda n: float(time_curve(n) - np.log(4)),
                                base.NS[j],
                                base.NS[k],
                            )
                        )
                nv = np.unique(np.r_[nv, crossings])
                sv = np.exp(time_curve(nv))
                sv[np.isclose(sv, 4, rtol=0, atol=1e-10)] = 4.0
                zv = risk_curve(nv)
                sv[(sv > 4) | (zv < 0) | (zv > 1.5)] = np.nan
                valid = np.flatnonzero(np.isfinite(sv))
                if len(valid):
                    branch_ends.append(
                        (
                            group,
                            np.array([sv[valid[0]], nv[valid[0]], zv[valid[0]]]),
                            np.array([sv[valid[-1]], nv[valid[-1]], zv[valid[-1]]]),
                        )
                    )
                ax.plot(sv, nv, zv, color="white", lw=3.4, zorder=8)
                ax.plot(sv, nv, zv, color="#b3154f", lw=1.8, zorder=9)
            tied_inf = np.isinf(opt) | (1 - optz <= 1e-8)
            infinity_labels = []
            for i in range(len(base.NS) - 1):
                jump = (
                    finite[i]
                    and finite[i + 1]
                    and abs(np.log(opt[i + 1] / opt[i])) > np.log(50)
                )
                inf_jump = tied_inf[i] != tied_inf[i + 1]
                if not (jump or inf_jump):
                    continue
                tx = np.array([opt[i], opt[i + 1]], float)
                tz = optz[i : i + 2].copy()
                ny = base.NS[i : i + 2].astype(float)
                if inf_jump:
                    # Extend from the actual visible END of the finite branch,
                    # never from an interior sample (which would create a fork).
                    finite_index = i + 1 if tied_inf[i] else i
                    match = next(
                        (entry for entry in branch_ends if finite_index in entry[0]),
                        None,
                    )
                    if match is None:
                        continue
                    anchor = match[1] if tied_inf[i] else match[2]
                    tx = np.array([anchor[0], 4.45])
                    ny = np.full(2, anchor[1])
                    tz = np.full(2, anchor[2])
                    assert np.ptp(ny) == 0 and np.ptp(tz) == 0
                if min(tx) > 4:
                    continue
                tt = np.linspace(0, 1, 101)
                xx = tx[0] + tt * (tx[1] - tx[0])
                yy = ny[0] + tt * (ny[1] - ny[0])
                zzline = tz[0] + tt * (tz[1] - tz[0])
                xx[(xx > (4.45 if inf_jump else 4)) | (zzline < 0) | (zzline > 1.5)] = (
                    np.nan
                )
                ax.plot(
                    xx,
                    yy,
                    zzline,
                    color="white",
                    lw=3.4,
                    ls=(0, (3, 2)),
                    zorder=8,
                    clip_on=not inf_jump,
                )
                ax.plot(
                    xx,
                    yy,
                    zzline,
                    color="#c62828",
                    lw=1.8,
                    ls=(0, (3, 2)),
                    zorder=9,
                    clip_on=not inf_jump,
                )
                # Finite jumps mark both sampled endpoints; infinity extensions
                # mark only the finite anchor, keeping the infinity end clear.
                visible = (tx <= 4) & (tz >= 0) & (tz <= 1.5)
                if inf_jump:
                    visible[1] = False
                ax.scatter(
                    tx[visible],
                    ny[visible],
                    tz[visible],
                    marker="^",
                    s=65,
                    color="#c62828",
                    edgecolors="white",
                    linewidth=0.65,
                    depthshade=False,
                    zorder=11,
                )
                if inf_jump:
                    infinity_labels.append((4.45, ny[0], tz[0]))
            inside = finite & (opt <= 4) & (optz <= 1.5)
            ax.scatter(
                opt[inside],
                base.NS[inside],
                optz[inside],
                s=10,
                color="#b3154f",
                edgecolors="white",
                linewidth=0.4,
                depthshade=False,
                zorder=10,
            )
            # Numerically tied minima use the infinite-time representative.
            if np.all(np.isinf(opt) | (1 - optz <= 1e-8)):
                ax.text2D(
                    0.18,
                    0.96,
                    r"$\widetilde t_{\mathrm{opt}}=\infty$",
                    transform=ax.transAxes,
                    fontsize=16.4,
                    color="black",
                )
            ax.set_xlim(0, 4)
            ax.set_ylim(512, 64)
            ax.set_zlim(0, 1.5)
            ax.set_xticks([0, 2, 4])
            ax.set_yticks([64, 256, 512])
            ax.set_zticks([0, 0.5, 1, 1.5])
            ax.set_ylabel("$n$", labelpad=10)
            ax.text2D(
                0.45,
                0.09,
                r"$\widetilde t$",
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
                axis.pane.set_edgecolor("#eee")
                axis._axinfo["grid"].update(color=(0.6, 0.6, 0.6, 0.2), linewidth=0.45)
                axis.line.set_color("#b0b0b0")
            counts = apply_surface_visibility(ax, surface)
            depth_report.append(dict(law=law, p=p, q=q, **counts))
            for position in infinity_labels:
                px, py, _ = proj3d.proj_transform(*position, ax.get_proj())
                ax.annotate(
                    r"$\infty$",
                    xy=(px, py),
                    xycoords="data",
                    xytext=(-9, 13),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    color="black",
                    fontsize=16.4,
                    zorder=12,
                )
    for row, q in enumerate(qs):
        fig.text(
            0.014,
            (2.074 + (len(qs) - 1 - row) * 2.924) / height,
            rf"$q={q}$",
            rotation=90,
            ha="center",
            va="center",
            fontsize=21.5,
        )
    cax = fig.add_axes([0.927, 0.25, 0.010, 0.5])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.set_label(r"$R_{\mathrm{GF}}/R_0$", fontsize=20.5, labelpad=4)
    bar.set_ticks([0, 0.5, 1, 1.5])
    bar.ax.tick_params(labelsize=16.4, length=2)
    fig.legend(
        handles=[
            Patch(facecolor=cmap(norm(1)), label="Theoretical prediction"),
            Line2D([], [], color="#b3154f", lw=2, label="Predicted optimum"),
            Line2D([], [], color="#b3154f", lw=1.4, alpha=0.45, label="Hidden path"),
            Line2D(
                [],
                [],
                color="#c62828",
                marker="^",
                ls="--",
                lw=1.8,
                ms=8,
                label="Jump point",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.49, 0.102 / height),
        frameon=False,
        ncol=4,
        fontsize=18.5,
        handlelength=1.7,
        columnspacing=1.5,
    )
    stem = (
        "gf_theory_q28_q32_q36_q40_4x4"
        if args.q is None
        else f"gf_theory_q{args.q}_1x4"
    )
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=220)
    import json

    (out / f"{stem}_depth_checks.json").write_text(json.dumps(depth_report, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
