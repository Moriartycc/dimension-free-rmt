from pathlib import Path

HERE = Path(__file__).resolve().parent
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib import patheffects


def outline_region(ax, mask, xedges, yedges):
    """Lightly hatch selected cells and outline their union."""
    for iy, ix in zip(*np.nonzero(mask)):
        ax.add_patch(
            Rectangle(
                (xedges[ix], yedges[iy]),
                xedges[ix + 1] - xedges[ix],
                yedges[iy + 1] - yedges[iy],
                facecolor="none",
                edgecolor=(1, 1, 1, 0.45),
                linewidth=0,
                hatch="///",
                zorder=2,
            )
        )
    horizontal = np.diff(np.pad(mask.astype(int), ((1, 1), (0, 0))), axis=0)
    vertical = np.diff(np.pad(mask.astype(int), ((0, 0), (1, 1))), axis=1)
    segments = [
        [(xedges[ix], yedges[iy]), (xedges[ix + 1], yedges[iy])]
        for iy, ix in zip(*np.nonzero(horizontal))
    ]
    segments += [
        [(xedges[ix], yedges[iy]), (xedges[ix], yedges[iy + 1])]
        for iy, ix in zip(*np.nonzero(vertical))
    ]
    boundary = LineCollection(segments, colors="white", linewidths=1.15, zorder=4)
    boundary.set_path_effects(
        [patheffects.Stroke(linewidth=2.0, foreground="#343434"), patheffects.Normal()]
    )
    ax.add_collection(boundary)
