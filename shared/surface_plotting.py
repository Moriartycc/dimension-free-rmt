"""Polygon clipping for risk surfaces at the displayed height limits."""

import numpy as np
from plotting import plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CASES = [("power", 0.2), ("critical", 0.2), ("power", 0.05), ("critical", 0.05)]
NS = np.array([64, 96, 128, 192, 256, 320, 384, 448, 512])
ZMIN, ZMAX = 0.5, 1.5


def clipped_surface(ax, x, y, z, cmap, norm):
    # Clip actual surface polygons at the view limits, rather than replacing
    # high risks by a flat ceiling or leaving a jagged missing-cell boundary.
    faces = []
    colors = []
    for i in range(z.shape[0] - 1):
        for j in range(z.shape[1] - 1):
            poly = [
                np.array([x[a, b], y[a, b], z[a, b]])
                for a, b in ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
            ]
            for bound, sign in ((ZMAX, -1), (ZMIN, 1)):
                result = []
                for a, b in zip(poly, poly[1:] + poly[:1]):
                    ina = sign * (a[2] - bound) >= 0
                    inb = sign * (b[2] - bound) >= 0
                    if ina:
                        result.append(a)
                    if ina != inb:
                        result.append(a + (b - a) * (bound - a[2]) / (b[2] - a[2]))
                poly = result
                if not poly:
                    break
            if len(poly) >= 3:
                faces.append(poly)
                colors.append(cmap(norm(np.mean([v[2] for v in poly]))))
    surf = Poly3DCollection(
        faces,
        facecolors=colors,
        edgecolors="none",
        linewidths=0,
        antialiased=False,
        alpha=1,
        rasterized=True,
    )
    ax.add_collection3d(surf)


__all__ = ["np", "plt", "CASES", "NS", "ZMIN", "ZMAX", "clipped_surface"]
