"""Surface-aware visibility for a fixed orthographic scientific figure."""

import numpy as np
from mpl_toolkits.mplot3d import proj3d


def surface_triangles(surface):
    faces = surface._faces
    invalid = np.broadcast_to(surface._invalid_vertices, faces.shape[:2])
    triangles = []
    for poly, bad in zip(faces, invalid):
        poly = poly[~bad]
        triangles.extend([poly[[0, j, j + 1]] for j in range(1, len(poly) - 1)])
    return np.asarray(triangles)


def visible_points(points, triangles, matrix):
    """Compare point depth against the frontmost projected surface triangle.

    A small local geometric tolerance avoids hiding a curve beneath its own
    piecewise-linear display mesh. More distant foreground hills still occlude it.
    """
    points = np.asarray(points, float)
    result = np.zeros(len(points), bool)
    valid = np.flatnonzero(np.isfinite(points).all(1))
    if not len(valid):
        return result
    flat = triangles.reshape(-1, 3)
    projected = np.array(proj3d.proj_transform(*flat.T, matrix)).T.reshape(-1, 3, 3)
    lower = projected[:, :, :2].min(1)
    upper = projected[:, :, :2].max(1)
    a, b, c = projected[:, 0], projected[:, 1], projected[:, 2]
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (
        a[:, 1] - c[:, 1]
    )
    usable = abs(det) > 1e-15
    for start in range(0, len(valid), 128):
        ids = valid[start : start + 128]
        world = points[ids]
        p = np.array(proj3d.proj_transform(*world.T, matrix)).T
        possible = (
            usable[:, None]
            & (p[None, :, 0] >= lower[:, None, 0] - 1e-12)
            & (p[None, :, 0] <= upper[:, None, 0] + 1e-12)
            & (p[None, :, 1] >= lower[:, None, 1] - 1e-12)
            & (p[None, :, 1] <= upper[:, None, 1] + 1e-12)
        )
        ti, pi = np.nonzero(possible)
        u = (
            (b[ti, 1] - c[ti, 1]) * (p[pi, 0] - c[ti, 0])
            + (c[ti, 0] - b[ti, 0]) * (p[pi, 1] - c[ti, 1])
        ) / det[ti]
        v = (
            (c[ti, 1] - a[ti, 1]) * (p[pi, 0] - c[ti, 0])
            + (a[ti, 0] - c[ti, 0]) * (p[pi, 1] - c[ti, 1])
        ) / det[ti]
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        ti, pi, u, v = ti[inside], pi[inside], u[inside], v[inside]
        w = 1 - u - v
        depth = u * a[ti, 2] + v * b[ti, 2] + w * c[ti, 2]
        nearest = np.full(len(ids), np.inf)
        np.minimum.at(nearest, pi, depth)
        front = depth <= nearest[pi] + 1e-12
        closest = np.full((len(ids), 3), np.nan)
        closest[pi[front]] = (
            u[front, None] * triangles[ti[front], 0]
            + v[front, None] * triangles[ti[front], 1]
            + w[front, None] * triangles[ti[front], 2]
        )
        distance = np.linalg.norm(
            (closest - world) / np.array([4.0, 448.0, 1.5]), axis=1
        )
        hidden = (nearest < p[:, 2] - 1e-5) & (distance > 0.025)
        result[ids] = ~hidden
    return result


def apply_surface_visibility(ax, surface, curves_only=False):
    triangles = surface_triangles(surface)
    matrix = ax.get_proj()
    chunks = []
    artists = []
    for line in ax.lines:
        if curves_only and line.get_zorder() < 8:
            continue
        coords = np.column_stack(line.get_data_3d())
        chunks.append(coords)
        artists.append(("line", line, len(coords)))
    for coll in ax.collections:
        if not curves_only and hasattr(coll, "_offsets3d"):
            coords = np.column_stack(coll._offsets3d)
            chunks.append(coords)
            artists.append(("scatter", coll, len(coords)))
    if not chunks:
        return dict(
            hidden_line_points=0,
            hidden_curve_points=0,
            hidden_markers=0,
            faint_curve_points=0,
        )
    all_points = np.concatenate(chunks)
    visible = visible_points(all_points, triangles, matrix)
    offset = 0
    counts = {
        "hidden_line_points": 0,
        "hidden_curve_points": 0,
        "hidden_markers": 0,
        "faint_curve_points": 0,
    }
    for (kind, artist, count), coords in zip(artists, chunks):
        mask = visible[offset : offset + count]
        offset += count
        hidden = np.isfinite(coords).all(1) & ~mask
        if kind == "line":
            counts["hidden_line_points"] += int(hidden.sum())
            if artist.get_zorder() >= 8:
                counts["hidden_curve_points"] += int(hidden.sum())
            if artist.get_color() in ("#b3154f", "#c62828") and hidden.any():
                # A faint path without the white halo shows through the hill.
                # Include one neighboring point at each visibility boundary.
                ghost = hidden.copy()
                ghost[:-1] |= hidden[1:]
                ghost[1:] |= hidden[:-1]
                faint = coords.copy()
                faint[~ghost] = np.nan
                ax.plot(
                    *faint.T,
                    color=artist.get_color(),
                    alpha=0.45,
                    lw=1.4,
                    ls=(0, (3, 2)) if artist.is_dashed() else "-",
                    zorder=7,
                    clip_on=artist.get_clip_on(),
                )
                counts["faint_curve_points"] += int(hidden.sum())
            coords[~mask] = np.nan
            artist.set_data_3d(*coords.T)
        else:
            counts["hidden_markers"] += int(hidden.sum())
            artist._offsets3d = tuple(coords[mask].T)
    return counts
