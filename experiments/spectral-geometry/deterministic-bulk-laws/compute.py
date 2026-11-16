from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))
"""Finite-n deterministic bulk densities for the two exact Table 1 examples."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
import limiting_laws as previous
from common import np, brentq
from scipy.special import zeta
from numpy.polynomial.legendre import leggauss

P = 1.2
SIZES = (8, 16, 32, 64, 128, 256, 512, 1024)
WINDOW = {"power": (4.05, 14.0), "critical": (-1.6, 6.0)}


def sigma(j, law):
    j = np.asarray(j, dtype=float)
    return j**-P if law == "power" else 1 / (j * (1 + np.log(j)) ** P)


def center(n, end):
    # Euler–Maclaurin for the sum strictly past end. The integral is explicit.
    log = 1 + np.log(end)
    f = sigma(end, "critical")
    derivative = -f * (1 + P / log) / end
    remainder = log ** (1 - P) / (P - 1) - f / 2 - derivative / 12
    return sigma(np.arange(n + 1, end + 1), "critical").sum() + remainder


def boundary_solver(law, n, refinement):
    cutoff = max(512, 4 * n) * refinement
    # Keep the first tail moment analytically. Quadrature is needed only for
    # the remaining quadratic tail in the centered inverse Stieltjes map.
    nodes, quad_weights = leggauss(8 + 2 * refinement)
    panels = 36 if refinement == 1 else 44
    t = (np.arange(panels)[:, None] + (1 + nodes) / 2).ravel()
    tail_j = (cutoff + 0.5) * np.exp(t)
    j = np.r_[np.arange(1, cutoff + 1), tail_j]
    w = np.r_[np.ones(cutoff), tail_j * np.tile(quad_weights / 2, panels)] / n
    d = sigma(j, law) / sigma(n, law)
    wd2 = w * d**2
    a = float(n * sigma(n, law))
    b = center(n, 65536 * refinement) if law == "critical" else 0.0
    tail_shift = zeta(P, n + 1) / a if law == "power" else 0.0

    def edge_condition(u):
        with np.errstate(divide="ignore"):
            return np.sum(wd2 / (u - d) ** 2) - 1

    def boundary_point(t, is_edge=False):
        sq = (t - d) ** 2
        if is_edge or edge_condition(t) <= 0:
            h = 0.0
        else:

            def residual(h2):
                return np.sum(wd2 / (sq + h2)) - 1

            hi = max(1.0, abs(t)) ** 2
            while residual(hi) > 0:
                hi *= 4
            with np.errstate(divide="ignore"):
                h = np.sqrt(brentq(residual, 0, hi, xtol=1e-23, rtol=1e-14))
        u = complex(t, h)
        z = (
            u
            + tail_shift
            + np.sum(w[:n] * u * d[:n] / (u - d[:n]))
            + np.sum(wd2[n:] / (u - d[n:]))
        )
        return u, z

    return d, w, a, b, cutoff, edge_condition, boundary_point


def finite(law, n, refinement):
    d, w, a, b, cutoff, edge_condition, boundary_point = boundary_solver(
        law, n, refinement
    )
    wd2 = w * d**2

    def point(t, is_edge=False):
        u, z = boundary_point(t, is_edge)
        h = u.imag
        s = -1 / u if h else complex(-1 / t, 0.0)
        v = d * s
        small = abs(v) < 1e-4
        primitive = np.empty_like(v)
        primitive[small] = v[small] ** 2 * (0.5 + v[small] * (-2 / 3 + 0.75 * v[small]))
        primitive[~small] = np.log1p(v[~small]) - v[~small] / (1 + v[~small])
        cdf = (np.angle(s) - np.sum(w * primitive.imag)) / np.pi
        return z.real, s.imag / np.pi, cdf, abs(z.imag) / (1 + abs(z.real))

    lower = brentq(edge_condition, -max(100.0, d[0]), -1e-12, xtol=3e-15)
    # At the edge the square-root law is better resolved on a graded grid.
    right = 18.0
    du = np.linspace(0, np.sqrt(right - lower), 1001 * refinement) ** 2
    grid = np.unique(np.r_[lower + du, lower + np.geomspace(1e-10, 0.02, 54)])
    inside = np.array([edge_condition(t) > 0 for t in grid])
    edges = [lower]
    for i in np.flatnonzero(inside[1:] != inside[:-1]):
        if i:
            edges.append(brentq(edge_condition, grid[i], grid[i + 1], xtol=3e-15))
    grid = np.union1d(grid, edges)
    edge_set = set(edges)
    x, rho, cdf, err = np.array([point(t, t in edge_set) for t in grid]).T
    assert np.all(np.diff(x) > -1e-11) and np.min(np.diff(cdf)) > -1e-10
    assert (
        rho.min() >= 0
        and rho[0] == 0
        and abs(cdf[0]) < 1e-10
        and x[-1] > WINDOW[law][1]
    )
    assert err.max() < 1e-10
    sec = 2 * np.sum(wd2 / (lower - d) ** 3)
    coeff = np.sqrt(2 / (-sec)) / (np.pi * lower**2)
    dx = x - x[0]
    near = (dx > 1e-8) & (dx < 1e-6)
    rel_edge = np.max(abs(rho[near] / (coeff * np.sqrt(dx[near])) - 1))
    assert rel_edge < 2e-4
    mass_error = (
        abs(np.trapezoid(rho, x) - (cdf[-1] - cdf[0]))
        if hasattr(np, "trapezoid")
        else abs(np.trapz(rho, x) - (cdf[-1] - cdf[0]))
    )
    return np.array([x, rho, cdf]), dict(
        a_n=a,
        b_n=b,
        edge=x[0],
        cutoff=cutoff,
        boundary_residual=err.max(),
        edge_residual=abs(edge_condition(lower)),
        edge_coefficient=coeff,
        edge_expansion_rel_error=rel_edge,
        density_integral_vs_primitive_error=mass_error,
        bulk_window_mass=float(np.interp(WINDOW[law][1], x, cdf)),
    )


def main():
    previous.ALPHA = P - 1
    theta = np.linspace(1e-6, np.pi - 1e-6, 40001)
    curves, checks = {}, {}
    for law in ("power", "critical"):
        x, rho, cdf, edge = previous.limit(law, theta)
        curves[law + "_limit"] = np.vstack(
            [np.r_[edge, x], np.r_[0.0, rho], np.r_[0.0, cdf]]
        )
        checks[law] = {"limit_edge": edge, "finite": {}}
        for n in SIZES:
            coarse, _ = finite(law, n, 1)
            fine, info = finite(law, n, 2)
            xx = np.linspace(*WINDOW[law], 8001)

            def interp(arr, i):
                return np.interp(xx, arr[0], arr[i], left=0)

            info["refined_edge_change"] = abs(fine[0, 0] - coarse[0, 0])
            info["refined_density_change"] = np.max(
                abs(interp(fine, 1) - interp(coarse, 1))
            )
            info["refined_cdf_change"] = np.max(
                abs(interp(fine, 2) - interp(coarse, 2))
            )
            if law == "critical":
                info["center_refinement_change"] = abs(center(n, 65536) - info["b_n"])
            assert (
                info["refined_edge_change"] < 1e-6
                and info["refined_density_change"] < 1e-3
            )
            assert info["density_integral_vs_primitive_error"] < 2e-4
            curves[f"{law}_{n}"] = fine
            checks[law]["finite"][n] = info
            print(
                law,
                n,
                "edge",
                fine[0, 0],
                "delta density",
                info["refined_density_change"],
                flush=True,
            )
    np.savez_compressed(HERE / "curves.npz", **curves)
    (HERE / "validation.json").write_text(json.dumps(checks, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()

__all__ = [
    "HERE",
    "WINDOW",
    "np",
    "brentq",
    "sigma",
    "boundary_solver",
    "finite",
    "main",
]
