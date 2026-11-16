from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))
"""Directional covariance-scale laws using the Figure 2 inverse Stieltjes map."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
FIG2 = HERE.parent / "deterministic-bulk-laws"
sys.path.insert(0, str(FIG2))
from compute import boundary_solver, sigma, np, brentq
from scipy.optimize import minimize_scalar
from scipy.integrate import quad, trapezoid

PANELS = (("power", 8), ("critical", 8), ("power", 64), ("critical", 64))


def solve(law, n, refinement):
    d, w, a, b, cutoff, condition, point = boundary_solver(law, n, refinement)
    sn = a / n
    ranks = np.array([1, n // 4, n // 2, n])
    dj = d[ranks - 1]
    lower = brentq(condition, -max(100.0, d[0]), -1e-12, xtol=3e-15)
    upper = brentq(condition, d[0], 3 * d[0], xtol=3e-14)
    edges = [lower, upper]
    # Each pole alone puts [d_j(1-1/sqrt(n)), d_j(1+1/sqrt(n))] in support
    # parameter space. On remaining candidate gaps the boundary condition is convex.
    for high, low in zip(d[: cutoff - 1], d[1:cutoff]):
        left, right = low * (1 + 1 / np.sqrt(n)), high * (1 - 1 / np.sqrt(n))
        if left >= right:
            continue
        minimum = minimize_scalar(
            condition, bounds=(left, right), method="bounded", options={"xatol": 1e-13}
        )
        if minimum.fun < 0:
            edges.extend(
                [brentq(condition, low, minimum.x), brentq(condition, minimum.x, high)]
            )
    bands = np.array(sorted(edges)).reshape(-1, 2)

    def negative_inverse(target):
        return brentq(
            lambda t: sn * point(t, True)[1].real + b / n - target,
            -max(100.0, 4 * d[0] + 2 * abs(target) / sn),
            lower,
            xtol=3e-14,
        )

    v0 = negative_inverse(0.0)
    atom = -v0 / (dj - v0)
    assert np.all((atom > 0) & (atom < 1))
    arrays = []
    weighted_bands = []
    support = []
    boundary_error = 0.0
    for left, right in bands:
        angles = np.linspace(0, np.pi, 1024 * refinement + 1)
        grid = left + (right - left) * np.sin(angles / 2) ** 2
        grid[0], grid[-1] = left, right
        pairs = np.array(
            [point(t, k == 0 or k == len(grid) - 1) for k, t in enumerate(grid)]
        )
        v, z = pairs.T
        boundary_error = max(boundary_error, np.max(abs(z.imag) / (1 + abs(z.real))))
        t = sn * z.real + b / n
        density = dj[:, None] * v.imag / (np.pi * t * abs(dj[:, None] - v) ** 2)
        assert t.min() > 0 and np.all(np.diff(t) > 0) and density.min() >= 0
        assert np.isfinite(density).all()
        arrays.append(np.vstack((t, density)))
        # The boundary equation sums all directions: sum w*d^2/|d-v|^2 = 1.
        weighted_bands.append(sn * v.imag / np.pi)
        support.append(t[[0, -1]].tolist())
    mass = np.sum([trapezoid(arr[1:], arr[0], axis=1) for arr in arrays], axis=0)
    first = np.sum(
        [trapezoid(arr[1:] * arr[0], arr[0], axis=1) for arr in arrays], axis=0
    )
    transform_error = 0.0
    for u in [-sn, -0.3, -3.0]:
        v = negative_inverse(u)
        transform = v / (u * (dj - v))
        integral = -atom / u + sum(
            trapezoid(arr[1:] / (arr[0] - u), arr[0], axis=1) for arr in arrays
        )
        transform_error = max(transform_error, np.max(abs(integral / transform - 1)))
    report = dict(
        law=law,
        n=n,
        ranks=ranks.tolist(),
        atom=atom.tolist(),
        positive_mass=mass.tolist(),
        first_moment=first.tolist(),
        target_first_moment=(sn * dj).tolist(),
        support=support,
        mass_error=float(max(abs(atom + mass - 1))),
        first_moment_rel_error=float(max(abs(first / (sn * dj) - 1))),
        negative_transform_rel_error=float(transform_error),
        boundary_residual=float(boundary_error),
        cutoff=cutoff,
    )
    # Independent trace-square target for the weighted mass. Integrate the
    # very small residual tail in log coordinates past the explicit sum.
    end = 100000
    j = np.arange(1, end + 1, dtype=float)
    log_end = np.log(end + 0.5)
    rate = 1.4 if law == "power" else 1.0
    tail = quad(
        lambda x: np.exp(-rate * x) * (1 if law == "power" else (1 + x) ** -2.4),
        log_end,
        np.inf,
        epsabs=1e-18,
    )[0]
    target = (np.sum(sigma(j, law) ** 2) + tail) / n
    weighted_mass = sum(
        trapezoid(rho, arr[0]) for rho, arr in zip(weighted_bands, arrays)
    )
    report.update(
        weighted_mass=float(weighted_mass),
        weighted_mass_target=float(target),
        weighted_zero_atom=0.0,
        weighted_mass_relative_error=float(abs(weighted_mass / target - 1)),
    )
    assert report["weighted_mass_relative_error"] < 1e-4
    return arrays, weighted_bands, report


def main():
    curves, report = {}, {}
    for law, n in PANELS:
        key = f"{law}_{n}"
        coarse, _, r0 = solve(law, n, 1)
        bands, weighted, r = solve(law, n, 2)
        assert len(coarse) == len(bands)
        r["refined_atom_change"] = float(np.max(abs(np.array(r["atom"]) - r0["atom"])))
        r["refined_edge_change"] = float(
            np.max(abs(np.array(r["support"]) - r0["support"]))
        )
        r["refined_mass_change"] = float(
            np.max(abs(np.array(r["positive_mass"]) - r0["positive_mass"]))
        )
        r["refined_relative_peak_changes"] = (
            abs(
                np.max(np.hstack(bands)[1:], axis=1)
                / np.max(np.hstack(coarse)[1:], axis=1)
                - 1
            )
        ).tolist()
        curves[key + "_atom"] = r["atom"]
        curves[key + "_rank"] = r["ranks"]
        for i, band in enumerate(bands):
            curves[f"{key}_band{i}"] = band
            curves[f"{key}_weighted{i}"] = weighted[i]
        r["weighted_mass_refinement_change"] = abs(
            r["weighted_mass"] - r0["weighted_mass"]
        )
        report[key] = r
        print(key, json.dumps(r), flush=True)
        assert r["mass_error"] < 1e-4 and r["first_moment_rel_error"] < 1e-4
        assert r["negative_transform_rel_error"] < 1e-4
    np.savez_compressed(HERE / "curves.npz", **curves)
    (HERE / "validation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

__all__ = ["HERE", "PANELS", "np", "json"]
