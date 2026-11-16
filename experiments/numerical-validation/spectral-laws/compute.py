from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))

"""Compute finite-d predictions and scales for Figure 6."""
from spectral_solver import Population, CASES, np, json

HERE = Path(__file__).resolve().parent


def main():
    scales = {}
    out = HERE / "data" / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    for law, p in CASES:
        by_n = {}
        for n in (2, 8, 32, 128):
            pop = Population(law, p, n)
            data, checks = pop.mesh()
            assert (
                max(
                    checks[k]
                    for k in ("mass_error", "gram_mass_error", "weighted_mass_error")
                )
                < 5e-4
            )
            np.savez_compressed(
                out / f"{law}_{p:g}_n{n}.npz",
                **{k: data[k] for k in ("x", "xr", "rho", "cdf", "support")},
            )
            by_n[str(n)] = dict(pop.scale)
        scales[f"{law}_{p:g}"] = {"by_n": by_n}
    (HERE / "data" / "scales.json").write_text(json.dumps(scales, indent=2))


if __name__ == "__main__":
    main()
