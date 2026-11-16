from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))
"""Dense-time theoretical GF surfaces for the unchanged finite-d setup."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
import spectral_solver as core
from spectral_solver import np, torch, json, time, Population, FlowRisk
import risks

core.Q = np.array([16.0, 24.0])
risks.Q = core.Q
risks.TAU2 = core.TAU**2
NS = np.array([64, 96, 128, 192, 256, 320, 384, 448, 512])
S = np.linspace(0, 2, 91) ** 2
OLD = HERE / "spectral_cache"


def main():
    (HERE / "theory").mkdir(exist_ok=True)
    ridge = np.load(HERE.parent / "ridge-filters" / "data" / "theory.npz")
    for law, p in core.CASES:
        for n in NS:
            name = f"{law}_{p:g}_n{n}"
            dest = HERE / "theory" / f"{name}.npz"
            tic = time.perf_counter()
            pop = Population(law, p, int(n))
            old = OLD / f"{name}.npz"
            if old.exists():
                data = dict(np.load(old))
                diag = json.loads(old.with_suffix(".json").read_text())
                out = dict(np.load(OLD / f"{name}_risk.npz"))
                rr = FlowRisk(data, pop)
            else:
                data, diag = pop.mesh()
                assert (
                    max(
                        diag[k]
                        for k in (
                            "mass_error",
                            "first_moment_error",
                            "weighted_mass_error",
                            "gram_mass_error",
                        )
                    )
                    < 5e-4
                )
                rr, out, _ = core.flow(pop, data)
            values = np.array([rr.gf(float(s * n / pop.tail)) for s in S])
            ix = np.flatnonzero(ridge["n"] == n)[0]
            analytic = ridge[f"{law}_{p:g}"][ix, 1, 0].copy()
            analytic[0] *= core.TAU**2
            target = np.vstack([analytic, analytic.sum(0)])
            error = float(np.max(abs(out["infinity"] / target - 1)))
            assert error < 2e-3, (name, error)
            ratio = values[:, 2] / out["infinity"][2]
            assert np.isfinite(ratio).all() and (ratio > 0).all()
            # Check newly computed surfaces against a finer spectral mesh at
            # representative early, valley, and late times.
            refined_error = None
            if not old.exists() and n in (192, 384):
                fine, _ = pop.mesh(2)
                rf = FlowRisk(fine, pop)
                probes = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0]) * n / pop.tail
                a = np.array([rr.gf(t)[:3] for t in probes])
                b = np.array([rf.gf(t)[:3] for t in probes])
                refined_error = float(np.max(abs(a / b - 1)))
                assert refined_error < 2e-3
                del rf, fine
            np.savez_compressed(
                dest,
                s=S,
                n=n,
                q=core.Q,
                ratio=ratio,
                values=values,
                infinity=out["infinity"],
                initial=out["initial"],
                optimum_s=out["opt_times"] * pop.tail / n,
                optimum_ratio=out["opt_risk"] / out["infinity"][2],
                a=pop.scale["a"],
                tail=pop.tail,
            )
            (HERE / "theory" / f"{name}.json").write_text(
                json.dumps(
                    dict(
                        seconds=time.perf_counter() - tic,
                        ridgeless_check=error,
                        mesh_refinement=refined_error,
                        used_existing_spectral_mesh=old.exists(),
                    ),
                    indent=2,
                )
            )
            print(
                name,
                round(time.perf_counter() - tic, 1),
                "opt",
                out["opt_times"] * pop.tail / n,
                flush=True,
            )
            del rr, data
            torch.cuda.empty_cache()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
