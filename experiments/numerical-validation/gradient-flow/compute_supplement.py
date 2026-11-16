from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))
"""Theory-only GF extension: reuse spectral quadrature for new source weights."""
from pathlib import Path
import sys, argparse

HERE = Path(__file__).resolve().parent
import spectral_solver as core
from spectral_solver import np, torch, json, time, Population, FlowRisk, brentq
import risks

core.Q = np.array([28.0, 32.0, 36.0, 40.0])
risks.Q = core.Q
risks.TAU2 = core.TAU**2
NS = np.array([64, 96, 128, 192, 256, 320, 384, 448, 512])
S = np.linspace(0, 2, 91) ** 2
SEARCH = np.geomspace(1e-10, 1e4, 161)
OLD = HERE / "spectral_cache"


def make_risk(pop, data):
    rr = FlowRisk(data, pop)
    assert np.max(np.sum(pop.theta[256:] ** 2, axis=0)) < 1e-150
    rr.dm = rr.dm[:, :256].contiguous()
    rr.atom = rr.atom[:256]
    rr.weights = rr.weights[:256]
    return rr


def calculate(pop, data):
    rr = make_risk(pop, data)
    factor = pop.n / pop.tail
    infinity = rr.filter(np.zeros_like(data["x"]), np.zeros_like(data["x"]))
    values = np.array([rr.gf(float(s * factor)) for s in S])
    search = []
    for s in SEARCH:
        if s * factor * data["x"][0] > 740:
            search.append(np.vstack([infinity, np.zeros(len(core.Q))]))
        else:
            search.append(rr.gf(float(s * factor)))
    search = np.array(search)
    opts = []
    optr = []
    rootsall = []
    for qi in range(len(core.Q)):
        roots = []
        for i in np.flatnonzero(
            np.sign(search[:-1, 3, qi]) * np.sign(search[1:, 3, qi]) < 0
        ):
            root = np.exp(
                brentq(
                    lambda z: rr.gf(float(np.exp(z) * factor))[3, qi],
                    np.log(SEARCH[i]),
                    np.log(SEARCH[i + 1]),
                    xtol=1e-9,
                )
            )
            roots.append(float(root))
        candidates = [0.0, *roots, np.inf]
        scores = [
            values[0, 2, qi],
            *[rr.gf(float(s * factor))[2, qi] for s in roots],
            infinity[2, qi],
        ]
        win = int(np.argmin(scores))
        opts.append(candidates[win])
        optr.append(scores[win])
        rootsall.append(roots)
    return (
        rr,
        dict(
            s=S,
            n=pop.n,
            q=core.Q,
            ratio=values[:, 2] / infinity[2],
            values=values,
            infinity=infinity,
            initial=values[0, :3],
            optimum_s=opts,
            optimum_ratio=np.array(optr) / infinity[2],
            tail=pop.tail,
            d=core.D,
            noise_sd=core.TAU,
        ),
        rootsall,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=NS.tolist())
    args = parser.parse_args()
    (HERE / "supplement").mkdir(exist_ok=True)
    start = time.time()
    for law, p in core.CASES:
        for n in args.n:
            name = f"{law}_{p:g}_n{n}"
            dest = HERE / "supplement" / f"{name}.npz"
            tic = time.time()
            pop = Population(law, p, n)
            old = OLD / f"{name}.npz"
            if old.exists():
                data = dict(np.load(old))
                reused = True
            else:
                data, mesh_check = pop.mesh()
                reused = False
            rr, out, roots = calculate(pop, data)
            _, _, kap = core.edge_and_root(pop.sigma, pop.weights, pop.trace, n)
            J = np.sum(pop.weights * (pop.sigma / (pop.sigma + kap)) ** 2) / n
            v = core.TAU**2 * J / (1 - J)
            b = np.sum(
                pop.source_s[:, None] ** 2
                * pop.theta**2
                * (kap / (kap + pop.source_s[:, None])) ** 2,
                axis=0,
            ) / (1 - J)
            target = np.stack([np.full(len(core.Q), v), b, b + v])
            err = float(np.max(abs(out["infinity"] / target - 1)))
            assert err < 2e-3, (name, err)
            assert np.isfinite(out["ratio"]).all() and np.min(out["ratio"]) > 0
            # Verify the new source weighting on a doubled mesh at the largest n.
            refinement = None
            if n == 512:
                finer, _ = pop.mesh(2)
                rf = make_risk(pop, finer)
                times = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0]) * n / pop.tail
                coarse = np.array([rr.gf(float(t))[:3] for t in times])
                fine = np.array([rf.gf(float(t))[:3] for t in times])
                refinement = float(np.max(abs(coarse / fine - 1)))
                assert refinement < 2e-3, (name, refinement)
                del rf, finer
            report = dict(
                theory_only=True,
                empirical_trials=0,
                d=core.D,
                q=core.Q.tolist(),
                noise_sd=core.TAU,
                reused_spectral_mesh=reused,
                ridgeless_relative_error=err,
                mesh_refinement_relative_error=refinement,
                stationary_s=roots,
                optimum_s=np.asarray(out["optimum_s"]).tolist(),
                optimum_ratio=np.asarray(out["optimum_ratio"]).tolist(),
                seconds=time.time() - tic,
            )
            np.savez_compressed(dest, **out)
            dest.with_suffix(".json").write_text(json.dumps(report, indent=2))
            print(
                name,
                "s*",
                out["optimum_s"],
                "ratio",
                out["optimum_ratio"],
                "seconds",
                round(time.time() - tic, 1),
                flush=True,
            )
            del rr, data, pop
            torch.cuda.empty_cache()
    (HERE / "status.json").write_text(
        json.dumps(
            dict(status="complete", theory_only=True, seconds=time.time() - start),
            indent=2,
        )
    )
    print("COMPLETE", round(time.time() - start, 1), flush=True)


if __name__ == "__main__":
    main()
