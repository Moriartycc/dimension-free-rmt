from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))

"""Deterministic ridge predictions, including population-refinement checks."""
from common import np, CASES
from covariance import finite_population
from ridge_prediction import edge_and_root, predict
from spectral_solver import sources, TAU, Q, D

HERE = Path(__file__).resolve().parent
NS = np.arange(16, 513, 4)
CS = np.array([-0.6, 0, 0.5, 1, 1.5, 2])
KS = (1, 2, 4)


def main():
    risks = {}
    edges = {}
    for law, p in CASES:
        pop = finite_population(law, p)
        ss, theta = sources(law, p)
        rows = []
        es = []
        for n in NS:
            edge, ke, _ = edge_and_root(*pop, n)
            es.append(edge)
            rows.append([predict(*pop, n, c, ss, theta, edge, ke)[0] for c in CS])
        name = f"{law}_{p:g}"
        risks[name] = np.array(rows)
        edges[name] = np.array(es)
        fine = finite_population(law, p, head=16384, step=0.375, order=20)
        for n in (64, 256, 512):
            for ci in (0, 1, 5):
                check, _ = predict(*fine, n, CS[ci], ss, theta)
                assert (
                    np.max(abs(risks[name][np.flatnonzero(NS == n)[0], ci] / check - 1))
                    < 1e-7
                )
        print(name, flush=True)
    np.savez_compressed(
        HERE / "data" / "theory.npz", **risks, n=NS, c=CS, k=KS, q=Q, tau=TAU, d=D
    )
    np.savez_compressed(HERE / "edges.npz", **edges, n=NS)


if __name__ == "__main__":
    main()
