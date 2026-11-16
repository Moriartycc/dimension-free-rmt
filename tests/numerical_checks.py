"""Small mathematical checks; never write into a publication data directory."""

from pathlib import Path
import sys, tempfile, subprocess, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from covariance import finite_population, spec
from ridge_prediction import edge_and_root, predict
from spectral_solver import Population, FlowRisk
import spectral_solver as solver
import risks
import torch


def main():
    torch.set_num_threads(2)
    for law in ("power", "critical"):
        p = 0.2
        d = 65536
        n = 32
        s, w, tr = finite_population(law, p, d=d)
        direct = spec(law, p, np.arange(1, d + 1, dtype=float))
        assert abs(tr / direct.sum() - 1) < 1e-10
        for k in (2, 3):
            assert abs(np.dot(w, s**k) / np.sum(direct**k) - 1) < 1e-10
        edge, ke, kap = edge_and_root(s, w, tr, n)
        j = np.arange(128)
        theta = np.exp(j * np.log(16.0) - solver.gammaln(j + 1))
        theta /= np.linalg.norm(theta)
        ss = spec(law, p, j.astype(float) + 1)
        vb, _ = predict(s, w, tr, n, 0, ss, theta[:, None], edge, ke)
        J = np.dot(w, (s / (s + kap)) ** 2) / n
        expected = np.array(
            [J / (1 - J), np.sum(ss**2 * theta**2 * (kap / (kap + ss)) ** 2) / (1 - J)]
        )
        np.testing.assert_allclose(
            vb[:, :, 0], np.broadcast_to(expected, (3, 2)), rtol=1e-10
        )
    # Boundary measures and the independent ridgeless formula exercise the shared GF engine.
    solver.D = 8192
    risks.Q = solver.Q
    pop = Population("critical", 0.2, 8)
    data, checks = pop.mesh()
    assert checks["gram_mass_error"] < 5e-4 and checks["mass_error"] < 5e-4
    flow = FlowRisk(data, pop)
    initial = flow.gf(0)
    np.testing.assert_allclose(initial[0], 0, atol=0)
    np.testing.assert_allclose(
        initial[1],
        np.sum(pop.source_s[:, None] ** 2 * pop.theta**2, axis=0),
        rtol=1e-12,
    )
    inf = flow.filter(np.zeros_like(data["x"]), np.zeros_like(data["x"]))
    vb, _ = predict(
        pop.sigma, pop.weights, pop.trace, pop.n, 0, pop.source_s, pop.theta
    )
    np.testing.assert_allclose(inf[0], 1e-4 * vb[0, 0], rtol=2e-3)
    np.testing.assert_allclose(inf[1], vb[0, 1], rtol=2e-3)
    # Explicit coordinate worker, including a final partial block and two independent seeds.
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments/numerical-validation/spectral-laws/worker.py"),
                "--device",
                "cpu",
                "--dimension",
                "257",
                "--sizes",
                "2",
                "8",
                "--stop",
                "2",
                "--output",
                folder,
            ],
            check=True,
        )
        a = np.load(Path(folder) / "r0000.npz")
        b = np.load(Path(folder) / "r0001.npz")
        assert int(a["d"]) == 257 and int(b["seed"]) - int(a["seed"]) == 104729
        assert not np.array_equal(a["power_0.2_n8"], b["power_0.2_n8"])
        a.close()
        b.close()
    mapping = json.loads((ROOT / "figures.json").read_text())
    for entry in mapping.values():
        assert (ROOT / entry["output"]).is_file()
        assert (ROOT / entry["configuration"]).is_file()
    print(
        "Passed: finite moments, ridgeless identities, boundary measures, GF endpoints, independent sampling, figure mapping."
    )


if __name__ == "__main__":
    main()
