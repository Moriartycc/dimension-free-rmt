"""Consolidate all production risks; validate seeds, identities, and edge events."""

from pathlib import Path
import json, hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
CASES = (("power", 0.2), ("critical", 0.2), ("power", 0.05), ("critical", 0.05))
NS = (16, 32, 64, 128, 192, 256, 384, 512)
CS = (-0.6, 0.0, 0.5, 1.0, 1.5, 2.0)
KS = (1, 2, 4)
Q = (16.0, 24.0)
COUNT = 400


def main():
    arrays = {
        f"{law}_{p:g}": np.empty((COUNT, len(NS), len(CS), len(KS), 3, len(Q)))
        for law, p in CASES
    }
    minima = {name: np.empty((COUNT, len(NS))) for name in arrays}
    seeds = []
    audit_max = 0.0
    edges = np.load(HERE / "edges.npz")
    for rep in range(COUNT):
        with np.load(HERE / "samples" / f"r{rep:03d}.npz") as data:
            seed = int(data["seed"])
            assert seed == 202611160050000 + 104729 * rep
            seeds.append(seed)
            for name, arr in arrays.items():
                for ni, n in enumerate(NS):
                    key = f"{name}_n{n}"
                    v = data[key]
                    assert (
                        v.shape == (6, 3, 3, 2)
                        and np.isfinite(v).all()
                        and np.min(v) > -1e-10
                    )
                    assert np.allclose(
                        v[:, :, 2], v[:, :, 0] + v[:, :, 1], rtol=1e-12, atol=1e-15
                    )
                    assert np.allclose(v[1], v[1, 0], rtol=1e-12, atol=1e-15)
                    assert np.allclose(
                        v[:, :, 0, 0], v[:, :, 0, 1], rtol=1e-12, atol=1e-15
                    )
                    e = data[key + "_eig"]
                    assert (
                        e.shape == (n,)
                        and np.isfinite(e).all()
                        and (e > 0).all()
                        and (np.diff(e) >= 0).all()
                    )
                    edge = float(edges[name][np.flatnonzero(edges["n"] == n)[0]])
                    minima[name][rep, ni] = e[0] / edge
                    arr[rep, ni] = v
                    if rep == 0:
                        L = data[key + "_L"]
                        y = data[key + "_y"]
                        r = data[key + "_r"]
                        signal = data[key + "_signal"]
                        for cj, c in enumerate(CS):
                            for kj, k in enumerate(KS):
                                lam = c * edge
                                phi = (1 - (lam / (e + lam)) ** k) / e
                                coef = phi[:, None] * y
                                variance = 1e-4 / n * np.dot(phi**2, np.diag(L))
                                bias = (
                                    signal
                                    - 2 * np.sum(r * coef, axis=0)
                                    + np.sum(coef * (L @ coef), axis=0)
                                )
                                expected = np.stack(
                                    [np.full(2, variance), bias, bias + variance]
                                )
                                err = float(
                                    np.max(
                                        abs(expected - v[cj, kj])
                                        / np.maximum(abs(v[cj, kj]), 1e-14)
                                    )
                                )
                                audit_max = max(audit_max, err)
                        assert audit_max < 1e-7
        if rep % 50 == 0:
            print("Validated", rep + 1, "/", COUNT, flush=True)
    assert len(set(seeds)) == COUNT
    out = HERE / "ridge_aggregated.npz"
    np.savez_compressed(
        out,
        **arrays,
        **{k + "_min_to_edge": v for k, v in minima.items()},
        seeds=seeds,
        n=NS,
        c=CS,
        k=KS,
        q=Q,
        noise_sd=0.01,
        d=2**24,
    )
    configs = [
        json.loads(p.read_text()) for p in sorted((HERE / "runs").glob("*/config.json"))
    ]
    for cfg in configs:
        assert cfg["d"] == 2**24 and cfg["noise_sd"] == 0.01 and cfg["n"] == list(NS)
        assert cfg["q"] == list(Q) and cfg["c"] == list(CS) and cfg["k"] == list(KS)
    audits = [
        json.loads(p.read_text())
        for p in sorted((HERE / "runs").glob("*/numerical_audit.json"))
    ]
    report = dict(
        trials=COUNT,
        d=2**24,
        n=NS,
        c=CS,
        k=KS,
        q=Q,
        noise_sd=0.01,
        axes=["trial", "n", "c", "k", "metric:variance,bias,total", "q"],
        unique_expected_seeds=True,
        config_files_validated=len(configs),
        rep0_recomputed_risk_max_relative_difference=audit_max,
        original_numerical_audits=audits,
        all_raw_risks_retained=True,
        minimum_sample_to_edge={k: float(v.min()) for k, v in minima.items()},
        cutoff_failures={k: int((v < 0.8).sum()) for k, v in minima.items()},
        cutoff_checks=COUNT * len(NS) * len(CASES),
        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
    )
    (HERE / "aggregation_checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
