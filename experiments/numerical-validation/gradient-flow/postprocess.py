"""Evaluate gradient flow on saved ridge designs, without resampling X."""

from pathlib import Path
import os, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "ridge-filters" / "samples"
CASES = ("power_0.2", "critical_0.2", "power_0.05", "critical_0.05")
NS = (64, 128, 192, 256, 384, 512)
S = np.linspace(0, 2, 91) ** 2
CONFIG = json.loads((HERE / "scales.json").read_text())


def one_trial(rep):
    result = {}
    maxerr = 0.0
    with np.load(SOURCE / f"r{rep:03d}.npz") as data:
        seed = int(data["seed"])
        assert seed == 202611160050000 + 104729 * rep
        for name in CASES:
            values = []
            baseline = []
            for n in NS:
                key = f"{name}_n{n}"
                e = data[key + "_eig"]
                L = data[key + "_L"]
                y = data[key + "_y"]
                r = data[key + "_r"]
                signal = data[key + "_signal"]
                t = S * n / CONFIG[key]
                phi = -np.expm1(-t[:, None] * e[None, :]) / e[None, :]

                def evaluate(f):
                    coef = f[:, :, None] * y[None, :, :]
                    # Matrix columns index time and source; one BLAS multiplication.
                    flat = coef.transpose(1, 0, 2).reshape(n, -1)
                    quadratic = np.sum(flat * (L @ flat), axis=0).reshape(len(f), 2)
                    bias = (
                        signal[None, :]
                        - 2 * np.einsum("neq,eq->nq", coef, r)
                        + quadratic
                    )
                    variance = 1e-4 / n * (f * f) @ np.diag(L)
                    return np.stack(
                        [
                            np.broadcast_to(variance[:, None], bias.shape),
                            bias,
                            bias + variance[:, None],
                        ],
                        axis=1,
                    )

                out = evaluate(phi)
                inf = evaluate((1 / e)[None, :])[0]
                reference = data[key][1, 0]
                err = np.max(abs(inf - reference) / np.maximum(abs(reference), 1e-14))
                maxerr = max(maxerr, float(err))
                assert err < 1e-7, (key, err)
                assert np.isfinite(out).all() and out.min() > -1e-10
                assert np.array_equal(out[0, 0], np.zeros(2))
                assert np.allclose(out[0, 1], signal, rtol=1e-13)
                assert np.all(np.diff(out[:, 0, 0]) >= -1e-14)
                values.append(out)
                baseline.append(inf)
            result[name] = np.array(values)
            result[name + "_infinity"] = np.array(baseline)
    return rep, seed, result, maxerr


def main():
    start = time.time()
    count = 400
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    arrays = {name: np.empty((count, len(NS), len(S), 3, 2)) for name in CASES}
    arrays.update(
        {name + "_infinity": np.empty((count, len(NS), 3, 2)) for name in CASES}
    )
    seeds = np.zeros(count, dtype=np.int64)
    maxerr = 0.0
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_trial, rep) for rep in range(count)]
        for future in as_completed(futures):
            rep, seed, result, error = future.result()
            seeds[rep] = seed
            for name, values in result.items():
                arrays[name][rep] = values
            done += 1
            maxerr = max(maxerr, error)
            if done % 10 == 0 or done == 1:
                status = dict(
                    done=done,
                    total=count,
                    elapsed_seconds=time.time() - start,
                    estimated_remaining_seconds=(time.time() - start)
                    / done
                    * (count - done),
                )
                (HERE / "status.json").write_text(json.dumps(status, indent=2))
                print(status, flush=True)
    assert len(set(seeds)) == count
    np.savez_compressed(
        HERE / "gf_aggregated.npz",
        **arrays,
        seeds=seeds,
        n=NS,
        s=S,
        q=(16, 24),
        d=2**24,
        noise_sd=0.01,
    )
    audit = dict(
        trials=count,
        n=NS,
        q=[16, 24],
        d=2**24,
        noise_sd=0.01,
        elapsed_seconds=time.time() - start,
        maximum_ridgeless_relative_error=maxerr,
        initial_and_monotone_variance_checks=True,
        unique_expected_seeds=True,
        source="Saved full-coordinate ridge sufficient statistics",
        axes=["trial", "n", "normalized_time", "metric:variance,bias,risk", "q"],
    )
    (HERE / "checks.json").write_text(json.dumps(audit, indent=2))
    print("COMPLETE", audit, flush=True)


if __name__ == "__main__":
    main()
