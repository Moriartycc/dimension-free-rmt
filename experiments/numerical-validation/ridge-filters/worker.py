"""Exact d=2^24 paired-design ridge production simulation."""

from pathlib import Path
import argparse, os, time, json

HERE = Path(__file__).resolve().parent
import numpy as np
import torch
import scipy.linalg as la
from scipy.special import gammaln

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
META = (
    HERE
    / "runs"
    / (
        os.environ.get("SLURM_ARRAY_JOB_ID", "local")
        + "_"
        + os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    )
)
META.mkdir(parents=True, exist_ok=True)
D = 2**24
CHUNK = 2**16
NS = (16, 32, 64, 128, 192, 256, 384, 512)
CASES = (("power", 0.2), ("critical", 0.2), ("power", 0.05), ("critical", 0.05))
CS = (-0.6, 0.0, 0.5, 1.0, 1.5, 2.0)
KS = (1, 2, 4)
Q = (16.0, 24.0)
TAU = 0.01
SEED = 202611160050000


def write_json(name, value):
    for attempt in range(30):
        try:
            (META / name).write_text(json.dumps(value, indent=2))
            return
        except PermissionError:
            time.sleep(0.1)
    raise RuntimeError("Could not save " + name)


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    (HERE / "samples").mkdir(exist_ok=True)
    j = torch.arange(1, D + 1, dtype=torch.float64, device="cuda")
    ss = [
        j.pow(-1 - p) if law == "power" else 1 / (j * (1 + j.log()).pow(1 + p))
        for law, p in CASES
    ]
    del j
    idx = np.arange(2048)[:, None]
    logs = idx * np.log(np.array(Q))[None, :] - gammaln(idx + 1)
    theta = np.exp(logs - logs.max(axis=0))
    theta /= np.linalg.norm(theta, axis=0)
    assert np.sum(theta[256:] ** 2) < 1e-300
    theta = torch.tensor(theta, device="cuda", dtype=torch.float64)
    signal = [
        (s[:2048, None].square() * theta.square()).sum(0).cpu().numpy() for s in ss
    ]
    edges = np.load(HERE / "edges.npz")
    config = dict(
        d=D,
        n=NS,
        q=Q,
        c=CS,
        k=KS,
        noise_sd=TAU,
        trials=a.trials,
        cases=CASES,
        cutoff_fraction=0.8,
        precision="float64",
        chunk=CHUNK,
        source="L2-normalized Poisson PMF coefficients; beta=sqrt(Sigma)theta",
        design="All d coordinates explicitly sampled; laws and n paired within a trial",
        risk="Conditional-on-design prediction risk, analytically averaged over response noise",
        evaluation="All raw risks retained; common lower-edge event recorded, no samples dropped",
        gpu=torch.cuda.get_device_name(),
        seed_rule=f"{SEED}+104729*rep",
    )
    write_json("config.json", config)
    timings = []
    audits = []
    cutoffs = []
    for rep in range(a.start, a.start + a.trials):
        if (HERE / "samples" / f"r{rep:03d}.npz").exists():
            continue
        started = time.perf_counter()
        gen = torch.Generator(device="cuda").manual_seed(SEED + rep * 104729)
        gram = torch.zeros((4, 512, 512), device="cuda", dtype=torch.float64)
        gram2 = torch.zeros_like(gram)
        yy = torch.zeros((4, 512, 2), device="cuda", dtype=torch.float64)
        rr = torch.zeros_like(yy)
        cpu_error = []
        for offset in range(0, D, CHUNK):
            z = torch.randn(
                (512, CHUNK), generator=gen, device="cuda", dtype=torch.float64
            )
            for ci, s in enumerate(ss):
                sp = s[offset : offset + CHUNK]
                weighted = z * sp
                g = weighted @ z.T
                h = weighted @ weighted.T
                gram[ci] += g
                gram2[ci] += h
                if offset == 0:
                    yy[ci] = z[:, :2048] @ (s[:2048, None] * theta)
                    rr[ci] = z[:, :2048] @ (s[:2048, None].square() * theta)
                    if rep == 0:
                        zz = z[:2].cpu().numpy()
                        s2 = sp.cpu().numpy()
                        for actual, expected in (
                            (g[:2, :2], (zz * s2) @ zz.T),
                            (h[:2, :2], (zz * s2**2) @ zz.T),
                        ):
                            cpu_error.append(
                                float(
                                    np.linalg.norm(actual.cpu().numpy() - expected)
                                    / np.linalg.norm(expected)
                                )
                            )
            if offset % (32 * CHUNK) == 0:
                sync()
                write_json(
                    "status.json",
                    dict(
                        phase="sampling",
                        trial=rep + 1,
                        total=a.trials,
                        coordinates_done=offset + CHUNK,
                        dimension=D,
                        elapsed_seconds=time.perf_counter() - started,
                    ),
                )
            del z, weighted, g, h
        sync()
        sampling = time.perf_counter() - started
        gg, hh, y, r = [t.cpu().numpy() for t in (gram, gram2, yy, rr)]
        processing_start = time.perf_counter()
        risks = {}
        saved = {}
        direct_checks = []
        for ci, (law, p) in enumerate(CASES):
            for n in NS:
                key = f"{law}_{p:g}_n{n}"
                e, u = la.eigh(
                    (gg[ci, :n, :n] + gg[ci, :n, :n].T) / (2 * n), check_finite=False
                )
                L = u.T @ ((hh[ci, :n, :n] + hh[ci, :n, :n].T) / (2 * n)) @ u
                yrot = u.T @ y[ci, :n] / np.sqrt(n)
                rrot = u.T @ r[ci, :n] / np.sqrt(n)
                edge = float(edges[f"{law}_{p:g}"][np.flatnonzero(edges["n"] == n)[0]])
                values = np.empty((len(CS), len(KS), 3, 2))
                for cj, c in enumerate(CS):
                    lam = c * edge
                    for kj, k in enumerate(KS):
                        phi = (1 - (lam / (e + lam)) ** k) / e
                        v = phi[:, None] * yrot
                        variance = TAU**2 / n * np.dot(phi**2, np.diag(L))
                        bias = (
                            signal[ci]
                            - 2 * np.sum(rrot * v, axis=0)
                            + np.sum(v * (L @ v), axis=0)
                        )
                        assert bias.min() > -1e-10 and np.isfinite(bias).all()
                        values[cj, kj] = np.stack(
                            [np.full(2, variance), bias, bias + variance]
                        )
                        if rep == 0 and n == 512 and c == -0.6 and k == 2:
                            # Direct full-coordinate residual audit, replayed below.
                            direct_checks.append(
                                (
                                    ci,
                                    torch.tensor(u @ v / np.sqrt(n), device="cuda"),
                                    bias,
                                )
                            )
                risks[key] = values
                saved.update(
                    {
                        key + "_eig": e,
                        key + "_L": L,
                        key + "_y": yrot,
                        key + "_r": rrot,
                        key + "_signal": signal[ci],
                    }
                )
                cutoffs.append(
                    dict(
                        trial=rep,
                        case=key,
                        min_to_edge=float(e[0] / edge),
                        pass_cutoff=bool(e[0] >= 0.8 * edge),
                        distance_to_negative_pole=float(e[0] / edge - 0.6),
                    )
                )
        processing = time.perf_counter() - processing_start
        # Persist sufficient statistics for reusing filters for the two saved sources.
        temporary = HERE / "samples" / f"r{rep:03d}.tmp.npz"
        np.savez_compressed(temporary, **risks, **saved, seed=SEED + rep * 104729)
        temporary.replace(HERE / "samples" / f"r{rep:03d}.npz")
        standard_elapsed = time.perf_counter() - started
        if rep == 0:
            assert max(cpu_error) < 1e-11
            gen = torch.Generator(device="cuda").manual_seed(SEED)
            direct = torch.zeros((4, 2), dtype=torch.float64, device="cuda")
            for offset in range(0, D, CHUNK):
                z = torch.randn(
                    (512, CHUNK), generator=gen, dtype=torch.float64, device="cuda"
                )
                for ci, coef, expected in direct_checks:
                    residual = -(z.T @ coef)
                    if offset == 0:
                        residual[:2048] += theta
                    direct[ci] += (
                        (ss[ci][offset : offset + CHUNK, None] * residual)
                        .square()
                        .sum(0)
                    )
            direct = direct.cpu().numpy()
            for ci, coef, expected in direct_checks:
                err = float(np.max(np.abs(direct[ci] / expected - 1)))
                assert err < 1e-6, (ci, err)
                audits.append(dict(case=CASES[ci], direct_bias_relative_error=err))
            write_json(
                "numerical_audit.json",
                dict(cpu_gram_relative_error=max(cpu_error), direct_bias=audits),
            )
        timings.append(
            dict(
                trial=rep,
                sampling_seconds=sampling,
                risk_seconds=processing,
                production_equivalent_seconds=standard_elapsed,
                total_with_audit_seconds=time.perf_counter() - started,
            )
        )
        write_json("timings.json", timings)
        write_json("cutoff_checks.json", cutoffs)
        print("TRIAL", rep, timings[-1], flush=True)
    if not timings:
        write_json(
            "status.json",
            dict(
                phase="complete",
                start=a.start,
                trials=a.trials,
                resumed_all_complete=True,
            ),
        )
        return
    mean = float(np.mean([r["production_equivalent_seconds"] for r in timings]))
    write_json(
        "status.json",
        dict(
            phase="complete",
            trials=a.trials,
            mean_seconds=mean,
            projected_400_hours=400 * mean / 3600,
            minimum_sample_to_edge=min(c["min_to_edge"] for c in cutoffs),
            cutoff_failures=sum(not c["pass_cutoff"] for c in cutoffs),
            cutoff_checks=len(cutoffs),
        ),
    )
    print("COMPLETE", mean, 400 * mean / 3600, flush=True)


if __name__ == "__main__":
    main()
