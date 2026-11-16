"""Exact, chunked finite-population Gaussian Gram sampler."""

import os, time, argparse
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
D = 2**24
NS = (32, 128)
CASES = (("power", 0.2), ("critical", 0.2), ("power", 0.05), ("critical", 0.05))
CHUNK = 2**16
SEED = 202611160001000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=2000)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(NS))
    ap.add_argument("--dimension", type=int, default=D)
    ap.add_argument("--output", type=Path, default=HERE / "samples")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    sizes = tuple(a.sizes)
    dimension = a.dimension
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = a.device

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def spectrum(offset):
        j = torch.arange(
            offset + 1,
            min(offset + CHUNK, dimension) + 1,
            dtype=torch.float64,
            device=device,
        )
        return [
            j.pow(-1 - p) if law == "power" else 1 / (j * (1 + j.log()).pow(1 + p))
            for law, p in CASES
        ]

    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    for rep in range(a.start, a.stop):
        path = out / f"r{rep:04d}.npz"
        if path.exists():
            continue
        tick = time.perf_counter()
        gen = torch.Generator(device=device).manual_seed(a.seed + rep * 104729)
        grams = torch.zeros(
            (4, max(sizes), max(sizes)), dtype=torch.float64, device=device
        )
        for offset in range(0, dimension, CHUNK):
            z = torch.randn(
                (max(sizes), min(CHUNK, dimension - offset)),
                generator=gen,
                dtype=torch.float64,
                device=device,
            )
            for i, s in enumerate(spectrum(offset)):
                grams[i].add_((z * s) @ z.T)
            del z
        eig = {}
        for n in sizes:
            ev = (
                torch.linalg.eigvalsh(
                    (grams[:, :n, :n] + grams[:, :n, :n].transpose(-1, -2)) / 2
                )
                .cpu()
                .numpy()
            )
            assert np.isfinite(ev).all() and (ev > 0).all()
            for i, (law, p) in enumerate(CASES):
                eig[f"{law}_{p:g}_n{n}"] = ev[i]
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            **eig,
            seed=a.seed + rep * 104729,
            d=dimension,
            device=device,
            seconds=time.perf_counter() - tick,
        )
        temporary.replace(path)
        print(rep, time.perf_counter() - tick, flush=True)


if __name__ == "__main__":
    main()
