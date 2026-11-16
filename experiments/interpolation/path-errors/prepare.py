"""Generate canonical, checkpointed pilot inputs once per law."""

import sys
from model import HERE
import numpy as np
from core import torch
from scipy.optimize import root
from model import N
from model import HEAD
from model import ORDER
from model import D
from model import CUTS
from model import TRIALS
from model import FIT_COUNT
from model import FIT_BATCH
from model import EVAL_COUNT
from model import FIT_SEED
from model import EVAL_SEED
from model import Law
import argparse
import hashlib
import json
import time
import gc
import ctypes


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(law, count, seed, prefix, save_cuts, raw_reference=False):
    folder = HERE / law.name / "inputs"
    paths = [folder / f"{prefix}_D{d}.pt" for d in save_cuts]
    if all(path.exists() for path in paths) and (
        not raw_reference or (folder / "eval_reference.npy").exists()
    ):
        return
    started = time.perf_counter()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    head = (
        torch.randn((HEAD, count, N), generator=gen, device="cuda")
        .permute(1, 2, 0)
        .contiguous()
    )
    fingerprint = hashlib.sha256(head.cpu().numpy().tobytes()).hexdigest()
    raw = None
    if raw_reference:
        raw = np.lib.format.open_memmap(
            folder / "eval_reference.npy",
            mode="w+",
            dtype=np.float32,
            shape=(2, N, CUTS[0]),
        )
        raw[:, :, :HEAD] = head[:2].cpu().numpy()
    moments = torch.zeros((count, ORDER, N, N), device="cuda", dtype=torch.float64)
    sums = torch.zeros(ORDER, device="cuda", dtype=torch.float64)
    cref = float(law.sigma(HEAD + 1))
    lo = HEAD
    for end in save_cuts:
        while lo < end:
            hi = min(lo + 1024, end)
            draws = torch.randn((hi - lo, count, N), generator=gen, device="cuda")
            draws = draws.permute(1, 2, 0).contiguous()
            if raw is not None and hi <= CUTS[0]:
                raw[:, :, lo:hi] = draws[:2].cpu().numpy()
            x = draws.double()
            power = torch.tensor(law.sigma(np.arange(lo + 1, hi + 1)), device="cuda")
            ratio = power / cref
            for k in range(ORDER):
                moments[:, k] += (x * power[None, None, :]) @ x.transpose(-1, -2)
                sums[k] += power.sum()
                power = power * ratio
            lo = hi
        destination = folder / f"{prefix}_D{end}.pt"
        temporary = destination.with_suffix(".tmp.pt")
        torch.save(
            dict(
                head=head.cpu(),
                moments=moments.cpu(),
                sums=sums.cpu(),
                count=count,
                d=end,
                law=law.name,
                seed=seed,
                fingerprint=fingerprint,
                gpu=torch.cuda.get_device_name(),
                headcut=HEAD,
                order=ORDER,
            ),
            temporary,
        )
        temporary.replace(destination)
        gc.collect()
        if sys.platform == "linux":
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        if raw is not None:
            raw.flush()
        torch.cuda.synchronize()
        print(
            law.name,
            prefix,
            "saved",
            end,
            "seconds",
            time.perf_counter() - started,
            flush=True,
        )
    del moments, head
    torch.cuda.empty_cache()


def prepare(name):
    law = Law(name)
    folder = HERE / name
    (folder / "inputs").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    counts = {}
    for start in range(0, FIT_COUNT, FIT_BATCH):
        prefix = f"fit_{start:04d}"
        generate(law, FIT_BATCH, FIT_SEED + start, prefix, (D,))
        counts[f"{prefix}_D{D}.pt"] = FIT_BATCH
    for start in range(0, TRIALS, EVAL_COUNT):
        count = min(EVAL_COUNT, TRIALS - start)
        prefix = f"eval_{start:04d}"
        generate(law, count, EVAL_SEED + start, prefix, (D,), raw_reference=start == 0)
        counts[f"{prefix}_D{D}.pt"] = count
    x, y, points = law.grid()
    zs = law.b + law.a * points
    w0 = np.empty(zs.shape, complex)
    for col in range(len(x)):
        w = -N + 0j
        heights = np.r_[np.geomspace(20, zs[-1, col].imag, 70), zs[::-1, col].imag]
        for index, height in enumerate(heights):
            z = zs[0, col].real + 1j * height

            def fun(pair):
                ww = complex(*pair)
                value = ww + N / (1 + law.empty(z, ww))
                return [value.real, value.imag]

            solution = root(fun, [w.real, w.imag], tol=1e-11)
            w = complex(*solution.x)
            if np.linalg.norm(fun(solution.x)) > 1e-7 or w.imag <= 0:
                raise RuntimeError(("population branch", name, col, height))
            if index >= 70:
                w0[len(y) - 1 - (index - 70), col] = w
    np.savez(
        folder / "population.npz",
        x=x,
        y=y,
        zeta=points,
        z=zs,
        w0=w0,
        support=law.support(),
        a=law.a,
        b=law.b,
        n=N,
        D=D,
    )
    hashes = {key: file_hash(folder / "inputs" / key) for key in counts}
    hashes["eval_reference.npy"] = file_hash(folder / "inputs" / "eval_reference.npy")
    metadata = dict(
        law=name,
        n=N,
        D=D,
        M=FIT_COUNT,
        trials=TRIALS,
        counts=counts,
        sha256=hashes,
        gpu=torch.cuda.get_device_name(),
        seconds=time.perf_counter() - started,
        head=HEAD,
        order=ORDER,
        shape=list(points.shape),
    )
    temporary = folder / "prepared.tmp.json"
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(folder / "prepared.json")
    print("PREPARED", name, metadata["seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("law", choices=["power", "critical"])
    prepare(parser.parse_args().law)
