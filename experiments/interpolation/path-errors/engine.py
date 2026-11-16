"""Paired cutoff, nested-M, precision, and runtime diagnostics at one point."""

import numpy as np
from core import torch
from scipy.optimize import root
from model import N
from model import HEAD
from model import ORDER
from model import CompressedSpectrum
from model import subset
from prepare import file_hash
import json
import time


def clock():
    torch.cuda.synchronize()
    return time.perf_counter()


def load_data(folder, prefixes, d, metadata_filename="prepared.json"):
    metadata = json.loads((folder / metadata_filename).read_text())
    result = None
    offset = 0
    for prefix in prefixes:
        path = folder / "inputs" / f"{prefix}_D{d}.pt"
        assert file_hash(path) == metadata["sha256"][path.name], path
        chunk = torch.load(path, map_location="cpu", weights_only=False)
        assert (chunk["law"], chunk["d"], chunk["headcut"], chunk["order"]) == (
            folder.name,
            d,
            HEAD,
            ORDER,
        )
        if result is None:
            count = sum(metadata["counts"][f"{p}_D{d}.pt"] for p in prefixes)
            result = dict(
                count=count,
                d=d,
                head=torch.empty((count, N, HEAD), device="cuda", dtype=torch.float32),
                moments=torch.empty(
                    (count, ORDER, N, N), device="cuda", dtype=torch.float64
                ),
                sums=chunk["sums"].cuda(),
            )
        assert torch.equal(chunk["sums"], result["sums"].cpu())
        end = offset + chunk["count"]
        result["head"][offset:end].copy_(chunk["head"])
        result["moments"][offset:end].copy_(chunk["moments"])
        offset = end
    assert offset == result["count"]
    return result


def backward(data, law, z, group_size, chunk_size=32):
    sp = CompressedSpectrum(law, data["d"])
    groups = data["count"] // group_size
    w = torch.zeros(groups, device="cuda", dtype=torch.complex128)
    paths = np.empty((groups, N + 1), complex)
    paths[:, N] = 0
    chunk_size = min(chunk_size, group_size)
    for k in range(N, 0, -1):
        totals = torch.zeros_like(w)
        for lo in range(0, data["count"], chunk_size):
            hi = min(lo + chunk_size, data["count"])
            group = lo // group_size
            values = sp.g_stable(
                subset(data, lo, hi), k - 1, z, w[group].expand(hi - lo)
            )
            totals[group] += values.sum()
        w -= 1 / (1 + totals / group_size)
        paths[:, k - 1] = w.cpu().numpy()
        if not np.isfinite(paths[:, k - 1]).all():
            raise RuntimeError(("nonfinite backward", law.name, k, z))
    return paths


def forward(data, law, z, w0, single=False):
    dtype = torch.complex64 if single else torch.complex128
    sp = CompressedSpectrum(law, data["d"], dtype=dtype)
    double = CompressedSpectrum(law, data["d"])
    original = data
    if single:
        data = dict(data, moments=data["moments"].float(), sums=data["sums"].float())
    count = data["count"]
    w = torch.full((count,), complex(w0), device="cuda", dtype=dtype)
    delta = -w / N
    paths = np.empty((count, N + 1), complex)
    paths[:, 0] = w.cpu().numpy()
    fallback = 0
    max_residual = 0.0
    for k in range(1, N + 1):
        previous = w.clone()
        predicted = previous + delta
        w = predicted.clone()
        for iteration in range(20):
            trace = sp.g_stable(data, k - 1, z, w)
            target = previous + 1 / (1 + trace)
            residual = abs(target - w)
            tolerance = (
                (2e-5 + 8 * torch.finfo(torch.float32).eps * abs(w))
                if single
                else (1e-9 + 32 * torch.finfo(torch.float64).eps * abs(w))
            )
            if bool(((residual <= tolerance) & torch.isfinite(residual)).all()):
                break
            if iteration < 19:
                w = target
        bad = (
            torch.where((residual > tolerance) | ~torch.isfinite(residual))[0]
            .cpu()
            .tolist()
        )
        for i in bad:
            dd = subset(original, i, i + 1)
            p = complex(previous[i].item())
            prediction = complex(predicted[i].item())

            def fun(pair):
                u = complex(*pair)
                tr = double.g_stable(dd, k - 1, z, [u]).item()
                value = u - p - 1 / (1 + tr)
                return [value.real, value.imag]

            candidates = []
            for start in (prediction, complex(w[i].item()), p):
                if not np.isfinite(start):
                    continue
                solution = root(fun, [start.real, start.imag], tol=1e-11)
                if np.linalg.norm(fun(solution.x)) < 1e-8:
                    candidates.append(complex(*solution.x))
            if not candidates:
                raise RuntimeError(
                    ("implicit solve failed", law.name, k, i, z, p, prediction)
                )
            w[i] = min(candidates, key=lambda u: abs(u - prediction))
            fallback += 1
        residual = abs(w - previous - 1 / (1 + sp.g_stable(data, k - 1, z, w)))
        if not bool(torch.isfinite(residual).all()):
            raise RuntimeError(("nonfinite implicit residual", k, z))
        max_residual = max(max_residual, residual.max().item())
        paths[:, k] = w.cpu().numpy()
        delta = w - previous
    return paths, dict(
        fallback=fallback,
        max_residual=max_residual,
        min_im=float(paths.imag.min()),
        max_series_ratio=float(np.max(abs(paths * sp.cref / z))),
    )


def end_trace(data, law, z, w):
    return CompressedSpectrum(law, data["d"]).g_stable(data, N - 1, z, w).cpu().numpy()
