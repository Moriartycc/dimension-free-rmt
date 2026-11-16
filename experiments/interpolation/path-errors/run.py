"""Preflight and checkpointed production for the finite-law grid."""

from model import FIT_COUNT, FIT_BATCH, TRIALS, EVAL_COUNT, Law, HERE, D, Spectrum

import numpy as np
from core import torch
from model import N
from model import CompressedSpectrum
from model import subset
from prepare import file_hash
import argparse
import json
from engine import clock
from engine import load_data
from engine import backward
from engine import forward
from engine import end_trace


def fit_prefixes():
    return [f"fit_{i:04d}" for i in range(0, FIT_COUNT, FIT_BATCH)]


def eval_prefixes():
    return [f"eval_{i:04d}" for i in range(0, TRIALS, EVAL_COUNT)]


def preflight(name):
    law = Law(name)
    folder = HERE / name
    pop = np.load(folder / "population.npz")
    col = int(np.argmin(abs(pop["x"] - law.lower_edge())))
    point = col
    z, w0 = pop["z"].flat[point], pop["w0"].flat[point]
    linear = law.empty(z, w0)
    scale = 1 + abs(linear)
    fit = load_data(folder, fit_prefixes(), D)
    info = dict(law=name, point=point, gpu=torch.cuda.get_device_name(), timings={})
    outputs = {}
    for m in (256, 512, 1024):
        started = clock()
        paths = backward(fit, law, z, m, chunk_size=128)
        outputs[f"M{m}_paths"] = paths
        outputs[f"M{m}_g"] = np.array([law.empty(z, u) for u in paths[:, 0]])
        info["timings"][f"M{m}"] = clock() - started
        print(name, "preflight M", m, info["timings"][f"M{m}"], flush=True)
    del fit
    torch.cuda.empty_cache()
    data = load_data(folder, ["eval_0000"], D)
    started = clock()
    paths, diagnostic = forward(data, law, z, w0)
    gn = end_trace(data, law, z, np.zeros(EVAL_COUNT, complex))
    gf = end_trace(data, law, z, paths[:, -1])
    info["timings"]["forward32"] = clock() - started
    errors = np.array(
        [
            (gn - outputs["M1024_g"][0]) / scale,
            (gn - linear) / scale,
            (gf - linear) / scale,
        ]
    )
    outputs.update(forward_paths=paths, errors=errors, z=z, w0=w0, scale=scale)
    sem = np.std(errors, axis=1, ddof=1) / np.sqrt(TRIALS)
    info["fit_se_over_eval_se"] = float(
        np.std(outputs["M256_g"], ddof=1) / 2 / scale / sem[0]
    )
    info["forward"] = diagnostic
    reference = folder / "inputs" / "eval_reference.npy"
    metadata = json.loads((folder / "prepared.json").read_text())
    assert file_hash(reference) == metadata["sha256"][reference.name]
    raw = torch.tensor(np.load(reference), device="cuda")
    direct, compressed = Spectrum(law, D), CompressedSpectrum(law, D)
    assert not bool(torch.any(direct.m)) and not bool(torch.any(compressed.m))
    audits = []
    for k in (0, 199, 399):
        w = paths[:2, k + 1]
        truth = direct.g_stable(raw, k, z, w, head=128)
        value = compressed.g_stable(subset(data, 0, 2), k, z, w)
        audits.extend((abs(truth - value) / (1 + abs(truth))).cpu().numpy())
    for audit_col in (0, 12, 24):
        zz, ww0 = pop["z"][0, audit_col], pop["w0"][0, audit_col]
        for k in (0, 199, 399):
            ww = np.full(2, ww0 * (1 - (k + 1) / N))
            truth = direct.g_stable(raw, k, zz, ww, head=128)
            value = compressed.g_stable(subset(data, 0, 2), k, zz, ww)
            audits.extend((abs(truth - value) / (1 + abs(truth))).cpu().numpy())
    exact_empty = np.sum(
        law.sigma(np.arange(1, D + 1)) / (-z - w0 * law.sigma(np.arange(1, D + 1)))
    )
    info["empty_error"] = float(abs(exact_empty - linear) / (1 + abs(exact_empty)))
    info["compression_error"] = float(max(audits))
    info["estimated_gpu_hours_per_law"] = float(
        425
        * (
            info["timings"]["M1024"]
            + info["timings"]["M256"]
            + info["timings"]["forward32"] * 400 / 32
        )
        / 3600
    )
    info["passed"] = bool(
        info["empty_error"] < 1e-9
        and info["compression_error"] < 1e-7
        and diagnostic["max_residual"] < 1e-7
        and diagnostic["max_series_ratio"] < 0.3
        and info["fit_se_over_eval_se"] < 0.5
    )
    np.savez_compressed(folder / "preflight.npz", **outputs)
    (folder / "preflight.json").write_text(json.dumps(info, indent=2))
    print("PREFLIGHT", json.dumps(info), flush=True)
    if not info["passed"]:
        raise RuntimeError("Preflight requires review; production must not start.")


def production(name, col, row_start=0, row_stop=17):
    assert 0 <= row_start < row_stop <= 17
    law = Law(name)
    folder = HERE / name
    checks = json.loads((folder / "preflight.json").read_text())
    assert checks["passed"]
    pop = np.load(folder / "population.npz")
    output = folder / "results"
    output.mkdir(exist_ok=True)
    fit = load_data(folder, fit_prefixes(), D)
    evaluation = load_data(folder, eval_prefixes(), D)
    assert evaluation["count"] == TRIALS
    print(
        "READY",
        name,
        "column",
        col,
        "rows",
        row_start,
        row_stop,
        "M",
        fit["count"],
        "trials",
        evaluation["count"],
        "GPU",
        torch.cuda.get_device_name(),
        flush=True,
    )
    # Reuse the calibrated lower-edge fit rather than repeating it.
    pre = np.load(folder / "preflight.npz")
    for row in range(row_start, row_stop):
        index = row * 25 + col
        destination = output / f"cell_{index:03d}.npz"
        if destination.exists():
            with np.load(destination) as saved:
                assert (
                    int(saved["trials"]) == TRIALS
                    and int(saved["D"]) == D
                    and int(saved["M"]) == FIT_COUNT
                )
            continue
        started = clock()
        print("START", name, "cell", index, flush=True)
        z, w0 = pop["z"].flat[index], pop["w0"].flat[index]
        linear = law.empty(z, w0)
        if index == checks["point"]:
            bp, blocks = pre["M1024_paths"], pre["M256_paths"]
        else:
            bp = backward(fit, law, z, FIT_COUNT, chunk_size=128)
            blocks = backward(fit, law, z, 256, chunk_size=128)
        bg = law.empty(z, bp[0, 0])
        sub_g = np.array([law.empty(z, u) for u in blocks[:, 0]])
        fit_se = np.std(sub_g, ddof=1) / 2
        gn, gf, wn = [np.empty(TRIALS, complex) for _ in range(3)]
        first_cross = np.full(TRIALS, -1, int)
        min_im = np.empty(TRIALS)
        fallback = 0
        max_residual = 0.0
        max_ratio = 0.0
        for lo in range(0, TRIALS, EVAL_COUNT):
            hi = min(lo + EVAL_COUNT, TRIALS)
            data = subset(evaluation, lo, hi)
            paths, diagnostic = forward(data, law, z, w0)
            wn[lo:hi] = paths[:, -1]
            gn[lo:hi] = end_trace(data, law, z, np.zeros(hi - lo, complex))
            gf[lo:hi] = end_trace(data, law, z, wn[lo:hi])
            crossing = paths.imag < 0
            first_cross[lo:hi] = np.where(crossing.any(1), crossing.argmax(1), -1)
            min_im[lo:hi] = paths.imag.min(1)
            fallback += diagnostic["fallback"]
            max_residual = max(max_residual, diagnostic["max_residual"])
            max_ratio = max(max_ratio, diagnostic["max_series_ratio"])
        if max_residual > 1e-7 or max_ratio >= 0.3:
            raise RuntimeError(
                ("production numerical audit", name, index, max_residual, max_ratio)
            )
        scale = 1 + abs(linear)
        errors = np.array(
            [(gn - bg) / scale, (gn - linear) / scale, (gf - linear) / scale]
        )
        assert np.isfinite(errors).all() and bp.imag[:, :-1].min() > 0
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            errors=errors,
            z=z,
            zeta=pop["zeta"].flat[index],
            g_n0=gn,
            g_nf=gf,
            w_n=wn,
            w_backward=bp[0, 0],
            sub_w=blocks[:, 0],
            g_linear=linear,
            fit_se=fit_se / scale,
            first_cross=first_cross,
            min_im=min_im,
            fallback=fallback,
            max_residual=max_residual,
            max_series_ratio=max_ratio,
            D=D,
            n=N,
            M=FIT_COUNT,
            trials=TRIALS,
            seconds=clock() - started,
            gpu=torch.cuda.get_device_name(),
        )
        temporary.replace(destination)
        print("SAVED", name, index, "seconds", clock() - started, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "production"])
    parser.add_argument("law", choices=["power", "critical"])
    parser.add_argument("--column", type=int)
    parser.add_argument("--row-start", type=int)
    parser.add_argument("--row-stop", type=int)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.law)
    else:
        row_start, row_stop = args.row_start, args.row_stop
        if row_start is None and row_stop is None:
            row_start, row_stop = 0, 17
        production(args.law, args.column, row_start, row_stop)
