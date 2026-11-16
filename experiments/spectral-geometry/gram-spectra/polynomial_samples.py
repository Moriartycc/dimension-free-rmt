"""Independent polynomial spectra with an analytic expectation tail."""

from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "4"
import numpy as np
from scipy.special import zeta
import torch


def polynomial_spectra(n, reps):
    directory = HERE / "replicate_cache"
    directory.mkdir(exist_ok=True)
    path = directory / f"polynomial_n{n}_r{reps}.npz"
    if path.exists():
        with np.load(path) as saved:
            values = saved["eigenvalues"]
        assert values.shape == (reps, n) and np.isfinite(values).all()
        return values

    values = np.empty((reps, n))
    max_check = 0.0
    cutoff = max(16384, 256 * n)
    tail = zeta(1.2, cutoff + 1) / n
    block = 8192
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    weights = torch.tensor(
        np.arange(1, cutoff + 1, dtype=float) ** -0.6 / np.sqrt(n),
        dtype=torch.float32,
        device=device,
    )
    generator = torch.Generator(device=device)
    for r in range(reps):
        generator.manual_seed(20261110 + 10000 * n + r)
        gram = torch.zeros((n, n), dtype=torch.float64, device=device)
        if r == 0:
            reference = torch.zeros_like(gram)
        for offset in range(0, cutoff, block):
            w = weights[offset : offset + block]
            x = torch.randn((n, len(w)), generator=generator, device=device)
            x.mul_(w)
            gram += (x @ x.T).double()
            if r == 0:
                xd = x.double()
                reference.addmm_(xd, xd.T)
        values[r] = np.linalg.eigvalsh(gram.cpu().numpy()) + tail
        if r == 0:
            exact = np.linalg.eigvalsh(reference.cpu().numpy()) + tail
            max_check = max(max_check, np.max(np.abs(values[r] / exact - 1)))
            assert max_check < 2e-4, max_check
        if (r + 1) % 100 == 0:
            print(f"Polynomial n={n}: {r + 1}/{reps}", flush=True)
    assert (values > 0).all() and np.isfinite(values).all()
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        eigenvalues=values,
        cutoff=cutoff,
        paired_double_relative_error=max_check,
    )
    os.replace(temporary, path)
    return values


if __name__ == "__main__":
    for size in (128, 256, 512, 1024):
        polynomial_spectra(size, 2000)
