"""Finite population model: alpha=beta=0.2, exactly 65536 coordinates."""

from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
import numpy as np
import torch
from core import Spectrum as OriginalSpectrum
from compressed import CompressedSpectrum as OriginalCompressedSpectrum
from scipy.optimize import root, brentq

N = 400
P = 1.2
HEAD = 4096
ORDER = 16
D = 65536
CUTS = (D,)
TRIALS = 400
FIT_COUNT = 1024
FIT_BATCH = 64
EVAL_COUNT = 32
FIT_SEED = 142026711
EVAL_SEED = 142026977


class Law:
    def __init__(self, name):
        if name not in ("power", "critical"):
            raise ValueError(name)
        self.name = name
        self.a = N * self.sigma(N)
        self.b = self.tail_moments(N)[0] if name == "critical" else 0.0
        self.explicit = self.sigma(np.arange(1, 32769))
        self.remainder = self.tail_moments(32768)

    def sigma(self, j):
        j = np.asarray(j, dtype=float)
        if self.name == "power":
            return j ** (-P)
        return 1 / (j * (1 + np.log(j)) ** P)

    @lru_cache(maxsize=None)
    def tail_moments(self, d, order=ORDER):
        # Finite sum only: there is no population mass beyond D.
        values = self.sigma(np.arange(d + 1, D + 1))
        return np.array([np.sum(values**p) for p in range(1, order + 1)])

    def empty(self, z, w):
        return (
            np.sum(self.explicit / (-z - w * self.explicit))
            - np.polynomial.polynomial.polyval(-w / z, self.remainder) / z
        )

    def population_w(self, z):
        w = -N + 0j
        for eta in np.geomspace(max(20.0, z.imag), z.imag, 180):
            zz = z.real + 1j * eta

            def residual(pair):
                u = complex(*pair)
                value = u + N / (1 + self.empty(zz, u))
                return [value.real, value.imag]

            solution = root(residual, [w.real, w.imag], tol=1e-11)
            w = complex(*solution.x)
            if np.linalg.norm(residual(solution.x)) > 1e-7 or w.imag <= 0:
                raise RuntimeError(("population", self.name, zz, solution.message))
        return w

    def lower_edge(self):
        values = self.sigma(np.arange(1, D + 1)) / self.sigma(N)
        u = brentq(
            lambda u: np.sum((values / (u - values)) ** 2) / N - 1,
            -max(100.0, values[0]),
            -1e-8,
        )
        return u + np.sum(u * values / (u - values)) / N - self.b / self.a

    def grid(self):
        left = 1.0 if self.name == "power" else -2.0
        x = np.arange(25) * 0.25 + left
        y = np.geomspace(0.005, 1.0, 17)
        return x, y, x[None, :] + 1j * y[:, None]

    def support(self):
        from scipy.optimize import minimize_scalar

        values = self.sigma(np.arange(1, D + 1)) / self.sigma(N)

        def condition(u):
            return np.sum((values / (u - values)) ** 2) / N - 1

        roots = [
            brentq(condition, -max(100.0, values[0]), -1e-8),
            brentq(condition, values[0] * (1 + 1e-10), 3 * values[0]),
        ]
        for high, low in zip(values[:-1], values[1:]):
            left, right = low * (1 + 1 / np.sqrt(N)), high * (1 - 1 / np.sqrt(N))
            if left >= right:
                continue
            minimum = minimize_scalar(condition, bounds=(left, right), method="bounded")
            if minimum.fun < 0:
                roots.extend(
                    [
                        brentq(condition, left, minimum.x),
                        brentq(condition, minimum.x, right),
                    ]
                )
        roots = np.sort(roots)
        return np.array(
            [u + np.sum(u * values / (u - values)) / N - self.b / self.a for u in roots]
        ).reshape(-1, 2)


class Spectrum(OriginalSpectrum):
    """Reuse the existing deflated resolvent with law-specific coefficients."""

    def __init__(self, law, d, dtype=torch.complex128, device="cuda"):
        self.d, self.device, self.dtype = d, device, dtype
        realtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.s = torch.tensor(
            law.sigma(np.arange(1, d + 1)), device=device, dtype=realtype
        )
        self.sref = float(law.sigma(d + 1))
        moments = law.tail_moments(d)
        self.m = torch.tensor(
            moments / self.sref ** np.arange(len(moments)),
            device=device,
            dtype=realtype,
        )


class CompressedSpectrum(Spectrum):
    def __init__(self, law, d, dtype=torch.complex128):
        super().__init__(law, HEAD, dtype=dtype)
        self.total, self.headcut, self.order = d, HEAD, ORDER
        self.cref = float(law.sigma(HEAD + 1))
        self.sref = float(law.sigma(d + 1))
        moments = law.tail_moments(d)
        realtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.m = torch.tensor(
            moments / self.sref ** np.arange(len(moments)),
            device=self.device,
            dtype=realtype,
        )

    g_stable = OriginalCompressedSpectrum.g_stable


def subset(data, start, stop):
    return dict(
        data,
        head=data["head"][start:stop],
        moments=data["moments"][start:stop],
        count=stop - start,
    )
