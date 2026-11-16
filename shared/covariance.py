"""Power/critical spectra and finite-population quadrature."""

from common import np
from scipy.special import zeta
from numpy.polynomial.legendre import leggauss

DIM = 2**24


def spec(law, p, j):
    return j ** (-1 - p) if law == "power" else 1 / (j * (1 + np.log(j)) ** (1 + p))


def tail(law, p, h):
    if law == "power":
        return zeta(1 + p, h + 1)
    s = spec(law, p, float(h))
    der = -s / h * (1 + (1 + p) / (1 + np.log(h)))
    return (1 + np.log(h)) ** (-p) / p - s / 2 - der / 12


def finite_population(law, p, d=DIM, head=4096, step=0.75, order=12):
    if d <= head:
        s = spec(law, p, np.arange(1, d + 1, dtype=float))
        return s, np.ones(d), float(s.sum())
    j = np.arange(1, head + 1, dtype=float)
    lo = np.log(head + 0.5)
    hi = np.log(d + 0.5)
    bounds = np.linspace(lo, hi, int(np.ceil((hi - lo) / step)) + 1)
    x, w = leggauss(order)
    nodes = (
        (bounds[1:, None] + bounds[:-1, None]) / 2 + np.diff(bounds)[:, None] * x / 2
    ).ravel()
    weights = (np.diff(bounds)[:, None] * w / 2).ravel() * np.exp(nodes)
    s = spec(law, p, np.r_[j, np.exp(nodes)])
    weights = np.r_[np.ones(head), weights]
    trace = s[:head].sum() + tail(law, p, head) - tail(law, p, d)
    return s, weights, trace
