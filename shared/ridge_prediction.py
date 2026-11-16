"""Inverse-series derivatives for iterated ridge, k=1,2,4."""

from common import np, brentq
from numpy.polynomial.polynomial import polymul

KS = (1, 2, 4)


def edge_and_root(s, weights, trace, n, c=0.0, edge=None, kapedge=None):
    def f(kap):
        return -kap + (trace - np.sum(weights * s * s / (kap + s))) / n

    if edge is None:
        root = brentq(
            lambda t: np.sum(weights * (s / (np.exp(t) + s)) ** 2) / n - 1,
            -120,
            10,
            xtol=1e-13,
        )
        kapedge = np.exp(root)
        edge = f(kapedge)
        assert edge > 0
    logkap = brentq(
        lambda t: f(np.exp(t)) / edge + c,
        np.log(kapedge),
        np.log(kapedge) + 60,
        xtol=1e-13,
    )
    return edge, kapedge, np.exp(logkap)


def predict(s, weights, trace, n, c, source_s, theta, edge=None, kapedge=None):
    edge, kapedge, kap = edge_and_root(s, weights, trace, n, c, edge, kapedge)
    d = s / edge
    w0 = -kap / edge
    order = 7
    fcoef = np.array(
        [0.0, 1 - np.sum(weights * (d / (w0 - d)) ** 2) / n]
        + [
            (-1.0) ** m * np.sum(weights * d * d / (w0 - d) ** (m + 1)) / n
            for m in range(2, order + 1)
        ]
    )
    coeff = np.zeros(order + 1)
    coeff[0] = w0
    coeff[1] = 1 / fcoef[1]
    for m in range(2, order + 1):
        delta = coeff.copy()
        delta[0] = 0
        power = delta.copy()
        known = 0.0
        for r in range(2, m + 1):
            power = polymul(power, delta)[: order + 1]
            if len(power) > m:
                known += fcoef[r] * power[m]
        coeff[m] = -known / fcoef[1]
    d = source_s / edge
    inv = np.zeros((4, len(d)))
    inv[0] = 1 / (w0 - d)
    for a in range(1, 4):
        inv[a] = -inv[0] * sum(coeff[i] * inv[a - i] for i in range(1, a + 1))
    A = d[None, :] * inv
    A[0] = w0 / (w0 - d)
    gram = np.einsum(
        "ij,lj,jq->ilq", A, A, source_s[:, None] ** 2 * theta**2, optimize=True
    )
    answer = []
    for k in KS:
        V = sum((k - abs(k - m)) * c ** (m - 1) * coeff[m] for m in range(1, 2 * k)) - 1
        B = np.zeros(theta.shape[1])
        for a in range(k):
            for b in range(k):
                hab = sum(
                    coeff[a + b - i - j + 1] * gram[i, j]
                    for i in range(a + 1)
                    for j in range(b + 1)
                )
                B += c ** (a + b) * hab
        assert V > 0 and np.all(B > 0), (n, c, k, V, B)
        answer.append(np.stack([np.full_like(B, V), B]))
    return np.array(answer), edge
