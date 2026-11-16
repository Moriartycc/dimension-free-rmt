"""Scalar inverse map used by the finite-population boundary solver."""

from common import np, brentq


class FiniteLaw:
    def condition(self, t):
        with np.errstate(divide="ignore"):
            return np.sum(self.d2 / (t - self.d) ** 2) - 1

    def inverse(self, z):
        return brentq(
            lambda u: self.sn * self.mapping(u).real - z,
            -max(100.0, 4 * self.d[0] + 2 * abs(z) / self.sn),
            self.lower,
            xtol=1e-13,
        )
