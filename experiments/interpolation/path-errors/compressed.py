"""Random moment compression of the sampled finite tail."""

from core import torch, Spectrum, N


class CompressedSpectrum(Spectrum):
    def g_stable(self, data, k, z, w, head=128):
        w = torch.as_tensor(w, device=self.device, dtype=self.dtype).reshape(-1)
        z = torch.as_tensor(z, device=self.device, dtype=self.dtype)
        a = self.s[None, :] / (-w[:, None] * self.s[None, :] - z)
        ta, tb = self.tail(z, w)
        # Deterministic trace includes all compressed eigenvalues, not just Gram contributions.
        power = torch.ones_like(w)
        ctrace = torch.zeros_like(w)
        for l in range(self.order):
            ctrace -= power * data["sums"][l] / z
            power *= -w * self.cref / z
        if k == 0:
            return a.sum(-1) + ta + ctrace
        X = data["head"]
        Y = X[:, :k, head:].to(self.dtype)
        H = X[:, :k, :head].to(self.dtype)
        at = a[:, head:]
        K = (Y * at[:, None, :]) @ Y.transpose(-1, -2)
        L = (Y * (at * at)[:, None, :]) @ Y.transpose(-1, -2)
        # One real batched product evaluates both complex matrix polynomials.
        exponents = torch.arange(self.order, device=self.device)
        powers = (-w[:, None] * self.cref / z) ** exponents[None, :]
        powers[:, 0] = 1
        powers = torch.where((w == 0)[:, None] & (exponents > 0)[None, :], 0, powers)
        ck = -powers / z
        cl = torch.zeros_like(ck)
        cl[:, 1:] = exponents[None, 1:] * self.cref * powers[:, :-1] / (z * z)
        weights = torch.stack([ck.real, ck.imag, cl.real, cl.imag], dim=1)
        C = data["moments"].reshape(w.shape[0], self.order, -1).to(weights.dtype)
        poly = torch.bmm(weights, C).reshape(w.shape[0], 4, N, N)[:, :, :k, :k]
        K += torch.complex(poly[:, 0], poly[:, 1])
        L += torch.complex(poly[:, 2], poly[:, 3])
        ix = torch.arange(k, device=self.device)
        K[:, ix, ix] += (1 + ta)[:, None]
        L[:, ix, ix] += tb[:, None]
        sol = torch.linalg.solve(K, torch.cat([L, H], dim=-1))
        U = sol[:, :, k:]
        small = H.transpose(-1, -2) @ U
        ih = torch.arange(head, device=self.device)
        small[:, ih, ih] += 1 / a[:, :head]
        deriv = U.transpose(-1, -2) @ (L @ U)
        deriv[:, ih, ih] += 1
        correction = torch.linalg.solve(small, deriv).diagonal(dim1=-2, dim2=-1).sum(-1)
        return (
            at.sum(-1)
            + ta
            + ctrace
            - sol[:, :, :k].diagonal(dim1=-2, dim2=-1).sum(-1)
            + correction
        )
