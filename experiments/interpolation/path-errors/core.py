"""Deflated weighted-resolvent evaluation for the finite-law experiment."""

import torch

N = 400


class Spectrum:
    def tail(self, z, w):
        a = torch.zeros_like(w)
        b = torch.zeros_like(w)
        power = torch.ones_like(w)
        for l in range(len(self.m) - 1):
            a = a - power * self.m[l] / z
            b = b + (l + 1) * power * self.m[l + 1] * self.sref / (z * z)
            power = power * (-w * self.sref / z)
        return a, b

    def g_stable(self, X, k, z, w, head=64):
        """Deflate large population eigenvalues before the sample-space solve."""
        if k == 0:
            return self.g(X, k, z, w)
        w = torch.as_tensor(w, device=self.device, dtype=self.dtype).reshape(-1)
        z = torch.as_tensor(z, device=self.device, dtype=self.dtype)
        a = self.s[None, :] / (-w[:, None] * self.s[None, :] - z)
        ta, tb = self.tail(z, w)
        Y = X[:, :k, head:].to(self.dtype)
        H = X[:, :k, :head].to(self.dtype)
        at = a[:, head:]
        K = (Y * at[:, None, :]) @ Y.transpose(-1, -2)
        L = (Y * (at * at)[:, None, :]) @ Y.transpose(-1, -2)
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
            - sol[:, :, :k].diagonal(dim1=-2, dim2=-1).sum(-1)
            + correction
        )

    def g(self, X, k, z, w):
        """X: batch x N x d standard Gaussian. k observations, returns G_(k+1)."""
        w = torch.as_tensor(w, device=self.device, dtype=self.dtype).reshape(-1)
        z = torch.as_tensor(z, device=self.device, dtype=self.dtype)
        a = self.s[None, :] / (-w[:, None] * self.s[None, :] - z)
        ta, tb = self.tail(z, w)
        total = a.sum(-1) + ta
        if k == 0:
            return total
        Y = X[:, :k, :].to(self.dtype)
        K = (Y * a[:, None, :]) @ Y.transpose(-1, -2)
        L = (Y * (a * a)[:, None, :]) @ Y.transpose(-1, -2)
        ix = torch.arange(k, device=self.device)
        K[:, ix, ix] += (1 + ta)[:, None]
        L[:, ix, ix] += tb[:, None]
        V = torch.linalg.solve(K, L)
        return total - V.diagonal(dim1=-2, dim2=-1).sum(-1)
