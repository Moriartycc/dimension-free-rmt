"""Real-boundary filter risks, including divided-difference diagonal limits."""

from common import np, torch, Q, TAU2


def gpu(a):
    return torch.as_tensor(
        a, dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu"
    )


class RealRisk:
    def filter(self, psi, minus_derivative):
        # Rows are outer x', columns inner x; atom at zero is explicit.
        psi = gpu(psi)
        minus_derivative = gpu(minus_derivative)
        kernel = (psi[None, :] - psi[:, None]) / self.safe
        kernel.diagonal().copy_(minus_derivative)
        L = self.atom + psi @ self.dm
        C = kernel @ self.dm + ((1 - psi) / self.x)[:, None] * self.atom
        phi = (1 - psi) / self.x
        V = TAU2 * torch.sum(self.mass * phi**2)
        B = (L**2 + torch.sum(self.mass[:, None] * C**2, dim=0)) @ self.weights
        return np.stack(
            [np.full(len(Q), V.item()), B.cpu().numpy(), V.item() + B.cpu().numpy()]
        )

    def gf(self, t):
        if float(t) in self.cache:
            return self.cache[float(t)]
        e = torch.exp(-t * self.x)
        # Stable divided exponential, including its continuous diagonal.
        lo = torch.minimum(self.x[:, None], self.x[None, :])
        dist = abs(self.delta)
        safe_dist = dist.clone()
        safe_dist.diagonal().fill_(1)
        kernel = torch.exp(-t * lo) * (-torch.expm1(-t * dist)) / safe_dist
        kernel.diagonal().copy_(t * e)
        phi = -torch.expm1(-t * self.x) / self.x
        L = self.atom + e @ self.dm
        Lp = (-self.x * e) @ self.dm
        C = kernel @ self.dm + phi[:, None] * self.atom
        V = TAU2 * torch.sum(self.mass * phi**2)
        Vp = 2 * TAU2 * torch.sum(self.mass * phi * e)
        B = (L**2 + torch.sum(self.mass[:, None] * C**2, dim=0)) @ self.weights
        # Differentiate the positive-spectrum convolution before adding the
        # atom; L-xC suffers cancellation at late times.
        Cp = (e[None, :] - self.x[:, None] * kernel) @ self.dm + e[:, None] * self.atom
        Bp = 2 * (L * Lp + torch.sum(self.mass[:, None] * C * Cp, dim=0)) @ self.weights
        value = np.stack(
            [
                np.full(len(Q), V.item()),
                B.cpu().numpy(),
                V.item() + B.cpu().numpy(),
                Vp.item() + Bp.cpu().numpy(),
            ]
        )
        self.cache[float(t)] = value
        return value
