"""Finite-population support, boundary measures, and gradient-flow risk."""

from common import np, torch, json, time, gammaln, brentq, CASES
from covariance import finite_population, spec
from ridge_prediction import edge_and_root
from prediction import FiniteLaw
from risks import RealRisk, gpu
from scipy.optimize import minimize_scalar

D = 2**24
Q = np.array([16.0, 24.0])
TAU = 0.01


def sources(law, p, count=2048):
    j = np.arange(count, dtype=float)[:, None]
    logs = j * np.log(Q[None, :]) - gammaln(j + 1)
    theta = np.exp(logs - logs.max(0))
    theta /= np.linalg.norm(theta, axis=0)
    return spec(law, p, np.arange(1, count + 1, dtype=float)), theta


class Population(FiniteLaw):
    def __init__(self, law, p, n, fine=False):
        self.law, self.p, self.n = law, p, n
        args = dict(head=16384, step=0.375, order=20) if fine else {}
        self.sigma, self.weights, self.trace = finite_population(law, p, d=D, **args)
        self.sn = float(spec(law, p, float(n)))
        self.d = self.sigma / self.sn
        self.d2 = self.weights * self.d**2 / n
        self.second = float(self.weights @ (self.sigma**2))
        self.source_s, self.theta = sources(law, p)
        self.tail = float(
            self.trace - spec(law, p, np.arange(1, n + 1, dtype=float)).sum()
        )
        self.scale = {
            "a": n * self.sn,
            "b": self.tail if law == "critical" else 0.0,
            "tail": self.tail,
        }
        self.lower = brentq(self.condition, -max(100.0, self.d[0]), -1e-12)
        self.upper = brentq(self.condition, self.d[0] * (1 + 1e-10), 3 * self.d[0])
        edges = [self.lower, self.upper]
        head = 16384 if fine else 4096
        for high, low in zip(self.d[: head - 1], self.d[1:head]):
            left, right = low * (1 + 1 / np.sqrt(n)), high * (1 - 1 / np.sqrt(n))
            if left >= right:
                continue
            m = minimize_scalar(
                self.condition,
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-12},
            )
            if m.fun < 0:
                edges.extend(
                    [
                        brentq(self.condition, low * (1 + 1e-12), m.x),
                        brentq(self.condition, m.x, high * (1 - 1e-12)),
                    ]
                )
        self.bands = np.array(sorted(edges)).reshape(-1, 2)
        self.u0 = self.inverse(0)

    def mapping(self, u):
        return (
            u
            + self.trace / (self.n * self.sn)
            + np.sum(self.weights * self.d**2 / (u - self.d)) / self.n
        )

    def point(self, t, edge=False):
        sq = (t - self.d) ** 2
        if edge:
            h = 0.0
        else:
            hi = max(1.0, abs(t)) ** 2
            while np.sum(self.d2 / (sq + hi)) > 1:
                hi *= 4
            h = np.sqrt(
                brentq(
                    lambda h2: np.sum(self.d2 / (sq + h2)) - 1,
                    0,
                    hi,
                    xtol=1e-23,
                    rtol=2e-14,
                )
            )
        u = complex(t, h)
        z = self.mapping(u)
        st = -1 / u
        v = self.d * st
        prim = np.empty_like(v)
        small = abs(v) < 1e-4
        prim[small] = v[small] ** 2 * (0.5 + v[small] * (-2 / 3 + 0.75 * v[small]))
        prim[~small] = np.log1p(v[~small]) - v[~small] / (1 + v[~small])
        F = (np.angle(st) - np.sum(self.weights * prim.imag) / self.n) / np.pi
        return u, z, F

    def mesh(self, resolution=1):
        arrays = []
        supports = []
        for left, right in self.bands:
            count = (1000 if left < 0 else 180) * resolution
            a = np.linspace(0, np.pi, count + 1)
            t = np.sinh(
                np.arcsinh(left)
                + (np.arcsinh(right) - np.arcsinh(left)) * np.sin(a / 2) ** 2
            )
            t[0] = left
            t[-1] = right
            values = np.array([self.point(v, i in (0, count)) for i, v in enumerate(t)])
            u, z, F = values.T
            x = z.real * self.sn
            assert np.all(np.diff(x) > 0)
            dx = np.r_[np.diff(x)[0] / 2, (x[2:] - x[:-2]) / 2, np.diff(x)[-1] / 2]
            arrays.append((x, u, F.real, dx))
            supports.append(x[[0, -1]])
        x, u, F, dx = [np.concatenate([a[j] for a in arrays]) for j in range(4)]
        sj = self.source_s[:800]
        dj = sj / self.sn
        mass = self.sn * u.imag / np.pi * dx
        directional = (
            dj[None, :]
            * u.imag[:, None]
            / (np.pi * x[:, None] * abs(dj[None, :] - u[:, None]) ** 2)
            * dx[:, None]
        )
        atom = -self.u0 / (dj - self.u0)
        xr = (self.n * x - self.scale["b"]) / self.scale["a"]
        rho = (-1 / u).imag / np.pi
        check = dict(
            mass_error=float(max(abs(atom + directional.sum(0) - 1))),
            first_moment_error=float(max(abs((x @ directional) / sj - 1))),
            weighted_mass_error=float(abs(mass.sum() / (self.second / self.n) - 1)),
            gram_mass_error=float(abs(np.sum(rho * dx / self.sn) - 1)),
            cdf_endpoint_error=float(max(abs(F[0]), abs(F[-1] - 1))),
            nodes=len(x),
        )
        return dict(
            x=x,
            u=u,
            cdf=F,
            dx=dx,
            xr=xr,
            rho=rho,
            mass_omega=mass,
            directional=directional,
            atom=atom,
            support=np.array(supports),
        ), check


class FlowRisk(RealRisk):
    def __init__(self, data, pop):
        self.x = gpu(data["x"])
        self.mass = gpu(
            data["mass_omega"] * (pop.second / pop.n / data["mass_omega"].sum())
        )
        self.dm = gpu(
            data["directional"]
            * ((1 - data["atom"]) / data["directional"].sum(0))[None, :]
        )
        self.atom = gpu(data["atom"])
        self.weights = gpu(pop.source_s[:800, None] ** 2 * pop.theta[:800] ** 2)
        self.delta = self.x[:, None] - self.x[None, :]
        self.safe = self.delta.clone()
        self.safe.diagonal().fill_(1)
        self.cache = {}


def flow(pop, data):
    rr = FlowRisk(data, pop)
    s = np.geomspace(1e-12, 1e3, 151)
    times = s * pop.n / pop.tail
    gf = np.array([rr.gf(t) for t in times])
    initial = np.stack(
        [np.zeros(2), np.sum(pop.source_s[:, None] ** 2 * pop.theta**2, axis=0)]
    )
    initial = np.vstack([initial, initial[1]])
    inf = rr.filter(np.zeros_like(data["x"]), np.zeros_like(data["x"]))
    rootsall = []
    opts = []
    optR = []
    for qi in range(2):
        roots = []
        for i in np.flatnonzero(gf[:-1, 3, qi] * gf[1:, 3, qi] < 0):
            root = brentq(
                lambda logt: rr.gf(np.exp(logt))[3, qi],
                np.log(times[i]),
                np.log(times[i + 1]),
                xtol=1e-8,
            )
            roots.append(float(np.exp(root)))
        candidates = [0.0, *roots, np.inf]
        risks = [initial[2, qi], *[rr.gf(t)[2, qi] for t in roots], inf[2, qi]]
        ix = int(np.argmin(risks))
        opts.append(candidates[ix])
        optR.append(risks[ix])
        rootsall.append(roots)
    out = dict(
        s=s,
        times=times,
        gf=gf,
        initial=initial,
        infinity=inf,
        opt_times=np.array(opts),
        opt_risk=np.array(optR),
    )
    return rr, out, rootsall


__all__ = [
    "np",
    "torch",
    "json",
    "time",
    "gammaln",
    "brentq",
    "CASES",
    "Q",
    "TAU",
    "D",
    "Population",
    "FlowRisk",
    "flow",
    "sources",
    "edge_and_root",
]
