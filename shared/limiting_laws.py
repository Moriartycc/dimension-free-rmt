"""Analytic limiting bulk laws in the centered/scaled coordinate."""

import numpy as np

PI = np.pi
ALPHA = 0.2


def limit(law, theta):
    # Parametrize the boundary S = r exp(i theta) by Im z(S) = 0.
    if law == "power":
        p = 1 + ALPHA
        b = 1 - 1 / p
        c = PI / (p * np.sin(PI / p))
        r = (np.sin(theta) / (c * np.sin(b * theta))) ** p
        s = r * np.exp(1j * theta)
        x = (-1 / s + c * s ** (-b)).real
        primitive = np.log(s) - c * b * s ** (1 - b) / (1 - b)
        edge_s = (1 / (b * c)) ** p
        edge = (1 / b - 1) / edge_s
    else:
        r = np.sinc(theta / PI)
        s = r * np.exp(1j * theta)
        x = (-1 / s - np.log(s)).real
        primitive = np.log(s) - s
        edge = -1.0
    return x, s.imag / PI, primitive.imag / PI, edge
