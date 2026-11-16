"""Shared numerical imports and the paper's finite-population defaults."""

import json
import time
import numpy as np
from scipy.special import gammaln
from scipy.optimize import brentq
import torch

torch.backends.cuda.matmul.allow_tf32 = False
D = 2**24
CASES = (("power", 0.2), ("critical", 0.2), ("power", 0.05), ("critical", 0.05))
Q = np.array([16.0, 24.0])
TAU2 = 1e-4

__all__ = [
    "np",
    "torch",
    "json",
    "time",
    "gammaln",
    "brentq",
    "CASES",
    "Q",
    "TAU2",
    "D",
]
