"""Numerically stable Cole-Hopf reference for the canonical PINN Burgers case.

This is a draft reference artifact outside evals/. It must not be treated as the
protected verifier until an explicit eval_integrity gate promotes the contract.
"""

from __future__ import annotations

import numpy as np


def exact(
    x: np.ndarray,
    t: np.ndarray,
    nu: float = 0.01 / np.pi,
    quadrature_order: int = 128,
) -> np.ndarray:
    """Return u=-2 nu d_x(log(phi)) from the Cole-Hopf heat-potential formula.

    The initial field is u(x,0)=-sin(pi*x), so a positive periodic Cole-Hopf
    potential is proportional to exp(-cos(pi*x)/(2*pi*nu)).  A common factor
    exp(-1/(2*pi*nu)) is removed to avoid overflow; it cancels in phi_x/phi.
    The whole-line Gaussian convolution is evaluated with Gauss-Hermite
    quadrature. The periodic extension induces zero velocity at x=+-1.
    """
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be finite and positive")
    if quadrature_order < 16:
        raise ValueError("quadrature_order must be at least 16")
    x, t = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64))
    if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("x and t must be finite and t must be nonnegative")
    initial = t == 0
    values = np.empty_like(x)
    values[initial] = -np.sin(np.pi * x[initial])
    evolving = ~initial
    if not np.any(evolving):
        return values

    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    kappa = 1.0 / (2.0 * np.pi * nu)
    xq = x[evolving, None] + 2.0 * np.sqrt(nu * t[evolving, None]) * nodes[None, :]
    potential = np.exp(-kappa * (np.cos(np.pi * xq) + 1.0))
    weighted = weights[None, :] * potential
    phi = weighted.sum(axis=1)
    phi_x = (weighted * (kappa * np.pi * np.sin(np.pi * xq))).sum(axis=1)
    values[evolving] = -2.0 * nu * phi_x / phi
    values[np.isclose(np.abs(x), 1.0, rtol=0.0, atol=1e-14)] = 0.0
    return values
