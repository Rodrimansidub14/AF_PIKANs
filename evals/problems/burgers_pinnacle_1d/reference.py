"""Cole-Hopf reference for the canonical 1D PINN Burgers benchmark."""

from __future__ import annotations

import numpy as np

NU_DEFAULT = 0.01 / np.pi
SHOCK_FORMATION_TIME = 1.0 / np.pi
FRONT_HALF_WIDTH = 5.0 * NU_DEFAULT


def exact(
    x: np.ndarray,
    t: np.ndarray,
    nu: float = NU_DEFAULT,
    quadrature_order: int = 128,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Evaluate u=-2 nu d_x(log(phi)) using a stable Cole-Hopf convolution.

    For u(x,0)=-sin(pi*x), the Cole-Hopf potential is proportional to
    exp(-cos(pi*x)/(2*pi*nu)). The common factor exp(-1/(2*pi*nu)) cancels in
    phi_x/phi and is removed to keep the quadrature bounded. The initial field
    is returned analytically at t=0. Gauss-Hermite quadrature evaluates the
    Gaussian heat convolution in chunks to bound memory usage on the verifier.
    """
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be finite and positive")
    if quadrature_order < 16 or chunk_size < 1:
        raise ValueError("quadrature_order must be >=16 and chunk_size must be positive")
    x, t = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64))
    if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("x and t must be finite and t must be nonnegative")

    values = np.empty_like(x)
    initial = t == 0
    values[initial] = -np.sin(np.pi * x[initial])
    evolving_indices = np.flatnonzero(~initial.ravel())
    if evolving_indices.size == 0:
        return values

    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    kappa = 1.0 / (2.0 * np.pi * nu)
    flat_x, flat_t, flat_values = x.ravel(), t.ravel(), values.ravel()
    for start in range(0, evolving_indices.size, chunk_size):
        selected = evolving_indices[start : start + chunk_size]
        coordinate = flat_x[selected, None] + 2.0 * np.sqrt(
            nu * flat_t[selected, None]
        ) * nodes[None, :]
        potential = np.exp(-kappa * (np.cos(np.pi * coordinate) + 1.0))
        weighted = weights[None, :] * potential
        phi = weighted.sum(axis=1)
        phi_x = (
            weighted * (kappa * np.pi * np.sin(np.pi * coordinate))
        ).sum(axis=1)
        flat_values[selected] = -2.0 * nu * phi_x / phi

    values[np.isclose(np.abs(x), 1.0, rtol=0.0, atol=1e-14)] = 0.0
    return values


def residual(
    u: np.ndarray,
    u_t: np.ndarray,
    u_x: np.ndarray,
    u_xx: np.ndarray,
    nu: float = NU_DEFAULT,
) -> np.ndarray:
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be finite and positive")
    return np.asarray(u_t) + np.asarray(u) * np.asarray(u_x) - nu * np.asarray(u_xx)


def critical_mask(x: np.ndarray, t: np.ndarray, nu: float = NU_DEFAULT) -> np.ndarray:
    """Post-formation layer near x=0; width is five viscous length scales."""
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be finite and positive")
    return (
        (np.abs(np.asarray(x)) <= 5.0 * nu)
        & (np.asarray(t) >= SHOCK_FORMATION_TIME)
    )


def eval_grid(nx: int = 1025, nt: int = 129) -> tuple[np.ndarray, np.ndarray]:
    """Refined, deterministic space-time validation grid for the viscous front."""
    xs = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    xx, tt = np.meshgrid(xs, ts, indexing="ij")
    return xx.ravel(), tt.ravel()
