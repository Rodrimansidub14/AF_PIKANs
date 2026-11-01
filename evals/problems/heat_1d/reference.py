"""Solución de referencia analítica para la ecuación del calor 1D.

u_t = alpha * u_xx,  x in [0,1],  t in [0,1]
u(x,0) = sin(pi x),  u(0,t) = u(1,t) = 0

Separación de variables da la solución cerrada
    u(x,t) = sin(pi x) * exp(-alpha * pi^2 * t)

Este módulo es parte del verificador: define la verdad contra la que se mide
un candidato. No debe importar nada de `src/mechanisms/`.
"""

from __future__ import annotations

import numpy as np

ALPHA_DEFAULT = 0.1


def exact(x: np.ndarray, t: np.ndarray, alpha: float = ALPHA_DEFAULT) -> np.ndarray:
    """Solución analítica evaluada en puntos (x, t) emparejados."""
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)


def residual(u_t: np.ndarray, u_xx: np.ndarray, alpha: float = ALPHA_DEFAULT) -> np.ndarray:
    """Residuo físico puntual. Cero para la solución exacta."""
    return u_t - alpha * u_xx


def eval_grid(nx: int = 512, nt: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Malla de evaluación densa, fija y held-out.

    Determinista por construcción: no acepta semilla y no depende de la
    política de muestreo bajo estudio. Esta es la defensa contra el leakage
    entre entrenamiento y evaluación.
    """
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    return xx.ravel(), tt.ravel()


def critical_mask(x: np.ndarray, t: np.ndarray, region: str) -> np.ndarray:
    """Máscara booleana de una región crítica declarada en problem.yaml."""
    if region == "boundary_neighborhood":
        return (x < 0.05) | (x > 0.95)
    if region == "early_transient":
        return t < 0.05
    raise ValueError(f"región crítica desconocida: {region!r}")
