"""Contrato de los mecanismos de focalización.

Un mecanismo es una política de muestreo: dado el estado del entrenamiento,
decide qué puntos de colocación usar en la siguiente ventana de épocas y con
qué pesos entran los términos de la loss.

Restricciones que el contrato impone al proposer:

1. Un mecanismo **no importa `evals`**. No conoce su grader ni la malla de
   evaluación.
2. Un mecanismo es **determinista dada una semilla**. Recibe un `Generator`
   explícito; no usa el RNG global.
3. Un mecanismo **respeta el dominio**. Muestrear fuera de el es un error, no
   una heurística.
4. Un mecanismo **declara su superficie de parámetros**. Lo que no está en
   `bounded_edit_surface` del manifiesto no se toca.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TrainingState:
    """Lo que un mecanismo puede saber del entrenamiento en curso.

    Deliberadamente pobre: no incluye métricas de evaluación ni la solución de
    referencia. Un mecanismo ve el residuo sobre sus propios puntos, que es
    información que un solver real también tendría.
    """

    epoch: int
    loss_history: tuple[float, ...] = ()
    last_collocation: np.ndarray | None = None
    last_residual: np.ndarray | None = None


@dataclass(frozen=True)
class MechanismOutput:
    """Lo que un mecanismo devuelve por ventana de épocas."""

    collocation: np.ndarray
    loss_weights: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Mechanism(Protocol):
    """Interfaz que todo mecanismo debe cumplir."""

    id: str
    family: str

    def propose(
        self,
        n_points: int,
        state: TrainingState,
        rng: np.random.Generator,
    ) -> MechanismOutput:
        """Devuelve los puntos de colocación para la siguiente ventana."""
        ...


_REGISTRY: dict[str, Callable[..., Mechanism]] = {}


def register(mechanism_id: str) -> Callable[[type], type]:
    """Registra un mecanismo para que el harness lo resuelva por id."""

    def decorator(cls: type) -> type:
        if mechanism_id in _REGISTRY:
            raise ValueError(f"mecanismo duplicado: {mechanism_id!r}")
        _REGISTRY[mechanism_id] = cls
        return cls

    return decorator


def get_mechanism(mechanism_id: str, **kwargs: object) -> Mechanism:
    """Resuelve un mecanismo por su id de ontología."""
    factory = _REGISTRY.get(mechanism_id)
    if factory is None:
        known = ", ".join(sorted(_REGISTRY)) or "ninguno"
        raise KeyError(f"mecanismo desconocido: {mechanism_id!r}. Registrados: {known}")
    return factory(**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


def validate_in_domain(points: np.ndarray, bounds: dict[str, tuple[float, float]]) -> None:
    """Falla ruidosamente si el mecanismo muestreó fuera del dominio.

    Un punto fuera del dominio no es un candidato creativo: es un bug que
    contaminaría la loss y haría incomparable la corrida.
    """
    if points.ndim != 2 or points.shape[1] != len(bounds):
        raise ValueError(
            f"se esperaban puntos de forma (n, {len(bounds)}), se recibio {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("el mecanismo produjo puntos no finitos")
    for axis, (name, (low, high)) in enumerate(bounds.items()):
        column = points[:, axis]
        if column.min() < low - 1e-12 or column.max() > high + 1e-12:
            raise ValueError(
                f"el mecanismo muestreo fuera del dominio en '{name}': "
                f"[{column.min():.6g}, {column.max():.6g}] no cabe en [{low}, {high}]"
            )
