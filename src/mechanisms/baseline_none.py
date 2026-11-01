"""Mecanismo nulo: muestreo uniforme fijo, sin focalización.

Es el control del experimento. Existe para que "sin mecanismo" sea una opción
ejecutable por el mismo camino de código que cualquier candidato, y no una
rama distinta del harness. Si el baseline corriera por otro camino, cualquier
diferencia medida después podría venir del camino y no del mecanismo.
"""

from __future__ import annotations

import numpy as np

from .base import Mechanism, MechanismOutput, TrainingState, register, validate_in_domain


@register("mechanism.baseline_none")
class BaselineNone(Mechanism):
    """Muestreo uniforme i.i.d. sobre el dominio, constante en el tiempo."""

    id = "mechanism.baseline_none"
    family = "baseline_none"

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
        loss_weights: dict[str, float] | None = None,
        resample_every: int = 0,
    ) -> None:
        self.bounds = bounds or {"x": (0.0, 1.0), "t": (0.0, 1.0)}
        # Pesos fijos: el baseline no programa la loss.
        self.loss_weights = loss_weights or {"physics": 1.0, "boundary": 1.0, "initial": 1.0}
        # 0 = muestrear una vez y reutilizar. Es el caso "malla fija".
        self.resample_every = resample_every
        self._cached: np.ndarray | None = None

    def propose(
        self,
        n_points: int,
        state: TrainingState,
        rng: np.random.Generator,
    ) -> MechanismOutput:
        reuse = (
            self._cached is not None
            and self._cached.shape[0] == n_points
            and (self.resample_every == 0 or state.epoch % self.resample_every != 0)
        )
        if reuse:
            points = self._cached
        else:
            columns = [
                rng.uniform(low, high, size=n_points) for low, high in self.bounds.values()
            ]
            points = np.column_stack(columns)
            self._cached = points

        validate_in_domain(points, self.bounds)
        return MechanismOutput(
            collocation=points,
            loss_weights=dict(self.loss_weights),
            diagnostics={"n_points": float(n_points), "resampled": float(not reuse)},
        )
