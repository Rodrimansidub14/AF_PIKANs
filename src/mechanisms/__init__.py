"""Mecanismos de focalización — la superficie que el agente puede escribir.

Este es el único paquete de código que el proposer puede modificar sin gate
humano. Todo lo demás (modelo base, definición de la PDE, `evals/`) requiere
aprobación explícita.

## Contrato

Un mecanismo decide **dónde mirar** durante el entrenamiento: qué puntos de
colocación muestrear y con qué peso entran a la loss. No decide qué es
correcto — eso lo fija `evals/`.

## La regla que hace esto seguro

Un mecanismo **no puede importar `evals`**. Si pudiera leer su propio grader o
la malla de evaluación, podría concentrar puntos exactamente donde se lo va a
medir y ganar la métrica sin resolver mejor la PDE. `tests/test_mechanism_isolation.py`
lo verifica de forma automática.
"""

from __future__ import annotations

from .base import Mechanism, MechanismOutput, TrainingState, get_mechanism, register
from .baseline_none import BaselineNone

__all__ = [
    "Mechanism",
    "MechanismOutput",
    "TrainingState",
    "BaselineNone",
    "get_mechanism",
    "register",
]
