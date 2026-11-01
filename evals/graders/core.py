"""Scorers exactos para candidatos SciML.

Invariante heredado de `context/research-ontology.toml`:

    "Cero, vacio y ausencia de medicion no son equivalentes."

Por eso un grader nunca devuelve 0.0 cuando no pudo medir. Devuelve
`value=None` con un `reason`, y el gate lo trata como no concluyente en vez de
como un exito perfecto. Esa distincion es la diferencia entre un verificador y
un generador de ruido optimista.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

MINIMIZE = "minimize"
MAXIMIZE = "maximize"


@dataclass(frozen=True)
class Score:
    """Resultado de un grader."""

    name: str
    value: float | None
    direction: str
    scope: str
    n: int
    reason: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _invalid(name: str, direction: str, scope: str, reason: str) -> Score:
    return Score(name=name, value=None, direction=direction, scope=scope, n=0, reason=reason)


def _clean(*arrays: np.ndarray) -> tuple[np.ndarray, ...] | None:
    """Descarta puntos no finitos de forma emparejada entre arreglos."""
    if not arrays:
        return None
    cleaned = tuple(np.asarray(a, dtype=float).ravel() for a in arrays)
    if len({a.size for a in cleaned}) != 1 or cleaned[0].size == 0:
        return None
    mask = np.ones(cleaned[0].size, dtype=bool)
    for a in cleaned:
        mask &= np.isfinite(a)
    if not mask.any():
        return None
    return tuple(a[mask] for a in cleaned)


# -- Exactitud contra la solucion de referencia ----------------------------

def relative_l2(prediction: np.ndarray, reference: np.ndarray, **_: Any) -> Score:
    """Error L2 relativo sobre la malla de evaluacion held-out."""
    name, scope = "relative_l2_error", "eval_grid"
    cleaned = _clean(prediction, reference)
    if cleaned is None:
        return _invalid(name, MINIMIZE, scope, "no_finite_points")
    pred, ref = cleaned
    denom = float(np.linalg.norm(ref))
    if denom == 0.0:
        return _invalid(name, MINIMIZE, scope, "reference_norm_zero")
    return Score(name, float(np.linalg.norm(pred - ref) / denom), MINIMIZE, scope, pred.size)


def max_abs_error(prediction: np.ndarray, reference: np.ndarray, **_: Any) -> Score:
    """Error absoluto maximo: detecta fallos locales que el L2 promedia."""
    name, scope = "max_abs_error", "eval_grid"
    cleaned = _clean(prediction, reference)
    if cleaned is None:
        return _invalid(name, MINIMIZE, scope, "no_finite_points")
    pred, ref = cleaned
    return Score(name, float(np.abs(pred - ref).max()), MINIMIZE, scope, pred.size)


# -- Consistencia fisica (no requiere solucion de referencia) --------------

def physics_residual_mean(residual: np.ndarray, **_: Any) -> Score:
    """Residuo medio de la PDE. El verificador no supervisado."""
    name, scope = "physics_residual_mean", "eval_grid"
    cleaned = _clean(residual)
    if cleaned is None:
        return _invalid(name, MINIMIZE, scope, "no_finite_points")
    (res,) = cleaned
    return Score(name, float(np.abs(res).mean()), MINIMIZE, scope, res.size)


def critical_region_residual_mean(
    residual: np.ndarray,
    critical_mask: np.ndarray,
    region: str = "unnamed",
    **_: Any,
) -> Score:
    """Residuo medio restringido a una region critica declarada.

    Existe porque el error global no sustituye una metrica local cuando la
    hipotesis habla de regiones criticas (invariante de la ontologia).
    """
    name, scope = "critical_region_residual_mean", f"region:{region}"
    res = np.asarray(residual, dtype=float).ravel()
    mask = np.asarray(critical_mask, dtype=bool).ravel()
    if res.size != mask.size:
        return _invalid(name, MINIMIZE, scope, "mask_shape_mismatch")
    if not mask.any():
        return _invalid(name, MINIMIZE, scope, "empty_region")
    cleaned = _clean(res[mask])
    if cleaned is None:
        return _invalid(name, MINIMIZE, scope, "no_finite_points")
    (sel,) = cleaned
    return Score(name, float(np.abs(sel).mean()), MINIMIZE, scope, sel.size)


def boundary_error_max(
    prediction: np.ndarray,
    reference: np.ndarray,
    boundary_mask: np.ndarray,
    **_: Any,
) -> Score:
    """Error maximo en la frontera: las BC son duras, no sugerencias."""
    name, scope = "boundary_error_max", "boundary"
    pred = np.asarray(prediction, dtype=float).ravel()
    ref = np.asarray(reference, dtype=float).ravel()
    mask = np.asarray(boundary_mask, dtype=bool).ravel()
    if not (pred.size == ref.size == mask.size):
        return _invalid(name, MINIMIZE, scope, "mask_shape_mismatch")
    if not mask.any():
        return _invalid(name, MINIMIZE, scope, "empty_boundary")
    cleaned = _clean(pred[mask], ref[mask])
    if cleaned is None:
        return _invalid(name, MINIMIZE, scope, "no_finite_points")
    p, r = cleaned
    return Score(name, float(np.abs(p - r).max()), MINIMIZE, scope, p.size)


# -- Salud del entrenamiento -----------------------------------------------

def stability_score(loss_history: np.ndarray, **_: Any) -> Score:
    """Salud del entrenamiento en [0, 1].

    Un candidato que alcanza buen error pero entrena de forma inestable no es
    un resultado utilizable: no se reproduce con otra semilla. Combina la
    fraccion de epocas finitas con cuanto se alejo el valor final del mejor
    visto (oscilacion o divergencia tardia).
    """
    name, scope = "stability_score", "training"
    hist = np.asarray(loss_history, dtype=float).ravel()
    if hist.size < 2:
        return _invalid(name, MAXIMIZE, scope, "history_too_short")
    finite = np.isfinite(hist)
    if not finite.any():
        return _invalid(name, MAXIMIZE, scope, "no_finite_points")
    finite_frac = float(finite.mean())
    vals = hist[finite]
    best = float(vals.min())
    final = float(vals[-1])
    if best <= 0.0:
        drift = 1.0
    elif final > 0.0:
        drift = min(1.0, best / final)
    else:
        drift = 0.0
    return Score(name, float(finite_frac * drift), MAXIMIZE, scope, hist.size)


def wall_time_sec(elapsed_sec: float | None = None, **_: Any) -> Score:
    """Costo de la corrida. Un mecanismo 2x mejor y 10x mas caro no sirve."""
    name, scope = "wall_time_sec", "run"
    if elapsed_sec is None or not math.isfinite(float(elapsed_sec)):
        return _invalid(name, MINIMIZE, scope, "not_measured")
    return Score(name, float(elapsed_sec), MINIMIZE, scope, 1)


GRADERS: dict[str, Callable[..., Score]] = {
    "relative_l2_error": relative_l2,
    "max_abs_error": max_abs_error,
    "physics_residual_mean": physics_residual_mean,
    "critical_region_residual_mean": critical_region_residual_mean,
    "boundary_error_max": boundary_error_max,
    "stability_score": stability_score,
    "wall_time_sec": wall_time_sec,
}


def grade_all(requested: list[str], **inputs: Any) -> dict[str, dict[str, Any]]:
    """Ejecuta los graders pedidos y devuelve un scorecard serializable.

    Un grader desconocido es un error del contrato, no un cero silencioso.
    """
    scorecard: dict[str, dict[str, Any]] = {}
    for metric in requested:
        grader = GRADERS.get(metric)
        if grader is None:
            scorecard[metric] = _invalid(metric, MINIMIZE, "unknown", "unknown_grader").as_dict()
            continue
        try:
            scorecard[metric] = grader(**inputs).as_dict()
        except TypeError as exc:
            scorecard[metric] = _invalid(
                metric, MINIMIZE, "unknown", f"missing_input:{exc}"
            ).as_dict()
    return scorecard
