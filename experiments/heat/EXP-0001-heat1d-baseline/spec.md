---
experiment_id: EXP-0001
slug: heat1d-baseline
status: decided            # draft | preregistered | running | scored | decided
suite: tier1_regression
problem_id: problem.heat_1d
model_id: model.pikan.baseline
mechanism_id: mechanism.baseline_none
hypothesis_id: H-CANON-BASE-001
preregistered_at:
preregistered_by:
---

# EXP-0001 — Baseline reproducible de PIKAN sobre Heat 1D

## Naturaleza de este experimento

Este es el único experimento del proyecto que **no compara contra un
baseline**, porque su producto *es* el baseline. Su criterio de aceptación no
es "mejor que", es "reproducible y medible".

Hasta que este experimento cierre, ningún gate de Tier 1 puede tener umbrales
y ningún candidato puede pasar de forma no trivial. Es el desbloqueo del loop
completo.

## Pregunta

¿Puede un PIKAN sin mecanismo de focalización resolver la ecuación del calor
1D de forma estable y reproducible, con artefactos trazables y varianza entre
semillas suficientemente baja como para servir de vara de comparación?

## Hipótesis

Un PIKAN base, entrenado con muestreo uniforme fijo sobre `problem.heat_1d`,
converge a una solución cuyo error L2 relativo sobre la malla held-out es
estable entre semillas independientes, y produce el conjunto completo de
métricas que las suites posteriores requieren.

## Métrica primaria

| Campo | Valor |
|---|---|
| Métrica | `relative_l2_error` |
| Dirección | minimize |
| Alcance | `eval_grid` (512 x 256, held-out) |
| Baseline | **ninguno — este experimento lo crea** |
| **Criterio** | dispersión entre semillas: `std/mean <= 0.15` |

El valor absoluto del error no es el criterio. Un baseline malo pero estable
sigue siendo un baseline válido; un baseline excelente pero irreproducible no
sirve para nada.

## Métricas de guardia

| Métrica | Restricción |
|---|---|
| `stability_score` | >= 0.95 en ambas semillas |
| `boundary_error_max` | finito y medido (las BC son duras) |
| `physics_residual_mean` | finito y medido |
| `critical_region_residual_mean` | finito y medido en `boundary_neighborhood` |
| `wall_time_sec` | medido (fija el denominador de costo de Tier 2) |

Ninguna de estas tiene umbral de calidad todavía. La exigencia es que estén
**medidas**: una métrica ausente aquí es una métrica que no se podrá usar como
gate después.

## Condición de falsación

El baseline se rechaza si ocurre cualquiera de estas:

1. `std/mean` de `relative_l2_error` entre semillas > 0.15.
2. `stability_score` < 0.95 en alguna semilla.
3. Cualquier métrica requerida vuelve `value: null` (no medida).
4. El entrenamiento produce NaN en alguna semilla.

Si se rechaza, **no se ajusta el umbral**: se corrige el modelo o el protocolo
y se vuelve a pre-registrar.

## Presupuesto

| Campo | Valor |
|---|---|
| Suite / tier | `tier1_regression` / 1 |
| Backend | `local_cpu` |
| Semillas | `[11, 17]` |
| Corridas máximas | 6 (permite 2 reintentos por fallo de infraestructura) |
| Costo máximo | 900 s de wall time total |

## Superficie de edición declarada

El agente puede modificar únicamente:

- [x] `src/mechanisms/baseline_none.py`
- [x] `training.*` en `manifest.yaml` (épocas, optimizador, puntos de colocación)
- [ ] ~~`evals/`~~ — protegido
- [ ] ~~definición de la PDE~~ — requiere gate `code_surface`
- [ ] ~~modelo base~~ — requiere gate `code_surface`

## Amenazas de validez

- **Leakage entrenamiento/evaluación.** La malla de evaluación es determinista
  y se genera desde `problem.yaml`, no desde la política de muestreo. En este
  experimento el riesgo es bajo porque el muestreo es uniforme fijo, pero el
  camino de código que lo garantiza se valida aquí, antes de que exista un
  mecanismo adaptativo que sí podría explotarlo.
- **Baseline débil.** Un baseline mal entrenado hace que cualquier mecanismo
  posterior parezca bueno. Mitigación: el valor absoluto del error se reporta
  y se revisa contra la literatura antes de congelar.
- **Dos semillas son pocas** para estimar varianza. Es una concesión de costo
  deliberada en Tier 1; Tier 2 usa cinco.

## Firma

Pendiente. El investigador debe:

1. revisar que el criterio de dispersión (0.15) es defendible,
2. cambiar `status` a `preregistered`,
3. añadir fecha y nombre.

Después de eso los umbrales quedan fijos.
