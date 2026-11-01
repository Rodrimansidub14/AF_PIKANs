---
experiment_id: EXP-NNNN
slug: 
status: draft            # draft | preregistered | running | scored | decided
suite: tier1_regression
problem_id: 
model_id: 
mechanism_id: 
hypothesis_id: 
preregistered_at: 
preregistered_by: 
---

# EXP-NNNN — <título>

## Pregunta

<Una pregunta, no un tema.>

## Hipótesis

<Afirmación falsable. Qué mecanismo, sobre qué problema, con qué efecto.>

## Métrica primaria

| Campo | Valor |
|---|---|
| Métrica | `critical_region_residual_mean` |
| Dirección | minimize |
| Región | `boundary_neighborhood` |
| Baseline | `heat_1d@model.pikan.baseline` |
| **Delta que cuenta** | mejora relativa >= 20% |

## Métricas de guardia

Se miden para impedir que la primaria se gane rompiendo otra cosa.

| Métrica | Restricción |
|---|---|
| `relative_l2_error` | no empeorar más de 5% vs baseline |
| `stability_score` | >= 0.95 |
| `wall_time_sec` | <= 1.5x baseline |

## Condición de falsación

<Qué resultado concreto refuta la hipótesis. Si no puedes escribirlo, la
hipótesis no es falsable y el experimento no debe abrirse.>

## Presupuesto

| Campo | Valor |
|---|---|
| Suite / tier | tier1_regression / 1 |
| Backend | local_cpu |
| Semillas | [11, 17] |
| Corridas máximas | |
| Costo máximo | |

## Superficie de edición declarada

El agente puede modificar únicamente:

- [ ] `src/mechanisms/<mecanismo>.py`
- [ ] los campos de `focalization.bounded_edit_surface` del manifiesto

No puede tocar `evals/`, el modelo base, ni la definición de la PDE.

## Amenazas de validez

- **Leakage entrenamiento/evaluación:** la malla de evaluación es held-out y
  fija; el mecanismo no la lee. Verificar en Tier 0.
- <otras>

## Firma

El pre-registro entra en vigor cuando el investigador cambia `status` a
`preregistered` y añade fecha. A partir de ese momento los umbrales no se
mueven.
