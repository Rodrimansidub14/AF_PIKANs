# Ficha de diagnóstico — burgers-f64-lbfgs-20261003__mlp__s303__lbfgs_final__b1f73bb6

**Problema:** burgers_1d · **Campaña:** burgers-f64-lbfgs-20261003 · **Brazo:** mlp · **Semilla:** 303 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-f64-lbfgs-20261003/mlp-seed-303/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0001651, u = 0.0001651
- Error normalizado (RMS, todas las componentes): 0.0001651; plano: 0.0001651 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4883; masa en el 5 % superior 0.7971; entropía normalizada 0.6282

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9048 | 1.337 | 0.9451 | 1.396 |
| steep5 | 0.05004 | 0.7501 | 14.99 | 0.9611 | 19.21 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.001818; p95 0.0008773; máx 0.08449
- Concentración: Gini 0.6189; masa en el 1 % / 5 % superior 0.953 / 0.9736; entropía normalizada 0.3874
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2232; con |Δu| 0.3041

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.05162; solape del 5 % superior 0.2275; masa de error en el 5 % superior del residuo 0.6993; masa de residuo en el 5 % superior del error 0.9217
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.248 (por parámetro 0.0002021); número de condición 5.629e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01846
- adam_loss_end: 0.001927
- adam_loss_start: 0.4199
- adam_n_logged: 200
- adam_rel_gain_last20pct: -8.419
- adam_seconds: 222.1
- adam_steps: 2e+04
- lbfgs_iterations: 6750
- lbfgs_seconds: 901.9
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
