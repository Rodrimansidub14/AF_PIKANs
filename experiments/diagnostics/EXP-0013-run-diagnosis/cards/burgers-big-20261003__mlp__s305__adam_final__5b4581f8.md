# Ficha de diagnóstico — burgers-big-20261003__mlp__s305__adam_final__5b4581f8

**Problema:** burgers_1d · **Campaña:** burgers-big-20261003 · **Brazo:** mlp · **Semilla:** 305 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-big-20261003/mlp-seed-305/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.004642, u = 0.004642
- Error normalizado (RMS, todas las componentes): 0.004642; plano: 0.004642 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4708; masa en el 5 % superior 0.6071; entropía normalizada 0.7493

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9521 | 1.406 | 0.7564 | 1.117 |
| steep5 | 0.05004 | 0.5337 | 10.66 | 0.5825 | 11.64 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.004994; p95 0.006191; máx 0.06661
- Concentración: Gini 0.446; masa en el 1 % / 5 % superior 0.5749 / 0.7008; entropía normalizada 0.7462
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.02339; con |Δu| 0.3254

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2784; solape del 5 % superior 0.1569; masa de error en el 5 % superior del residuo 0.4065; masa de residuo en el 5 % superior del error 0.5063
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.887 (por parámetro 0.0002178); número de condición 5.165e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.03397
- adam_loss_end: 0.0001756
- adam_loss_start: 0.4175
- adam_n_logged: 240
- adam_rel_gain_last20pct: 0.9064
- adam_seconds: 489.7
- adam_steps: 2.4e+04
- lbfgs_iterations: 7550
- lbfgs_seconds: 1813
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
