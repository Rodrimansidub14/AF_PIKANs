# Ficha de diagnóstico — dfg-long-20261001__mlp__s801__lbfgs_final__43c03fa4

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** mlp · **Semilla:** 801 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-mlp-seed-801/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.4634, p = 0.6538, u = 0.4591, v = 0.5534
- Error normalizado (RMS, todas las componentes): 0.9719; plano: 0.4634 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3291; masa en el 5 % superior 0.4615; entropía normalizada 0.8835

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3042 | 4.516 | 0.7226 | 10.73 |
| steep5 | 0.05009 | 0.3146 | 6.281 | 0.717 | 14.31 |
| wake | 0.15 | 0.1018 | 0.6786 | 0.01081 | 0.07208 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01423; p95 0.0247; máx 0.3615
- Concentración: Gini 0.7139; masa en el 1 % / 5 % superior 0.6276 / 0.8512; entropía normalizada 0.6228
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.5187; con |Δu| 0.5339

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4539; solape del 5 % superior 0.4565; masa de error en el 5 % superior del residuo 0.3321; masa de residuo en el 5 % superior del error 0.3983
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 12.49 (por parámetro 0.00157); número de condición 3.994e+16

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.007506
- adam_loss_end: 0.001079
- adam_loss_start: 0.002945
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.11
- adam_seconds: 736.8
- adam_steps: 2e+04
- lbfgs_iterations: 1.155e+04
- lbfgs_log_slope_last10: -0.001044
- lbfgs_loss_end: 0.0006892
- lbfgs_loss_start: 0.001077
- lbfgs_n_logged: 462
- lbfgs_seconds: 600
- lbfgs_to_adam_loss_ratio: 0.6388
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
