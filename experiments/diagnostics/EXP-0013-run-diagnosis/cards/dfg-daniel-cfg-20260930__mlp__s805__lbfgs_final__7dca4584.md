# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s805__lbfgs_final__7dca4584

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 805 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-mlp-seed-805/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.8945, p = 1.061, u = 0.8906, v = 1.017
- Error normalizado (RMS, todas las componentes): 1.719; plano: 0.8945 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2998; masa en el 5 % superior 0.3411; entropía normalizada 0.9197

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2995 | 4.446 | 0.2874 | 4.267 |
| steep5 | 0.05009 | 0.2737 | 5.464 | 0.2526 | 5.042 |
| wake | 0.15 | 0.1561 | 1.041 | 0.03272 | 0.2182 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005415; p95 0.01224; máx 0.03414
- Concentración: Gini 0.524; masa en el 1 % / 5 % superior 0.2904 / 0.7041; entropía normalizada 0.7837
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.5308; con |Δu| 0.4817

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4673; solape del 5 % superior 0.7283; masa de error en el 5 % superior del residuo 0.2637; masa de residuo en el 5 % superior del error 0.5723
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.231 (por parámetro 0.0002805); número de condición 1.575e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.003804
- adam_loss_end: 0.002644
- adam_loss_start: 0.003435
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.0003261
- adam_seconds: 74.97
- adam_steps: 2000
- lbfgs_loss_end: 0.001646
- lbfgs_loss_start: 0.002526
- lbfgs_n_logged: 10
- lbfgs_seconds: 48.75
- lbfgs_to_adam_loss_ratio: 0.6225
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
