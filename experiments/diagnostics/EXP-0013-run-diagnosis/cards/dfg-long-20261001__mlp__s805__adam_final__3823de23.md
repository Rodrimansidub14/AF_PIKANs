# Ficha de diagnóstico — dfg-long-20261001__mlp__s805__adam_final__3823de23

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** mlp · **Semilla:** 805 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-mlp-seed-805/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.8291, p = 0.9992, u = 0.8256, v = 0.9045
- Error normalizado (RMS, todas las componentes): 1.581; plano: 0.8291 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3116; masa en el 5 % superior 0.3592; entropía normalizada 0.9137

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2798 | 4.153 | 0.3519 | 5.224 |
| steep5 | 0.05009 | 0.266 | 5.31 | 0.3631 | 7.25 |
| wake | 0.15 | 0.1433 | 0.9559 | 0.008322 | 0.0555 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.009167; p95 0.023; máx 0.07532
- Concentración: Gini 0.7457; masa en el 1 % / 5 % superior 0.2796 / 0.7425; entropía normalizada 0.7439
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.68; con |Δu| 0.3162

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2441; solape del 5 % superior 0.6087; masa de error en el 5 % superior del residuo 0.294; masa de residuo en el 5 % superior del error 0.571
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 3.754 (por parámetro 0.000472); número de condición 1.54e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.001
- adam_loss_end: 0.001421
- adam_loss_start: 0.003435
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.03141
- adam_seconds: 745.3
- adam_steps: 2e+04
- lbfgs_iterations: 1.142e+04
- lbfgs_log_slope_last10: -0.0002708
- lbfgs_loss_end: 0.0008037
- lbfgs_loss_start: 0.00142
- lbfgs_n_logged: 457
- lbfgs_seconds: 600.2
- lbfgs_to_adam_loss_ratio: 0.5656
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
