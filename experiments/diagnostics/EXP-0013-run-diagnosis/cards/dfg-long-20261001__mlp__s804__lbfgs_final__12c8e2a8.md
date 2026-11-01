# Ficha de diagnóstico — dfg-long-20261001__mlp__s804__lbfgs_final__12c8e2a8

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** mlp · **Semilla:** 804 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-mlp-seed-804/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.4759, p = 0.6744, u = 0.4711, v = 0.596
- Error normalizado (RMS, todas las componentes): 1.016; plano: 0.4759 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3346; masa en el 5 % superior 0.4688; entropía normalizada 0.8804

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3093 | 4.591 | 0.6713 | 9.965 |
| steep5 | 0.05009 | 0.3192 | 6.372 | 0.6869 | 13.71 |
| wake | 0.15 | 0.1004 | 0.6693 | 0.01067 | 0.07115 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01258; p95 0.02342; máx 0.2054
- Concentración: Gini 0.7089; masa en el 1 % / 5 % superior 0.5761 / 0.844; entropía normalizada 0.6626
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4883; con |Δu| 0.5024

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4205; solape del 5 % superior 0.5326; masa de error en el 5 % superior del residuo 0.352; masa de residuo en el 5 % superior del error 0.598
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 9.837 (por parámetro 0.001237); número de condición 4.329e+16

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.003552
- adam_loss_end: 0.001158
- adam_loss_start: 0.002858
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.346
- adam_seconds: 739.5
- adam_steps: 2e+04
- lbfgs_iterations: 1.158e+04
- lbfgs_log_slope_last10: -0.0003693
- lbfgs_loss_end: 0.000755
- lbfgs_loss_start: 0.001155
- lbfgs_n_logged: 463
- lbfgs_seconds: 600.6
- lbfgs_to_adam_loss_ratio: 0.6522
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
