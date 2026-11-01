# Ficha de diagnóstico — dfg-long-20261001__mlp__s805__lbfgs_final__f1c7bbdc

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** mlp · **Semilla:** 805 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-mlp-seed-805/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.4793, p = 0.6743, u = 0.4742, v = 0.6286
- Error normalizado (RMS, todas las componentes): 1.037; plano: 0.4793 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3438; masa en el 5 % superior 0.4842; entropía normalizada 0.8733

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3268 | 4.851 | 0.6799 | 10.09 |
| steep5 | 0.05009 | 0.3353 | 6.694 | 0.6771 | 13.52 |
| wake | 0.15 | 0.09728 | 0.6487 | 0.009239 | 0.06161 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0123; p95 0.0245; máx 0.1609
- Concentración: Gini 0.7142; masa en el 1 % / 5 % superior 0.5109 / 0.8377; entropía normalizada 0.6856
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.5575; con |Δu| 0.4497

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3239; solape del 5 % superior 0.5826; masa de error en el 5 % superior del residuo 0.3837; masa de residuo en el 5 % superior del error 0.6043
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 10.69 (por parámetro 0.001345); número de condición 2.941e+16

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

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
