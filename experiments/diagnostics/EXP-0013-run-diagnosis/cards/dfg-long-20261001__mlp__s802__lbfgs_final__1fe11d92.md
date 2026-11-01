# Ficha de diagnóstico — dfg-long-20261001__mlp__s802__lbfgs_final__1fe11d92

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** mlp · **Semilla:** 802 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-mlp-seed-802/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.4381, p = 0.6198, u = 0.4338, v = 0.5379
- Error normalizado (RMS, todas las componentes): 0.9283; plano: 0.4381 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3358; masa en el 5 % superior 0.4626; entropía normalizada 0.882

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3015 | 4.475 | 0.7021 | 10.42 |
| steep5 | 0.05009 | 0.3114 | 6.218 | 0.6833 | 13.64 |
| wake | 0.15 | 0.1006 | 0.6711 | 0.01695 | 0.1131 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01336; p95 0.02573; máx 0.273
- Concentración: Gini 0.7069; masa en el 1 % / 5 % superior 0.5616 / 0.8306; entropía normalizada 0.6557
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3178; con |Δu| 0.4104

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5619; solape del 5 % superior 0.4826; masa de error en el 5 % superior del residuo 0.3244; masa de residuo en el 5 % superior del error 0.507
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 10.32 (por parámetro 0.001297); número de condición 2.407e+16

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.0008413
- adam_loss_end: 0.001156
- adam_loss_start: 0.003065
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.1729
- adam_seconds: 740.7
- adam_steps: 2e+04
- lbfgs_iterations: 1.158e+04
- lbfgs_log_slope_last10: -0.0009876
- lbfgs_loss_end: 0.0006874
- lbfgs_loss_start: 0.001151
- lbfgs_n_logged: 463
- lbfgs_seconds: 600.7
- lbfgs_to_adam_loss_ratio: 0.5945
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, residual_aligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
