# Ficha de diagnóstico — dfg-long-20261001__mlp__s802__adam_final__fc3793f4

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** mlp · **Semilla:** 802 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-mlp-seed-802/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.6697, p = 0.877, u = 0.6645, v = 0.8304
- Error normalizado (RMS, todas las componentes): 1.378; plano: 0.6697 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3184; masa en el 5 % superior 0.4125; entropía normalizada 0.8952

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3359 | 4.986 | 0.3987 | 5.918 |
| steep5 | 0.05009 | 0.3219 | 6.426 | 0.3966 | 7.917 |
| wake | 0.15 | 0.1295 | 0.8634 | 0.03371 | 0.2248 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01378; p95 0.03549; máx 0.07985
- Concentración: Gini 0.6028; masa en el 1 % / 5 % superior 0.2317 / 0.6905; entropía normalizada 0.785
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.39; con |Δu| 0.5112

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4948; solape del 5 % superior 0.4913; masa de error en el 5 % superior del residuo 0.2682; masa de residuo en el 5 % superior del error 0.4561
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 6.765 (por parámetro 0.0008506); número de condición 7.125e+16

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

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
