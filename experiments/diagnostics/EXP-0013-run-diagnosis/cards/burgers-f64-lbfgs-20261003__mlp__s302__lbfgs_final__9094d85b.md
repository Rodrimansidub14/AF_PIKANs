# Ficha de diagnóstico — burgers-f64-lbfgs-20261003__mlp__s302__lbfgs_final__9094d85b

**Problema:** burgers_1d · **Campaña:** burgers-f64-lbfgs-20261003 · **Brazo:** mlp · **Semilla:** 302 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-f64-lbfgs-20261003/mlp-seed-302/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.00101, u = 0.00101
- Error normalizado (RMS, todas las componentes): 0.00101; plano: 0.00101 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.7668; masa en el 5 % superior 0.9962; entropía normalizada 0.4557

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9974 | 1.473 | 0.9577 | 1.415 |
| steep5 | 0.05004 | 0.9946 | 19.87 | 0.9904 | 19.79 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.00355; p95 0.0008694; máx 0.2572
- Concentración: Gini 0.6517; masa en el 1 % / 5 % superior 0.9888 / 0.994; entropía normalizada 0.2266
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2015; con |Δu| 0.242

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.07876; solape del 5 % superior 0.315; masa de error en el 5 % superior del residuo 0.5513; masa de residuo en el 5 % superior del error 0.9715
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.917 (por parámetro 0.0001941); número de condición 5.702e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01164
- adam_loss_end: 0.0001252
- adam_loss_start: 0.4179
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.9432
- adam_seconds: 218.4
- adam_steps: 2e+04
- lbfgs_iterations: 6900
- lbfgs_seconds: 900.8
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
