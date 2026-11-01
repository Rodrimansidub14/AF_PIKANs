# Ficha de diagnóstico — burgers-big-20261003__mlp__s301__lbfgs_final__923e16e2

**Problema:** burgers_1d · **Campaña:** burgers-big-20261003 · **Brazo:** mlp · **Semilla:** 301 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-big-20261003/mlp-seed-301/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0001435, u = 0.0001435
- Error normalizado (RMS, todas las componentes): 0.0001435; plano: 0.0001435 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4631; masa en el 5 % superior 0.536; entropía normalizada 0.8039

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.84 | 1.241 | 0.6867 | 1.015 |
| steep5 | 0.05004 | 0.3966 | 7.925 | 0.6132 | 12.25 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0006527; p95 0.0009962; máx 0.02278
- Concentración: Gini 0.5062; masa en el 1 % / 5 % superior 0.5581 / 0.7164; entropía normalizada 0.7002
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1755; con |Δu| 0.255

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.05503; solape del 5 % superior 0.1257; masa de error en el 5 % superior del residuo 0.3235; masa de residuo en el 5 % superior del error 0.4053
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.382 (por parámetro 0.0002054); número de condición 5.542e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.1473
- adam_loss_end: 0.0009335
- adam_loss_start: 0.4189
- adam_n_logged: 240
- adam_rel_gain_last20pct: -1.342
- adam_seconds: 485.8
- adam_steps: 2.4e+04
- lbfgs_iterations: 7450
- lbfgs_seconds: 1811
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
