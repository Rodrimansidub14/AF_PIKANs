# Ficha de diagnóstico — burgers-f64-lbfgs-20261003__mlp__s305__lbfgs_final__7ae1910b

**Problema:** burgers_1d · **Campaña:** burgers-f64-lbfgs-20261003 · **Brazo:** mlp · **Semilla:** 305 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-f64-lbfgs-20261003/mlp-seed-305/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0003148, u = 0.0003148
- Error normalizado (RMS, todas las componentes): 0.0003148; plano: 0.0003148 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5412; masa en el 5 % superior 0.9098; entropía normalizada 0.5161

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9675 | 1.429 | 0.949 | 1.402 |
| steep5 | 0.05004 | 0.8854 | 17.69 | 0.9641 | 19.26 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.001925; p95 0.0008751; máx 0.09999
- Concentración: Gini 0.6035; masa en el 1 % / 5 % superior 0.9587 / 0.9758; entropía normalizada 0.3404
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2217; con |Δu| 0.2823

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.05158; solape del 5 % superior 0.194; masa de error en el 5 % superior del residuo 0.7786; masa de residuo en el 5 % superior del error 0.9278
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.173 (por parámetro 0.0002003); número de condición 5.759e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.2396
- adam_loss_end: 0.002006
- adam_loss_start: 0.4175
- adam_n_logged: 200
- adam_rel_gain_last20pct: -11.89
- adam_seconds: 227
- adam_steps: 2e+04
- lbfgs_iterations: 6750
- lbfgs_seconds: 902
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
