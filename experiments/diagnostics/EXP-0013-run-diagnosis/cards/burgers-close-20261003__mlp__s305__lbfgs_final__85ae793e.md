# Ficha de diagnóstico — burgers-close-20261003__mlp__s305__lbfgs_final__85ae793e

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** mlp · **Semilla:** 305 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/mlp-seed-305/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0002421, u = 0.0002421
- Error normalizado (RMS, todas las componentes): 0.0002421; plano: 0.0002421 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4721; masa en el 5 % superior 0.6669; entropía normalizada 0.69

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9017 | 1.332 | 0.9288 | 1.372 |
| steep5 | 0.05004 | 0.5635 | 11.26 | 0.9351 | 18.69 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.001995; p95 0.001197; máx 0.08966
- Concentración: Gini 0.5602; masa en el 1 % / 5 % superior 0.9259 / 0.9526; entropía normalizada 0.3818
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1982; con |Δu| 0.2688

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.02936; solape del 5 % superior 0.1509; masa de error en el 5 % superior del residuo 0.5151; masa de residuo en el 5 % superior del error 0.8415
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.216 (por parámetro 0.0002014); número de condición 5.682e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.2396
- adam_loss_end: 0.002006
- adam_loss_start: 0.4175
- adam_n_logged: 200
- adam_rel_gain_last20pct: -11.89
- adam_seconds: 228.8
- adam_steps: 2e+04
- lbfgs_iterations: 2233
- lbfgs_seconds: 58.07
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
