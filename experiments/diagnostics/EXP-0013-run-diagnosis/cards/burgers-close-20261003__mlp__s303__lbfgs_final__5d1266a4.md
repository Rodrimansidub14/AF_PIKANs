# Ficha de diagnóstico — burgers-close-20261003__mlp__s303__lbfgs_final__5d1266a4

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** mlp · **Semilla:** 303 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/mlp-seed-303/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0003444, u = 0.0003444
- Error normalizado (RMS, todas las componentes): 0.0003444; plano: 0.0003444 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5548; masa en el 5 % superior 0.9354; entropía normalizada 0.5253

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9658 | 1.427 | 0.9429 | 1.393 |
| steep5 | 0.05004 | 0.9185 | 18.35 | 0.9601 | 19.19 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.002098; p95 0.001018; máx 0.09424
- Concentración: Gini 0.6152; masa en el 1 % / 5 % superior 0.9498 / 0.9724; entropía normalizada 0.3814
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2458; con |Δu| 0.3095

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.03714; solape del 5 % superior 0.285; masa de error en el 5 % superior del residuo 0.8524; masa de residuo en el 5 % superior del error 0.9247
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.271 (por parámetro 0.0002027); número de condición 5.594e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01846
- adam_loss_end: 0.001927
- adam_loss_start: 0.4199
- adam_n_logged: 200
- adam_rel_gain_last20pct: -8.419
- adam_seconds: 228.8
- adam_steps: 2e+04
- lbfgs_iterations: 2613
- lbfgs_seconds: 67.19
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
