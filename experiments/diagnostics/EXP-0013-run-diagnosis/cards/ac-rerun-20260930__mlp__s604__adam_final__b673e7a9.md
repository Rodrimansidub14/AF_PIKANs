# Ficha de diagnóstico — ac-rerun-20260930__mlp__s604__adam_final__b673e7a9

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** mlp · **Semilla:** 604 · **Etapa:** adam_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/mlp-seed-604/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 921 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.3691, u = 0.3691
- Error normalizado (RMS, todas las componentes): 0.3691; plano: 0.3691 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.7427; masa en el 5 % superior 0.3578; entropía normalizada 0.852

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.6717 | 1.093 | 0.9554 | 1.555 |
| steep5 | 0.05005 | 0.1282 | 2.561 | 0.01174 | 0.2346 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01198; p95 0.02682; máx 0.08408
- Concentración: Gini 0.6658; masa en el 1 % / 5 % superior 0.3062 / 0.7491; entropía normalizada 0.7778
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.05437; con |Δu| -0.1361

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5285; solape del 5 % superior 0.0009852; masa de error en el 5 % superior del residuo 0.003717; masa de residuo en el 5 % superior del error 0.01163
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 15.55 (por parámetro 0.01689); número de condición 5.893e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.0005244
- adam_loss_end: 0.004717
- adam_loss_start: 0.5523
- adam_n_logged: 400
- adam_rel_gain_last20pct: 0.1148
- adam_seconds: 345.3
- adam_steps: 4e+04
- lbfgs_iterations: 1.119e+04
- lbfgs_seconds: 246.1
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_aligned

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
