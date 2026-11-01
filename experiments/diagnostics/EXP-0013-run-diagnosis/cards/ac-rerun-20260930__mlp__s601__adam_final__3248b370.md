# Ficha de diagnóstico — ac-rerun-20260930__mlp__s601__adam_final__3248b370

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** mlp · **Semilla:** 601 · **Etapa:** adam_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/mlp-seed-601/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 921 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1578, u = 0.1578
- Error normalizado (RMS, todas las componentes): 0.1578; plano: 0.1578 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.8103; masa en el 5 % superior 0.8366; entropía normalizada 0.7429

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.995 | 1.619 | 0.8801 | 1.432 |
| steep5 | 0.05005 | 0.01013 | 0.2024 | 0.1516 | 3.029 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005927; p95 0.01243; máx 0.04984
- Concentración: Gini 0.5153; masa en el 1 % / 5 % superior 0.2278 / 0.5576; entropía normalizada 0.8506
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.09382; con |Δu| 0.005719

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4178; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.006057; masa de residuo en el 5 % superior del error 0.02925
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 18.88 (por parámetro 0.0205); número de condición 5.582e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.003894
- adam_loss_end: 0.001107
- adam_loss_start: 0.5833
- adam_n_logged: 400
- adam_rel_gain_last20pct: 0.6912
- adam_seconds: 340.1
- adam_steps: 4e+04
- lbfgs_iterations: 1.32e+04
- lbfgs_seconds: 289.2
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
