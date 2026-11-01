# Ficha de diagnóstico — burgers-f64-lbfgs-20261003__mlp__s304__adam_final__c00aeea3

**Problema:** burgers_1d · **Campaña:** burgers-f64-lbfgs-20261003 · **Brazo:** mlp · **Semilla:** 304 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-f64-lbfgs-20261003/mlp-seed-304/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.01176, u = 0.01176
- Error normalizado (RMS, todas las componentes): 0.01176; plano: 0.01176 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4198; masa en el 5 % superior 0.5779; entropía normalizada 0.7692

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9606 | 1.419 | 0.8793 | 1.299 |
| steep5 | 0.05004 | 0.5058 | 10.11 | 0.7583 | 15.15 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0135; p95 0.01267; máx 0.4463
- Concentración: Gini 0.4142; masa en el 1 % / 5 % superior 0.7341 / 0.7967; entropía normalizada 0.6267
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1104; con |Δu| 0.1608

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1572; solape del 5 % superior 0.1629; masa de error en el 5 % superior del residuo 0.4966; masa de residuo en el 5 % superior del error 0.7171
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.264 (por parámetro 0.0002025); número de condición 5.348e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.02195
- adam_loss_end: 0.0005161
- adam_loss_start: 0.4185
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.9251
- adam_seconds: 231.8
- adam_steps: 2e+04
- lbfgs_iterations: 6700
- lbfgs_seconds: 900.9
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
