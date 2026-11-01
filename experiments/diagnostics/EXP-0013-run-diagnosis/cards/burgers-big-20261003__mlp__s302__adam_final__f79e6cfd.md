# Ficha de diagnóstico — burgers-big-20261003__mlp__s302__adam_final__f79e6cfd

**Problema:** burgers_1d · **Campaña:** burgers-big-20261003 · **Brazo:** mlp · **Semilla:** 302 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-big-20261003/mlp-seed-302/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.001856, u = 0.001856
- Error normalizado (RMS, todas las componentes): 0.001856; plano: 0.001856 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5645; masa en el 5 % superior 0.8903; entropía normalizada 0.5698

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9789 | 1.446 | 0.7776 | 1.149 |
| steep5 | 0.05004 | 0.8221 | 16.43 | 0.6596 | 13.18 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.002932; p95 0.004035; máx 0.09005
- Concentración: Gini 0.5406; masa en el 1 % / 5 % superior 0.6153 / 0.7779; entropía normalizada 0.6844
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1177; con |Δu| 0.2402

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2027; solape del 5 % superior 0.3042; masa de error en el 5 % superior del residuo 0.7155; masa de residuo en el 5 % superior del error 0.5085
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.349 (por parámetro 0.0002046); número de condición 5.369e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.04733
- adam_loss_end: 2.27e-05
- adam_loss_start: 0.4179
- adam_n_logged: 240
- adam_rel_gain_last20pct: 0.7475
- adam_seconds: 489.6
- adam_steps: 2.4e+04
- lbfgs_iterations: 9050
- lbfgs_seconds: 1802
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
