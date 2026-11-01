# Ficha de diagnóstico — burgers-close-20261003__mlp__s303__adam_final__6d8ef03f

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** mlp · **Semilla:** 303 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/mlp-seed-303/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.01418, u = 0.01418
- Error normalizado (RMS, todas las componentes): 0.01418; plano: 0.01418 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4302; masa en el 5 % superior 0.3286; entropía normalizada 0.8948

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.944 | 1.395 | 0.8626 | 1.274 |
| steep5 | 0.05004 | 0.1896 | 3.788 | 0.6337 | 12.66 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0176; p95 0.01922; máx 0.4634
- Concentración: Gini 0.4112; masa en el 1 % / 5 % superior 0.5902 / 0.6767; entropía normalizada 0.721
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.07026; con |Δu| 0.04479

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1607; solape del 5 % superior 0.1389; masa de error en el 5 % superior del residuo 0.1676; masa de residuo en el 5 % superior del error 0.5655
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.639 (por parámetro 0.0002117); número de condición 5.271e+301

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

- low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
