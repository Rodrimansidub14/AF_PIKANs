# Ficha de diagnóstico — burgers-close-20261003__rbfw48__s303__adam_final__d1d89e1d

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** rbf (ancho 48) · **Semilla:** 303 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/rbf48-seed-303/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 29376 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.009757, u = 0.009757
- Error normalizado (RMS, todas las componentes): 0.009757; plano: 0.009757 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.7773; masa en el 5 % superior 0.9856; entropía normalizada 0.4721

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.987 | 1.458 | 0.9248 | 1.366 |
| steep5 | 0.05004 | 0.9751 | 19.48 | 0.8014 | 16.01 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.007936; p95 0.009437; máx 0.2945
- Concentración: Gini 0.665; masa en el 1 % / 5 % superior 0.7423 / 0.8933; entropía normalizada 0.6002
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.02086; con |Δu| 0.4474

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4926; solape del 5 % superior 0.4946; masa de error en el 5 % superior del residuo 0.8448; masa de residuo en el 5 % superior del error 0.7613
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 15.39 (por parámetro 0.0005241); número de condición 7.003e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.171 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06609 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.007401
- adam_loss_end: 9.245e-05
- adam_loss_start: 0.2488
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.8348
- adam_seconds: 806.9
- adam_steps: 2e+04
- lbfgs_iterations: 3110
- lbfgs_seconds: 179.3
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
