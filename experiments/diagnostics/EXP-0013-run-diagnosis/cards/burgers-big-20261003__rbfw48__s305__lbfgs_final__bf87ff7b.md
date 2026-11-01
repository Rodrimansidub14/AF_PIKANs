# Ficha de diagnóstico — burgers-big-20261003__rbfw48__s305__lbfgs_final__bf87ff7b

**Problema:** burgers_1d · **Campaña:** burgers-big-20261003 · **Brazo:** rbf (ancho 48) · **Semilla:** 305 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-big-20261003/rbf48-seed-305/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 29376 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0001569, u = 0.0001569
- Error normalizado (RMS, todas las componentes): 0.0001569; plano: 0.0001569 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5647; masa en el 5 % superior 0.9361; entropía normalizada 0.5158

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9752 | 1.441 | 0.6898 | 1.019 |
| steep5 | 0.05004 | 0.9207 | 18.4 | 0.6248 | 12.49 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0007148; p95 0.000954; máx 0.0225
- Concentración: Gini 0.5758; masa en el 1 % / 5 % superior 0.671 / 0.836; entropía normalizada 0.6385
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1608; con |Δu| 0.2736

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.08001; solape del 5 % superior 0.2659; masa de error en el 5 % superior del residuo 0.7258; masa de residuo en el 5 % superior del error 0.5777
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 14.56 (por parámetro 0.0004958); número de condición 6.758e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.164 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.0658 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.06149
- adam_loss_end: 0.001087
- adam_loss_start: 0.2543
- adam_n_logged: 240
- adam_rel_gain_last20pct: -1.952
- adam_seconds: 1872
- adam_steps: 2.4e+04
- lbfgs_iterations: 5600
- lbfgs_seconds: 1811
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
