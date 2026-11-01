# Ficha de diagnóstico — burgers-f64-lbfgs-20261003__rbfw48__s305__lbfgs_final__9834b33e

**Problema:** burgers_1d · **Campaña:** burgers-f64-lbfgs-20261003 · **Brazo:** rbf (ancho 48) · **Semilla:** 305 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-f64-lbfgs-20261003/rbf48-seed-305/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 29376 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0002097, u = 0.0002097
- Error normalizado (RMS, todas las componentes): 0.0002097; plano: 0.0002097 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5606; masa en el 5 % superior 0.958; entropía normalizada 0.4163

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9791 | 1.446 | 0.9912 | 1.464 |
| steep5 | 0.05004 | 0.9458 | 18.9 | 0.9883 | 19.75 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.004418; p95 0.0009894; máx 0.2433
- Concentración: Gini 0.6962; masa en el 1 % / 5 % superior 0.9886 / 0.9951; entropía normalizada 0.2774
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1603; con |Δu| 0.271

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.05138; solape del 5 % superior 0.2407; masa de error en el 5 % superior del residuo 0.8779; masa de residuo en el 5 % superior del error 0.9849
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 15.31 (por parámetro 0.0005213); número de condición 6.879e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.166 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06581 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.09267
- adam_loss_end: 0.0005343
- adam_loss_start: 0.2543
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.3087
- adam_seconds: 792
- adam_steps: 2e+04
- lbfgs_iterations: 5050
- lbfgs_seconds: 907.6
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
