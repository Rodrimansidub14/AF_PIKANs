# Ficha de diagnóstico — burgers-big-20261003__rbfw48__s304__lbfgs_final__45bed284

**Problema:** burgers_1d · **Campaña:** burgers-big-20261003 · **Brazo:** rbf (ancho 48) · **Semilla:** 304 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-big-20261003/rbf48-seed-304/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 29376 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 7.07e-05, u = 7.07e-05
- Error normalizado (RMS, todas las componentes): 7.07e-05; plano: 7.07e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5078; masa en el 5 % superior 0.7212; entropía normalizada 0.7037

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.8486 | 1.254 | 0.5617 | 0.8298 |
| steep5 | 0.05004 | 0.6075 | 12.14 | 0.4149 | 8.29 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.000512; p95 0.000759; máx 0.01253
- Concentración: Gini 0.5371; masa en el 1 % / 5 % superior 0.5683 / 0.7551; entropía normalizada 0.7085
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1548; con |Δu| 0.2658

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.0438; solape del 5 % superior 0.1629; masa de error en el 5 % superior del residuo 0.3723; masa de residuo en el 5 % superior del error 0.3149
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 17.14 (por parámetro 0.0005836); número de condición 6.072e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.162 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.0658 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.0768
- adam_loss_end: 0.0001177
- adam_loss_start: 0.2524
- adam_n_logged: 240
- adam_rel_gain_last20pct: 0.7072
- adam_seconds: 1879
- adam_steps: 2.4e+04
- lbfgs_iterations: 5350
- lbfgs_seconds: 1814
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
