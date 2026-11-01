# Ficha de diagnóstico — burgers-f64-lbfgs-20261003__rbfw48__s301__lbfgs_final__6ff88ab4

**Problema:** burgers_1d · **Campaña:** burgers-f64-lbfgs-20261003 · **Brazo:** rbf (ancho 48) · **Semilla:** 301 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-f64-lbfgs-20261003/rbf48-seed-301/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 29376 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0001564, u = 0.0001564
- Error normalizado (RMS, todas las componentes): 0.0001564; plano: 0.0001564 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5332; masa en el 5 % superior 0.9107; entropía normalizada 0.4761

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9621 | 1.421 | 0.9703 | 1.433 |
| steep5 | 0.05004 | 0.8868 | 17.72 | 0.9688 | 19.36 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.002682; p95 0.001106; máx 0.1454
- Concentración: Gini 0.6571; masa en el 1 % / 5 % superior 0.964 / 0.985; entropía normalizada 0.3433
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1291; con |Δu| 0.2477

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.02214; solape del 5 % superior 0.2575; masa de error en el 5 % superior del residuo 0.8534; masa de residuo en el 5 % superior del error 0.9451
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 14.41 (por parámetro 0.0004904); número de condición 8.553e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.159 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06585 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.09009
- adam_loss_end: 0.00436
- adam_loss_start: 0.2583
- adam_n_logged: 200
- adam_rel_gain_last20pct: -29.04
- adam_seconds: 788.8
- adam_steps: 2e+04
- lbfgs_iterations: 5550
- lbfgs_seconds: 910.4
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
