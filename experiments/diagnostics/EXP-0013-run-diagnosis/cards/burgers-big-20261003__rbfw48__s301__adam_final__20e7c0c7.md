# Ficha de diagnóstico — burgers-big-20261003__rbfw48__s301__adam_final__20e7c0c7

**Problema:** burgers_1d · **Campaña:** burgers-big-20261003 · **Brazo:** rbf (ancho 48) · **Semilla:** 301 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-big-20261003/rbf48-seed-301/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 29376 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.005515, u = 0.005515
- Error normalizado (RMS, todas las componentes): 0.005515; plano: 0.005515 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5625; masa en el 5 % superior 0.3773; entropía normalizada 0.8776

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9394 | 1.388 | 0.9539 | 1.409 |
| steep5 | 0.05004 | 0.1318 | 2.633 | 0.8204 | 16.39 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0143; p95 0.01254; máx 0.2278
- Concentración: Gini 0.5994; masa en el 1 % / 5 % superior 0.7231 / 0.8763; entropía normalizada 0.655
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.03493; con |Δu| 0.2032

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5646; solape del 5 % superior 0.09341; masa de error en el 5 % superior del residuo 0.1242; masa de residuo en el 5 % superior del error 0.1632
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 17.17 (por parámetro 0.0005844); número de condición 6.221e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.162 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06636 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.0008127
- adam_loss_end: 0.0009604
- adam_loss_start: 0.2583
- adam_n_logged: 240
- adam_rel_gain_last20pct: -3.731
- adam_seconds: 1878
- adam_steps: 2.4e+04
- lbfgs_iterations: 5650
- lbfgs_seconds: 1805
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, residual_aligned, low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
