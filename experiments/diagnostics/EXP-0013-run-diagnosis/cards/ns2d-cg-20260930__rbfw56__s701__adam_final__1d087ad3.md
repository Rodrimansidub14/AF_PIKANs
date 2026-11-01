# Ficha de diagnóstico — ns2d-cg-20260930__rbfw56__s701__adam_final__1d087ad3

**Problema:** ns2d_cg · **Campaña:** ns2d-cg-20260930 · **Brazo:** rbf (ancho 56) · **Semilla:** 701 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/ns2d-cg-20260930/ns/rbf56-seed-701/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 40992 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.07223, p = 0.3828, u = 0.04144, v = 0.3976
- Error normalizado (RMS, todas las componentes): 0.5534; plano: 0.07223 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3166; masa en el 5 % superior 0.4171; entropía normalizada 0.9071

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_corner | 0.171 | 0.2046 | 1.196 | 0.9306 | 5.441 |
| steep5 | 0.05004 | 0.06725 | 1.344 | 0.1434 | 2.867 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.03114; p95 0.02088; máx 0.566
- Concentración: Gini 0.5682; masa en el 1 % / 5 % superior 0.8345 / 0.9188; entropía normalizada 0.578
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.8365; con |Δu| 0.5251

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.4189; solape del 5 % superior 0.08682; masa de error en el 5 % superior del residuo 0.08478; masa de residuo en el 5 % superior del error 0.1339
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 33.06 (por parámetro 0.0008066); número de condición 1.191e+06
- Bases (capa 0): efectivas 10.79; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 3.664 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.801 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.02408
- adam_loss_end: 9.155e-05
- adam_loss_start: 0.003216
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.0627
- adam_seconds: 2911
- adam_steps: 2e+04
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- (ninguno)
