# Ficha de diagnóstico — ns2d-cg-20260930__rbfw56__s704__adam_final__69b8271c

**Problema:** ns2d_cg · **Campaña:** ns2d-cg-20260930 · **Brazo:** rbf (ancho 56) · **Semilla:** 704 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/ns2d-cg-20260930/ns/rbf56-seed-704/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 40992 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.06859, p = 0.3521, u = 0.04061, v = 0.3938
- Error normalizado (RMS, todas las componentes): 0.5298; plano: 0.06859 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.327; masa en el 5 % superior 0.4631; entropía normalizada 0.8931

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_corner | 0.171 | 0.1905 | 1.114 | 0.9799 | 5.729 |
| steep5 | 0.05004 | 0.06218 | 1.243 | 0.5571 | 11.13 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0293; p95 0.01836; máx 0.4977
- Concentración: Gini 0.6716; masa en el 1 % / 5 % superior 0.836 / 0.9697; entropía normalizada 0.5727
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.6428; con |Δu| 0.5181

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.2898; solape del 5 % superior 0.09922; masa de error en el 5 % superior del residuo 0.09434; masa de residuo en el 5 % superior del error 0.09199
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 32.28 (por parámetro 0.0007874); número de condición 1.09e+06
- Bases (capa 0): efectivas 10.79; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 4.083 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.773 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.02605
- adam_loss_end: 4.349e-05
- adam_loss_start: 0.003625
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.1703
- adam_seconds: 2912
- adam_steps: 2e+04
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- (ninguno)
