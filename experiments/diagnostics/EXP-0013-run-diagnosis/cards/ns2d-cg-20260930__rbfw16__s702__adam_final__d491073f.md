# Ficha de diagnóstico — ns2d-cg-20260930__rbfw16__s702__adam_final__d491073f

**Problema:** ns2d_cg · **Campaña:** ns2d-cg-20260930 · **Brazo:** rbf (ancho 16) · **Semilla:** 702 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/ns2d-cg-20260930/ns/rbf16-seed-702/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.06409, p = 0.3172, u = 0.04013, v = 0.3617
- Error normalizado (RMS, todas las componentes): 0.4827; plano: 0.06409 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3572; masa en el 5 % superior 0.4571; entropía normalizada 0.8872

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_corner | 0.171 | 0.282 | 1.649 | 0.9281 | 5.426 |
| steep5 | 0.05004 | 0.1111 | 2.221 | 0.4069 | 8.132 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.03198; p95 0.03725; máx 0.4138
- Concentración: Gini 0.5731; masa en el 1 % / 5 % superior 0.6522 / 0.8891; entropía normalizada 0.6643
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.7704; con |Δu| 0.6233

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.2356; solape del 5 % superior 0.2558; masa de error en el 5 % superior del residuo 0.149; masa de residuo en el 5 % superior del error 0.4302
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 33.91 (por parámetro 0.00841); número de condición 7.651e+06
- Bases (capa 0): efectivas 10.79; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 4.389 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.826 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.06697
- adam_loss_end: 0.0001054
- adam_loss_start: 0.004549
- adam_n_logged: 200
- adam_rel_gain_last20pct: -0.6089
- adam_seconds: 1253
- adam_steps: 2e+04
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- plateau_adam

## 8. Límites y datos ausentes

- (ninguno)
