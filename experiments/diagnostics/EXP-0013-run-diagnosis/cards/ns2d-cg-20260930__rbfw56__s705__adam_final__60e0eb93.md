# Ficha de diagnóstico — ns2d-cg-20260930__rbfw56__s705__adam_final__60e0eb93

**Problema:** ns2d_cg · **Campaña:** ns2d-cg-20260930 · **Brazo:** rbf (ancho 56) · **Semilla:** 705 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/ns2d-cg-20260930/ns/rbf56-seed-705/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 40992 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.06276, p = 0.2772, u = 0.04325, v = 0.3712
- Error normalizado (RMS, todas las componentes): 0.4654; plano: 0.06276 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2466; masa en el 5 % superior 0.4112; entropía normalizada 0.9128

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_corner | 0.171 | 0.2416 | 1.413 | 0.9678 | 5.659 |
| steep5 | 0.05004 | 0.09354 | 1.869 | 0.8725 | 17.44 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.08834; p95 0.03536; máx 1.898
- Concentración: Gini 0.5857; masa en el 1 % / 5 % superior 0.9087 / 0.9626; entropía normalizada 0.5165
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.8787; con |Δu| 0.5993

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.3481; solape del 5 % superior 0.2589; masa de error en el 5 % superior del residuo 0.1117; masa de residuo en el 5 % superior del error 0.5983
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 31.53 (por parámetro 0.0007692); número de condición 2.151e+06
- Bases (capa 0): efectivas 10.79; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 3.535 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.686 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.05747
- adam_loss_end: 0.0005347
- adam_loss_start: 0.003949
- adam_n_logged: 200
- adam_rel_gain_last20pct: -4.19
- adam_seconds: 2928
- adam_steps: 2e+04
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- (ninguno)
