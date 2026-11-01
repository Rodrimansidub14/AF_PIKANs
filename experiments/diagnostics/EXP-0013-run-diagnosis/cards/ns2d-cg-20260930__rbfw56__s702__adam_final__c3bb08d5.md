# Ficha de diagnóstico — ns2d-cg-20260930__rbfw56__s702__adam_final__c3bb08d5

**Problema:** ns2d_cg · **Campaña:** ns2d-cg-20260930 · **Brazo:** rbf (ancho 56) · **Semilla:** 702 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/ns2d-cg-20260930/ns/rbf56-seed-702/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 40992 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.05719, p = 0.2337, u = 0.0403, v = 0.3708
- Error normalizado (RMS, todas las componentes): 0.4401; plano: 0.05719 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4908; masa en el 5 % superior 0.5815; entropía normalizada 0.837

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_corner | 0.171 | 0.2342 | 1.37 | 0.9834 | 5.75 |
| steep5 | 0.05004 | 0.1044 | 2.087 | 0.2795 | 5.587 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.03747; p95 0.02313; máx 0.6512
- Concentración: Gini 0.6562; masa en el 1 % / 5 % superior 0.8164 / 0.9715; entropía normalizada 0.5701
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.6706; con |Δu| 0.5889

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.2823; solape del 5 % superior 0.2047; masa de error en el 5 % superior del residuo 0.1418; masa de residuo en el 5 % superior del error 0.3174
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 28.96 (por parámetro 0.0007065); número de condición 1.848e+06
- Bases (capa 0): efectivas 10.79; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 4.116 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.756 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.1389
- adam_loss_end: 0.0001241
- adam_loss_start: 0.00411
- adam_n_logged: 200
- adam_rel_gain_last20pct: -0.1664
- adam_seconds: 2928
- adam_steps: 2e+04
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- (ninguno)
