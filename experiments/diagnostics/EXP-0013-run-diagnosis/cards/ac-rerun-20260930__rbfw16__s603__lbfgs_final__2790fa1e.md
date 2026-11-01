# Ficha de diagnóstico — ac-rerun-20260930__rbfw16__s603__lbfgs_final__2790fa1e

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** rbf (ancho 16) · **Semilla:** 603 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/rbf16-seed-603/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.001671, u = 0.001671
- Error normalizado (RMS, todas las componentes): 0.001671; plano: 0.001671 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.7888; masa en el 5 % superior 0.9699; entropía normalizada 0.6026

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.9992 | 1.626 | 0.6356 | 1.034 |
| steep5 | 0.05005 | 0.003943 | 0.07879 | 0.06746 | 1.348 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001253; p95 0.0002351; máx 0.001887
- Concentración: Gini 0.4707; masa en el 1 % / 5 % superior 0.2927 / 0.5145; entropía normalizada 0.8472
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1099; con |Δu| 0.1207

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1188; solape del 5 % superior 0.136; masa de error en el 5 % superior del residuo 0.3198; masa de residuo en el 5 % superior del error 0.1498
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 42.03 (por parámetro 0.01152); número de condición 5.323e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 10.38 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.241 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.1855
- adam_loss_end: 9.329e-06
- adam_loss_start: 0.09492
- adam_n_logged: 400
- adam_rel_gain_last20pct: 0.9456
- adam_seconds: 771.6
- adam_steps: 4e+04
- lbfgs_iterations: 2412
- lbfgs_seconds: 85.54
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
