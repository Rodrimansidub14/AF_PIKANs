# Ficha de diagnóstico — ns2d-cg-20260930__mlp__s705__adam_final__97488f6e

**Problema:** ns2d_cg · **Campaña:** ns2d-cg-20260930 · **Brazo:** mlp · **Semilla:** 705 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/ns2d-cg-20260930/ns/mlp-seed-705/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 41003 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.08782, p = 0.3861, u = 0.05854, v = 0.5705
- Error normalizado (RMS, todas las componentes): 0.6913; plano: 0.08782 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2856; masa en el 5 % superior 0.3952; entropía normalizada 0.92

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_corner | 0.171 | 0.1949 | 1.14 | 0.7382 | 4.316 |
| steep5 | 0.05004 | 0.05592 | 1.118 | 0.2229 | 4.454 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.03312; p95 0.03915; máx 0.32
- Concentración: Gini 0.4294; masa en el 1 % / 5 % superior 0.4575 / 0.6978; entropía normalizada 0.7787
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.8033; con |Δu| 0.6136

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.2578; solape del 5 % superior 0.08217; masa de error en el 5 % superior del residuo 0.09098; masa de residuo en el 5 % superior del error 0.1066
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 16.57 (por parámetro 0.0004041); número de condición 3.408e+08

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.05542
- adam_loss_end: 0.0004101
- adam_loss_start: 0.1347
- adam_n_logged: 200
- adam_rel_gain_last20pct: -0.1724
- adam_seconds: 685.3
- adam_steps: 2e+04
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
