# Ficha de diagnóstico — ns2d-cg-20260930__mlp__s704__adam_final__bc2ceed7

**Problema:** ns2d_cg · **Campaña:** ns2d-cg-20260930 · **Brazo:** mlp · **Semilla:** 704 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/ns2d-cg-20260930/ns/mlp-seed-704/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 41003 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1044, p = 0.4095, u = 0.0816, v = 0.5032
- Error normalizado (RMS, todas las componentes): 0.6539; plano: 0.1044 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3838; masa en el 5 % superior 0.3976; entropía normalizada 0.9051

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_corner | 0.171 | 0.2529 | 1.479 | 0.9315 | 5.447 |
| steep5 | 0.05004 | 0.0711 | 1.421 | 0.3415 | 6.826 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.03345; p95 0.03701; máx 0.3291
- Concentración: Gini 0.5763; masa en el 1 % / 5 % superior 0.5406 / 0.8852; entropía normalizada 0.6911
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1728; con |Δu| 0.07231

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1672; solape del 5 % superior 0.1271; masa de error en el 5 % superior del residuo 0.1159; masa de residuo en el 5 % superior del error 0.3135
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 14.17 (por parámetro 0.0003455); número de condición 5.683e+09

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.0158
- adam_loss_end: 0.0004614
- adam_loss_start: 0.1057
- adam_n_logged: 200
- adam_rel_gain_last20pct: -0.748
- adam_seconds: 697.5
- adam_steps: 2e+04
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
