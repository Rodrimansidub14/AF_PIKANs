# Ficha de diagnóstico — ns2d-cg-20260930__mlp__s702__adam_final__b37f6b62

**Problema:** ns2d_cg · **Campaña:** ns2d-cg-20260930 · **Brazo:** mlp · **Semilla:** 702 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/ns2d-cg-20260930/ns/mlp-seed-702/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 41003 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0941, p = 0.4227, u = 0.06675, v = 0.4829
- Error normalizado (RMS, todas las componentes): 0.6452; plano: 0.0941 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3714; masa en el 5 % superior 0.3898; entropía normalizada 0.9074

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_corner | 0.171 | 0.2747 | 1.606 | 0.9441 | 5.52 |
| steep5 | 0.05004 | 0.07968 | 1.592 | 0.4554 | 9.102 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.03157; p95 0.03915; máx 0.2916
- Concentración: Gini 0.5893; masa en el 1 % / 5 % superior 0.5441 / 0.9035; entropía normalizada 0.6852
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.00328; con |Δu| -0.03619

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3217; solape del 5 % superior 0.1488; masa de error en el 5 % superior del residuo 0.1297; masa de residuo en el 5 % superior del error 0.3258
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 15.38 (por parámetro 0.0003751); número de condición 2.264e+09

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01109
- adam_loss_end: 0.0001611
- adam_loss_start: 0.1197
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.7434
- adam_seconds: 685.9
- adam_steps: 2e+04
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
