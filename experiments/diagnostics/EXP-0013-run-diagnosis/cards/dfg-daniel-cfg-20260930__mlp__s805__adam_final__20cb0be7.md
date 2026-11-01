# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s805__adam_final__20cb0be7

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 805 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-mlp-seed-805/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9724, p = 0.9567, u = 0.9721, v = 1.035
- Error normalizado (RMS, todas las componentes): 1.712; plano: 0.9724 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2798; masa en el 5 % superior 0.3477; entropía normalizada 0.922

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3263 | 4.843 | 0.2317 | 3.439 |
| steep5 | 0.05009 | 0.2994 | 5.977 | 0.2242 | 4.476 |
| wake | 0.15 | 0.1598 | 1.065 | 0.03722 | 0.2482 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.006525; p95 0.0166; máx 0.02412
- Concentración: Gini 0.4279; masa en el 1 % / 5 % superior 0.1139 / 0.4414; entropía normalizada 0.8697
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.5373; con |Δu| 0.3314

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1798; solape del 5 % superior 0.3043; masa de error en el 5 % superior del residuo 0.1315; masa de residuo en el 5 % superior del error 0.2571
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.313 (por parámetro 0.0002909); número de condición 1.654e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.003804
- adam_loss_end: 0.002644
- adam_loss_start: 0.003435
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.0003261
- adam_seconds: 74.97
- adam_steps: 2000
- lbfgs_loss_end: 0.001646
- lbfgs_loss_start: 0.002526
- lbfgs_n_logged: 10
- lbfgs_seconds: 48.75
- lbfgs_to_adam_loss_ratio: 0.6225
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
