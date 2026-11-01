# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s801__adam_final__6584fc21

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 801 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-mlp-seed-801/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.8977, p = 0.9435, u = 0.8954, v = 1.06
- Error normalizado (RMS, todas las componentes): 1.678; plano: 0.8977 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2822; masa en el 5 % superior 0.3557; entropía normalizada 0.9185

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3343 | 4.962 | 0.06109 | 0.9069 |
| steep5 | 0.05009 | 0.3068 | 6.125 | 0.03993 | 0.7971 |
| wake | 0.15 | 0.1596 | 1.064 | 0.2548 | 1.699 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01636; p95 0.02438; máx 0.02703
- Concentración: Gini 0.2068; masa en el 1 % / 5 % superior 0.02614 / 0.1204; entropía normalizada 0.9752
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1083; con |Δu| 0.5225

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3348; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.02274; masa de residuo en el 5 % superior del error 0.04278
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.553 (por parámetro 0.000321); número de condición 1.36e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.004174
- adam_loss_end: 0.003011
- adam_loss_start: 0.002945
- adam_n_logged: 20
- adam_rel_gain_last20pct: -0.1776
- adam_seconds: 74.82
- adam_steps: 2000
- lbfgs_loss_end: 0.001559
- lbfgs_loss_start: 0.002419
- lbfgs_n_logged: 10
- lbfgs_seconds: 48.16
- lbfgs_to_adam_loss_ratio: 0.5178
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
