# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s801__lbfgs_final__d98fcf21

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 801 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-mlp-seed-801/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.8429, p = 0.9608, u = 0.8392, v = 1.017
- Error normalizado (RMS, todas las componentes): 1.632; plano: 0.8429 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3201; masa en el 5 % superior 0.3696; entropía normalizada 0.9089

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3269 | 4.853 | 0.175 | 2.597 |
| steep5 | 0.05009 | 0.2991 | 5.971 | 0.165 | 3.295 |
| wake | 0.15 | 0.1521 | 1.015 | 0.04969 | 0.3314 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.007511; p95 0.01518; máx 0.1264
- Concentración: Gini 0.5036; masa en el 1 % / 5 % superior 0.4062 / 0.6243; entropía normalizada 0.7644
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4024; con |Δu| 0.2081

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4164; solape del 5 % superior 0.5652; masa de error en el 5 % superior del residuo 0.2185; masa de residuo en el 5 % superior del error 0.5385
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 3.067 (por parámetro 0.0003856); número de condición 1.121e+17

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

- null_state, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
