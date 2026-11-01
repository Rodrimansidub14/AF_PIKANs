# Ficha de diagnóstico — dfg-long-20261001__mlp__s801__adam_final__71061508

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** mlp · **Semilla:** 801 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-mlp-seed-801/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.6085, p = 0.8309, u = 0.6027, v = 0.7858
- Error normalizado (RMS, todas las componentes): 1.293; plano: 0.6085 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3295; masa en el 5 % superior 0.4505; entropía normalizada 0.8828

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3646 | 5.412 | 0.4487 | 6.661 |
| steep5 | 0.05009 | 0.3522 | 7.031 | 0.4384 | 8.752 |
| wake | 0.15 | 0.1131 | 0.7544 | 0.02356 | 0.1571 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01457; p95 0.03323; máx 0.1059
- Concentración: Gini 0.676; masa en el 1 % / 5 % superior 0.3032 / 0.7391; entropía normalizada 0.7596
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4953; con |Δu| 0.5233

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4064; solape del 5 % superior 0.587; masa de error en el 5 % superior del residuo 0.3303; masa de residuo en el 5 % superior del error 0.5827
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 6.624 (por parámetro 0.0008329); número de condición 4.068e+16

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.007506
- adam_loss_end: 0.001079
- adam_loss_start: 0.002945
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.11
- adam_seconds: 736.8
- adam_steps: 2e+04
- lbfgs_iterations: 1.155e+04
- lbfgs_log_slope_last10: -0.001044
- lbfgs_loss_end: 0.0006892
- lbfgs_loss_start: 0.001077
- lbfgs_n_logged: 462
- lbfgs_seconds: 600
- lbfgs_to_adam_loss_ratio: 0.6388
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
