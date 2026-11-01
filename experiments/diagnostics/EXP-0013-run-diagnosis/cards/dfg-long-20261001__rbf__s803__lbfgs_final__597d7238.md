# Ficha de diagnóstico — dfg-long-20261001__rbf__s803__lbfgs_final__597d7238

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** rbf · **Semilla:** 803 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-rbf-seed-803/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1689, p = 0.2988, u = 0.1654, v = 0.2398
- Error normalizado (RMS, todas las componentes): 0.4173; plano: 0.1689 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2859; masa en el 5 % superior 0.4391; entropía normalizada 0.8983

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2889 | 4.289 | 0.9451 | 14.03 |
| steep5 | 0.05009 | 0.2869 | 5.728 | 0.939 | 18.75 |
| wake | 0.15 | 0.1208 | 0.8057 | 0.006073 | 0.0405 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0379; p95 0.02194; máx 2.638
- Concentración: Gini 0.5715; masa en el 1 % / 5 % superior 0.9415 / 0.973; entropía normalizada 0.223
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2428; con |Δu| 0.4063

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.6009; solape del 5 % superior 0.413; masa de error en el 5 % superior del residuo 0.2957; masa de residuo en el 5 % superior del error 0.912
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 35.01 (por parámetro 0.008684); número de condición 3.807e+06
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.4606 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.03 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 7.277e-05
- adam_loss_end: 0.0007668
- adam_loss_start: 0.002698
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.03402
- adam_seconds: 2493
- adam_steps: 2e+04
- lbfgs_iterations: 4025
- lbfgs_log_slope_last10: -0.009257
- lbfgs_loss_end: 0.0002728
- lbfgs_loss_start: 0.0007588
- lbfgs_n_logged: 161
- lbfgs_seconds: 600.2
- lbfgs_to_adam_loss_ratio: 0.3557
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, residual_aligned

## 8. Límites y datos ausentes

- (ninguno)
