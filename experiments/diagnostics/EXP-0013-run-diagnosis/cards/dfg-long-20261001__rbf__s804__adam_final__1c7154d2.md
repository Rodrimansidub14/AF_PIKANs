# Ficha de diagnóstico — dfg-long-20261001__rbf__s804__adam_final__1c7154d2

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** rbf · **Semilla:** 804 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-rbf-seed-804/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.4544, p = 0.6573, u = 0.449, v = 0.6115
- Error normalizado (RMS, todas las componentes): 1.004; plano: 0.4544 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.345; masa en el 5 % superior 0.4928; entropía normalizada 0.8699

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3479 | 5.164 | 0.6414 | 9.521 |
| steep5 | 0.05009 | 0.3513 | 7.013 | 0.6226 | 12.43 |
| wake | 0.15 | 0.09691 | 0.6463 | 0.02152 | 0.1435 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01495; p95 0.02698; máx 0.2182
- Concentración: Gini 0.6414; masa en el 1 % / 5 % superior 0.5439 / 0.8032; entropía normalizada 0.6924
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4364; con |Δu| 0.5696

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4482; solape del 5 % superior 0.5587; masa de error en el 5 % superior del residuo 0.3823; masa de residuo en el 5 % superior del error 0.6404
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 21.32 (por parámetro 0.005288); número de condición 8.723e+06
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 1.696 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.794 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.00398
- adam_loss_end: 0.0008448
- adam_loss_start: 0.002627
- adam_n_logged: 200
- adam_rel_gain_last20pct: -0.02821
- adam_seconds: 2498
- adam_steps: 2e+04
- lbfgs_iterations: 4075
- lbfgs_log_slope_last10: -0.006342
- lbfgs_loss_end: 0.0003069
- lbfgs_loss_start: 0.0007752
- lbfgs_n_logged: 163
- lbfgs_seconds: 600.4
- lbfgs_to_adam_loss_ratio: 0.3633
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, plateau_adam

## 8. Límites y datos ausentes

- (ninguno)
