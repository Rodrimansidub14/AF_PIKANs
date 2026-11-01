# Ficha de diagnóstico — dfg-long-20261001__rbf__s805__adam_final__6f3d448b

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** rbf · **Semilla:** 805 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-rbf-seed-805/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.4915, p = 0.6753, u = 0.4868, v = 0.6254
- Error normalizado (RMS, todas las componentes): 1.041; plano: 0.4915 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3374; masa en el 5 % superior 0.4779; entropía normalizada 0.8758

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.334 | 4.958 | 0.5399 | 8.014 |
| steep5 | 0.05009 | 0.3381 | 6.75 | 0.4773 | 9.528 |
| wake | 0.15 | 0.09701 | 0.647 | 0.04258 | 0.284 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01674; p95 0.03905; máx 0.1825
- Concentración: Gini 0.6212; masa en el 1 % / 5 % superior 0.327 / 0.6383; entropía normalizada 0.7732
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4295; con |Δu| 0.5156

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3741; solape del 5 % superior 0.537; masa de error en el 5 % superior del residuo 0.3618; masa de residuo en el 5 % superior del error 0.5028
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 21.67 (por parámetro 0.005375); número de condición 1.019e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 1.501 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.735 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.00598
- adam_loss_end: 0.000892
- adam_loss_start: 0.002521
- adam_n_logged: 200
- adam_rel_gain_last20pct: -0.02027
- adam_seconds: 2489
- adam_steps: 2e+04
- lbfgs_iterations: 4050
- lbfgs_log_slope_last10: -0.005466
- lbfgs_loss_end: 0.0004831
- lbfgs_loss_start: 0.0008228
- lbfgs_n_logged: 162
- lbfgs_seconds: 602.8
- lbfgs_to_adam_loss_ratio: 0.5416
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, plateau_adam

## 8. Límites y datos ausentes

- (ninguno)
