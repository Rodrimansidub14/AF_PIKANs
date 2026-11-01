# Ficha de diagnóstico — EXP-0011__rbfprad__s904__adam_final__eedfa875

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf+rad · **Semilla:** 904 · **Etapa:** adam_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/rad/seed-904/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 9.097e-06, u = 9.097e-06
- Error normalizado (RMS, todas las componentes): 9.097e-06; plano: 9.097e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4468; masa en el 5 % superior 0.295; entropía normalizada 0.9215

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.01591 | 0.3182 | 0.04904 | 0.9806 |
| t_q1 | 0.2481 | 0.123 | 0.4957 | 0.1531 | 0.6174 |
| t_q2 | 0.2481 | 0.1894 | 0.7637 | 0.185 | 0.7459 |
| t_q3 | 0.2481 | 0.3359 | 1.354 | 0.2596 | 1.046 |
| t_q4 | 0.2558 | 0.3517 | 1.375 | 0.4023 | 1.572 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0003138; p95 0.0004755; máx 0.004675
- Concentración: Gini 0.4778; masa en el 1 % / 5 % superior 0.4806 / 0.6441; entropía normalizada 0.7907
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.0177; con |Δu| -0.2409

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.003018; solape del 5 % superior 0.05009; masa de error en el 5 % superior del residuo 0.04109; masa de residuo en el 5 % superior del error 0.02598
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 6.954 (por parámetro 0.001906); número de condición 9.444e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.0288
- adam_loss_end: 7.198e-08
- adam_loss_start: 1.543e-05
- adam_n_logged: 40
- adam_rel_gain_last20pct: -0.2568
- adam_seconds: 489.1
- adam_steps: 2e+04
- lbfgs_iterations: 1.5e+04
- lbfgs_seconds: 660.9
- refresh_seconds: 0.8263
- meta_lbfgs_stop_reason: max_iterations

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
