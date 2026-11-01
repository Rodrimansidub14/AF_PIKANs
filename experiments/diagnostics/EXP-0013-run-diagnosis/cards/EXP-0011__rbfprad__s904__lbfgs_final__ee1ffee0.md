# Ficha de diagnóstico — EXP-0011__rbfprad__s904__lbfgs_final__ee1ffee0

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf+rad · **Semilla:** 904 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/rad/seed-904/checkpoint-lbfgs-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 9.039e-06, u = 9.039e-06
- Error normalizado (RMS, todas las componentes): 9.039e-06; plano: 9.039e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4483; masa en el 5 % superior 0.3019; entropía normalizada 0.921

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.01507 | 0.3014 | 0.05246 | 1.049 |
| t_q1 | 0.2481 | 0.1169 | 0.4714 | 0.1686 | 0.6797 |
| t_q2 | 0.2481 | 0.1793 | 0.7226 | 0.1898 | 0.7651 |
| t_q3 | 0.2481 | 0.329 | 1.326 | 0.2579 | 1.04 |
| t_q4 | 0.2558 | 0.3748 | 1.465 | 0.3837 | 1.5 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0002964; p95 0.0004516; máx 0.004269
- Concentración: Gini 0.4712; masa en el 1 % / 5 % superior 0.4662 / 0.6295; entropía normalizada 0.7982
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.02366; con |Δu| -0.2208

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.001691; solape del 5 % superior 0.04768; masa de error en el 5 % superior del residuo 0.03983; masa de residuo en el 5 % superior del error 0.02572
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 6.954 (por parámetro 0.001906); número de condición 9.455e+300
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

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
