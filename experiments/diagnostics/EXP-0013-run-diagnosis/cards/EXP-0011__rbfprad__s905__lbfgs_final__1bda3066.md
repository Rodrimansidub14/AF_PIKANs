# Ficha de diagnóstico — EXP-0011__rbfprad__s905__lbfgs_final__1bda3066

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf+rad · **Semilla:** 905 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/rad/seed-905/checkpoint-lbfgs-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 7.214e-06, u = 7.214e-06
- Error normalizado (RMS, todas las componentes): 7.214e-06; plano: 7.214e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4682; masa en el 5 % superior 0.3556; entropía normalizada 0.9091

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.007069 | 0.1414 | 0.04369 | 0.8736 |
| t_q1 | 0.2481 | 0.0874 | 0.3523 | 0.2115 | 0.8527 |
| t_q2 | 0.2481 | 0.2229 | 0.8986 | 0.174 | 0.7016 |
| t_q3 | 0.2481 | 0.467 | 1.883 | 0.2636 | 1.063 |
| t_q4 | 0.2558 | 0.2227 | 0.8706 | 0.3508 | 1.371 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.000203; p95 0.000388; máx 0.002847
- Concentración: Gini 0.4274; masa en el 1 % / 5 % superior 0.1762 / 0.3784; entropía normalizada 0.9027
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.01433; con |Δu| -0.07297

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.02384; solape del 5 % superior 0.07725; masa de error en el 5 % superior del residuo 0.0587; masa de residuo en el 5 % superior del error 0.06556
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.099 (por parámetro 0.001946); número de condición 8.495e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.1475
- adam_loss_end: 3.715e-08
- adam_loss_start: 1.396e-05
- adam_n_logged: 40
- adam_rel_gain_last20pct: -0.4839
- adam_seconds: 484.5
- adam_steps: 2e+04
- lbfgs_iterations: 1.5e+04
- lbfgs_seconds: 663.5
- refresh_seconds: 0.8226
- meta_lbfgs_stop_reason: max_iterations

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
