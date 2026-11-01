# Ficha de diagnóstico — EXP-0011__rbfprad__s901__lbfgs_final__527291f6

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf+rad · **Semilla:** 901 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/rad/seed-901/checkpoint-lbfgs-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 9.606e-06, u = 9.606e-06
- Error normalizado (RMS, todas las componentes): 9.606e-06; plano: 9.606e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4317; masa en el 5 % superior 0.2378; entropía normalizada 0.933

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.005142 | 0.1028 | 0.07786 | 1.557 |
| t_q1 | 0.2481 | 0.1457 | 0.5875 | 0.2857 | 1.152 |
| t_q2 | 0.2481 | 0.2843 | 1.146 | 0.1792 | 0.7225 |
| t_q3 | 0.2481 | 0.2287 | 0.9219 | 0.1903 | 0.7671 |
| t_q4 | 0.2558 | 0.3413 | 1.334 | 0.3447 | 1.348 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0002119; p95 0.0003759; máx 0.005919
- Concentración: Gini 0.457; masa en el 1 % / 5 % superior 0.2981 / 0.4925; entropía normalizada 0.8557
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1062; con |Δu| -0.09575

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.07355; solape del 5 % superior 0.02474; masa de error en el 5 % superior del residuo 0.03913; masa de residuo en el 5 % superior del error 0.02937
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.318 (por parámetro 0.002006); número de condición 8.846e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.1033
- adam_loss_end: 8.135e-07
- adam_loss_start: 1.562e-05
- adam_n_logged: 40
- adam_rel_gain_last20pct: -16.36
- adam_seconds: 482.3
- adam_steps: 2e+04
- lbfgs_iterations: 1.262e+04
- lbfgs_seconds: 700.3
- refresh_seconds: 0.812
- meta_lbfgs_stop_reason: time_cap

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
