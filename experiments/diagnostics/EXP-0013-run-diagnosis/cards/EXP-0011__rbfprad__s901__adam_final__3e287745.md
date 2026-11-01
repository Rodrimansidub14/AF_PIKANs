# Ficha de diagnóstico — EXP-0011__rbfprad__s901__adam_final__3e287745

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf+rad · **Semilla:** 901 · **Etapa:** adam_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/rad/seed-901/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 6.226e-05, u = 6.226e-05
- Error normalizado (RMS, todas las componentes): 6.226e-05; plano: 6.226e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5579; masa en el 5 % superior 0.3567; entropía normalizada 0.8893

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.0003434 | 0.006867 | 0.02094 | 0.4187 |
| t_q1 | 0.2481 | 0.01799 | 0.07253 | 0.07168 | 0.289 |
| t_q2 | 0.2481 | 0.05103 | 0.2057 | 0.0989 | 0.3987 |
| t_q3 | 0.2481 | 0.2733 | 1.102 | 0.2642 | 1.065 |
| t_q4 | 0.2558 | 0.6577 | 2.571 | 0.5652 | 2.209 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0005796; p95 0.001193; máx 0.004774
- Concentración: Gini 0.5103; masa en el 1 % / 5 % superior 0.2432 / 0.5534; entropía normalizada 0.8573
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.1255; con |Δu| -0.3228

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1746; solape del 5 % superior 0.06639; masa de error en el 5 % superior del residuo 0.05592; masa de residuo en el 5 % superior del error 0.05739
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.341 (por parámetro 0.002012); número de condición 8.811e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.33 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
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

- residual_misaligned, plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
