# Ficha de diagnóstico — S2-control-adam__rbf__s904__adam_final__e22f2623

**Problema:** heat_1d · **Campaña:** S2-control-adam · **Brazo:** rbf · **Semilla:** 904 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2/runs/tier2/seed-904/rbf/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0001289, u = 0.0001289
- Error normalizado (RMS, todas las componentes): 0.0001289; plano: 0.0001289 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4644; masa en el 5 % superior 0.2121; entropía normalizada 0.9281

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.0004189 | 0.008376 | 0.007426 | 0.1485 |
| t_q1 | 0.2481 | 0.02784 | 0.1122 | 0.03415 | 0.1377 |
| t_q2 | 0.2481 | 0.1705 | 0.6873 | 0.1202 | 0.4846 |
| t_q3 | 0.2481 | 0.515 | 2.076 | 0.2383 | 0.9606 |
| t_q4 | 0.2558 | 0.2866 | 1.12 | 0.6074 | 2.374 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0006676; p95 0.001179; máx 0.007286
- Concentración: Gini 0.5356; masa en el 1 % / 5 % superior 0.3538 / 0.6624; entropía normalizada 0.8182
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.01798; con |Δu| -0.3887

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.04705; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.005453; masa de residuo en el 5 % superior del error 0.01346
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.388 (por parámetro 0.002025); número de condición 9.266e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01606
- adam_loss_end: 6.066e-07
- adam_loss_start: 0.00636
- adam_n_logged: 200
- adam_rel_gain_last20pct: -5.912
- lbfgs_loss_end: 1.116e-08
- lbfgs_loss_start: 1.145e-08
- lbfgs_n_logged: 3
- lbfgs_to_adam_loss_ratio: 0.01839

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
