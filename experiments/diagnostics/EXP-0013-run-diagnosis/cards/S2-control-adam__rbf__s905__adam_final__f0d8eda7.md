# Ficha de diagnóstico — S2-control-adam__rbf__s905__adam_final__f0d8eda7

**Problema:** heat_1d · **Campaña:** S2-control-adam · **Brazo:** rbf · **Semilla:** 905 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2/runs/tier2/seed-905/rbf/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.000517, u = 0.000517
- Error normalizado (RMS, todas las componentes): 0.000517; plano: 0.000517 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4799; masa en el 5 % superior 0.2061; entropía normalizada 0.926

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 8.064e-05 | 0.001612 | 0.003519 | 0.07036 |
| t_q1 | 0.2481 | 0.01024 | 0.04128 | 0.07338 | 0.2958 |
| t_q2 | 0.2481 | 0.1126 | 0.4539 | 0.2403 | 0.9686 |
| t_q3 | 0.2481 | 0.3884 | 1.566 | 0.3581 | 1.443 |
| t_q4 | 0.2558 | 0.4887 | 1.91 | 0.3283 | 1.283 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0008383; p95 0.001679; máx 0.007862
- Concentración: Gini 0.4276; masa en el 1 % / 5 % superior 0.07303 / 0.2583; entropía normalizada 0.9278
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.3059; con |Δu| 0.1013

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4488; solape del 5 % superior 0.1243; masa de error en el 5 % superior del residuo 0.1188; masa de residuo en el 5 % superior del error 0.1076
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.373 (por parámetro 0.002021); número de condición 8.615e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.03932
- adam_loss_end: 5.215e-07
- adam_loss_start: 0.009326
- adam_n_logged: 200
- adam_rel_gain_last20pct: -0.8093
- lbfgs_loss_end: 9.188e-09
- lbfgs_loss_start: 9.247e-09
- lbfgs_n_logged: 2
- lbfgs_to_adam_loss_ratio: 0.01762

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
