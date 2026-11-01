# Ficha de diagnóstico — EXP-0011__rbfctlreplicate__s904__adam_final__90e7d6c0

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf(ctl-replicate) · **Semilla:** 904 · **Etapa:** adam_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/ctl/seed-904/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0002434, u = 0.0002434
- Error normalizado (RMS, todas las componentes): 0.0002434; plano: 0.0002434 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5396; masa en el 5 % superior 0.2686; entropía normalizada 0.909

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 4.051e-05 | 0.0008101 | 0.008173 | 0.1634 |
| t_q1 | 0.2481 | 0.001618 | 0.006522 | 0.03704 | 0.1493 |
| t_q2 | 0.2481 | 0.03331 | 0.1343 | 0.1476 | 0.5949 |
| t_q3 | 0.2481 | 0.2703 | 1.09 | 0.3276 | 1.321 |
| t_q4 | 0.2558 | 0.6948 | 2.716 | 0.4878 | 1.907 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0004753; p95 0.0009307; máx 0.01353
- Concentración: Gini 0.4788; masa en el 1 % / 5 % superior 0.2262 / 0.4084; entropía normalizada 0.8591
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.2324; con |Δu| -0.2419

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4589; solape del 5 % superior 0.2492; masa de error en el 5 % superior del residuo 0.1409; masa de residuo en el 5 % superior del error 0.1116
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.383 (por parámetro 0.002024); número de condición 9.264e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.1589
- adam_loss_end: 1.402e-07
- adam_loss_start: 1.543e-05
- adam_n_logged: 40
- adam_rel_gain_last20pct: 0.09063
- adam_seconds: 484.2
- adam_steps: 2e+04
- lbfgs_iterations: 0
- lbfgs_seconds: 6.166e-05
- refresh_seconds: 0
- meta_lbfgs_stop_reason: time_cap

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
