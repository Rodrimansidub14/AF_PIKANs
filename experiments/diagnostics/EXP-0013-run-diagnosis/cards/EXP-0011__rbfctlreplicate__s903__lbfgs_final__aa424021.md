# Ficha de diagnóstico — EXP-0011__rbfctlreplicate__s903__lbfgs_final__aa424021

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf(ctl-replicate) · **Semilla:** 903 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/ctl/seed-903/checkpoint-lbfgs-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 9.908e-05, u = 9.908e-05
- Error normalizado (RMS, todas las componentes): 9.908e-05; plano: 9.908e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5902; masa en el 5 % superior 0.2599; entropía normalizada 0.8994

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 4.5e-05 | 0.0008999 | 0.009772 | 0.1954 |
| t_q1 | 0.2481 | 0.0037 | 0.01491 | 0.0409 | 0.1649 |
| t_q2 | 0.2481 | 0.1072 | 0.432 | 0.2229 | 0.8985 |
| t_q3 | 0.2481 | 0.4609 | 1.858 | 0.2812 | 1.134 |
| t_q4 | 0.2558 | 0.4283 | 1.674 | 0.455 | 1.779 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0004083; p95 0.0008643; máx 0.003514
- Concentración: Gini 0.4872; masa en el 1 % / 5 % superior 0.1518 / 0.3753; entropía normalizada 0.8956
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.2376; con |Δu| -0.1604

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2619; solape del 5 % superior 0.1237; masa de error en el 5 % superior del residuo 0.08013; masa de residuo en el 5 % superior del error 0.09353
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.839 (por parámetro 0.002423); número de condición 8.434e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.3 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.1839
- adam_loss_end: 1.47e-07
- adam_loss_start: 1.181e-05
- adam_n_logged: 40
- adam_rel_gain_last20pct: 0.8647
- adam_seconds: 486.9
- adam_steps: 2e+04
- lbfgs_iterations: 0
- lbfgs_seconds: 5.723e-05
- refresh_seconds: 0
- meta_lbfgs_stop_reason: time_cap

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
