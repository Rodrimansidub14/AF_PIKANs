# Ficha de diagnóstico — EXP-0011__rbfctlreplicate__s905__adam_final__ff2f54dd

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf(ctl-replicate) · **Semilla:** 905 · **Etapa:** adam_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/ctl/seed-905/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0002387, u = 0.0002387
- Error normalizado (RMS, todas las componentes): 0.0002387; plano: 0.0002387 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4743; masa en el 5 % superior 0.2077; entropía normalizada 0.9292

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.0001062 | 0.002123 | 0.002376 | 0.04751 |
| t_q1 | 0.2481 | 0.03497 | 0.141 | 0.05551 | 0.2238 |
| t_q2 | 0.2481 | 0.309 | 1.246 | 0.1349 | 0.5437 |
| t_q3 | 0.2481 | 0.351 | 1.415 | 0.2401 | 0.9678 |
| t_q4 | 0.2558 | 0.305 | 1.192 | 0.5695 | 2.226 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0008478; p95 0.001703; máx 0.01014
- Concentración: Gini 0.4148; masa en el 1 % / 5 % superior 0.08265 / 0.2747; entropía normalizada 0.9256
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.2122; con |Δu| -0.28

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1855; solape del 5 % superior 0.04707; masa de error en el 5 % superior del residuo 0.06977; masa de residuo en el 5 % superior del error 0.04552
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.383 (por parámetro 0.002024); número de condición 8.609e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.2781
- adam_loss_end: 7.617e-07
- adam_loss_start: 1.396e-05
- adam_n_logged: 40
- adam_rel_gain_last20pct: -3.245
- adam_seconds: 482.4
- adam_steps: 2e+04
- lbfgs_iterations: 0
- lbfgs_seconds: 5.504e-05
- refresh_seconds: 0
- meta_lbfgs_stop_reason: time_cap

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
