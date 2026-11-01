# Ficha de diagnóstico — EXP-0011__rbfctlreplicate__s901__adam_final__99ecb70b

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf(ctl-replicate) · **Semilla:** 901 · **Etapa:** adam_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/ctl/seed-901/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.000236, u = 0.000236
- Error normalizado (RMS, todas las componentes): 0.000236; plano: 0.000236 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.582; masa en el 5 % superior 0.3678; entropía normalizada 0.8865

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 1.925e-05 | 0.0003849 | 0.008071 | 0.1614 |
| t_q1 | 0.2481 | 0.005354 | 0.02159 | 0.04056 | 0.1635 |
| t_q2 | 0.2481 | 0.02084 | 0.08403 | 0.127 | 0.512 |
| t_q3 | 0.2481 | 0.1175 | 0.4736 | 0.3112 | 1.254 |
| t_q4 | 0.2558 | 0.8563 | 3.347 | 0.5213 | 2.038 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.000696; p95 0.001251; máx 0.007154
- Concentración: Gini 0.4394; masa en el 1 % / 5 % superior 0.1442 / 0.3226; entropía normalizada 0.9153
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.1771; con |Δu| -0.5579

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5295; solape del 5 % superior 0.2776; masa de error en el 5 % superior del residuo 0.13; masa de residuo en el 5 % superior del error 0.1124
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.648 (por parámetro 0.002096); número de condición 8.792e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.0211
- adam_loss_end: 3.449e-07
- adam_loss_start: 1.562e-05
- adam_n_logged: 40
- adam_rel_gain_last20pct: -5.655
- adam_seconds: 486.7
- adam_steps: 2e+04
- lbfgs_iterations: 0
- lbfgs_seconds: 6.538e-05
- refresh_seconds: 0
- meta_lbfgs_stop_reason: time_cap

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_aligned, plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
