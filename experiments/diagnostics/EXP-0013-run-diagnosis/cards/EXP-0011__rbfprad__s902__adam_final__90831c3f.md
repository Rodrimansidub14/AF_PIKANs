# Ficha de diagnóstico — EXP-0011__rbfprad__s902__adam_final__90831c3f

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf+rad · **Semilla:** 902 · **Etapa:** adam_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/rad/seed-902/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 9.125e-06, u = 9.125e-06
- Error normalizado (RMS, todas las componentes): 9.125e-06; plano: 9.125e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4759; masa en el 5 % superior 0.3178; entropía normalizada 0.9137

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.01291 | 0.2582 | 0.04769 | 0.9535 |
| t_q1 | 0.2481 | 0.07073 | 0.2851 | 0.1591 | 0.6414 |
| t_q2 | 0.2481 | 0.2159 | 0.8704 | 0.1969 | 0.7936 |
| t_q3 | 0.2481 | 0.2693 | 1.086 | 0.2156 | 0.8692 |
| t_q4 | 0.2558 | 0.4441 | 1.736 | 0.4284 | 1.675 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0002051; p95 0.0003977; máx 0.002078
- Concentración: Gini 0.4511; masa en el 1 % / 5 % superior 0.2154 / 0.4382; entropía normalizada 0.8885
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.02135; con |Δu| -0.07781

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.02163; solape del 5 % superior 0.1919; masa de error en el 5 % superior del residuo 0.1063; masa de residuo en el 5 % superior del error 0.1569
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.934 (por parámetro 0.002175); número de condición 8.824e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.02472
- adam_loss_end: 3.907e-08
- adam_loss_start: 1.513e-05
- adam_n_logged: 40
- adam_rel_gain_last20pct: 0.09708
- adam_seconds: 487.5
- adam_steps: 2e+04
- lbfgs_iterations: 1.202e+04
- lbfgs_seconds: 700
- refresh_seconds: 0.8275
- meta_lbfgs_stop_reason: time_cap

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
