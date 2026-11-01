# Ficha de diagnóstico — EXP-0011__rbfprad__s903__adam_final__b42b1179

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf+rad · **Semilla:** 903 · **Etapa:** adam_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/rad/seed-903/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 1.817e-05, u = 1.817e-05
- Error normalizado (RMS, todas las componentes): 1.817e-05; plano: 1.817e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.531; masa en el 5 % superior 0.4194; entropía normalizada 0.8878

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.00181 | 0.03619 | 0.03329 | 0.6656 |
| t_q1 | 0.2481 | 0.02385 | 0.09615 | 0.1323 | 0.5333 |
| t_q2 | 0.2481 | 0.07496 | 0.3022 | 0.1689 | 0.6809 |
| t_q3 | 0.2481 | 0.2591 | 1.045 | 0.2345 | 0.9454 |
| t_q4 | 0.2558 | 0.6421 | 2.51 | 0.4643 | 1.815 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0002756; p95 0.0005111; máx 0.003883
- Concentración: Gini 0.4534; masa en el 1 % / 5 % superior 0.2325 / 0.4496; entropía normalizada 0.882
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.03706; con |Δu| -0.1908

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1049; solape del 5 % superior 0.1153; masa de error en el 5 % superior del residuo 0.07721; masa de residuo en el 5 % superior del error 0.09734
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.25 (por parámetro 0.002261); número de condición 8.545e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.05255
- adam_loss_end: 8.493e-08
- adam_loss_start: 1.181e-05
- adam_n_logged: 40
- adam_rel_gain_last20pct: -0.1974
- adam_seconds: 485.2
- adam_steps: 2e+04
- lbfgs_iterations: 1.5e+04
- lbfgs_seconds: 533.6
- refresh_seconds: 0.8403
- meta_lbfgs_stop_reason: max_iterations

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
