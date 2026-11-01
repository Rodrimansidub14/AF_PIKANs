# Ficha de diagnóstico — EXP-0011__rbfctlreplicate__s902__lbfgs_final__7af8e63a

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf(ctl-replicate) · **Semilla:** 902 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/ctl/seed-902/checkpoint-lbfgs-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0003082, u = 0.0003082
- Error normalizado (RMS, todas las componentes): 0.0003082; plano: 0.0003082 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4785; masa en el 5 % superior 0.3242; entropía normalizada 0.9154

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.0001574 | 0.003148 | 0.004351 | 0.08699 |
| t_q1 | 0.2481 | 0.03862 | 0.1557 | 0.05012 | 0.2021 |
| t_q2 | 0.2481 | 0.1714 | 0.6909 | 0.108 | 0.4352 |
| t_q3 | 0.2481 | 0.1492 | 0.6013 | 0.2648 | 1.067 |
| t_q4 | 0.2558 | 0.6408 | 2.505 | 0.5771 | 2.256 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.001159; p95 0.002472; máx 0.006671
- Concentración: Gini 0.4643; masa en el 1 % / 5 % superior 0.09511 / 0.3079; entropía normalizada 0.9129
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.2226; con |Δu| -0.3774

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1916; solape del 5 % superior 0.1038; masa de error en el 5 % superior del residuo 0.09002; masa de residuo en el 5 % superior del error 0.1129
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.47 (por parámetro 0.002322); número de condición 8.784e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.156
- adam_loss_end: 8.964e-07
- adam_loss_start: 1.513e-05
- adam_n_logged: 40
- adam_rel_gain_last20pct: -9.992
- adam_seconds: 488.9
- adam_steps: 2e+04
- lbfgs_iterations: 0
- lbfgs_seconds: 6.319e-05
- refresh_seconds: 0
- meta_lbfgs_stop_reason: time_cap

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
