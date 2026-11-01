# Ficha de diagnóstico — EXP-0011__rbfprad__s902__lbfgs_final__fd4b2524

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf+rad · **Semilla:** 902 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/rad/seed-902/checkpoint-lbfgs-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 8.722e-06, u = 8.722e-06
- Error normalizado (RMS, todas las componentes): 8.722e-06; plano: 8.722e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4654; masa en el 5 % superior 0.2992; entropía normalizada 0.9182

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.0152 | 0.3039 | 0.05434 | 1.087 |
| t_q1 | 0.2481 | 0.08482 | 0.3419 | 0.1797 | 0.7244 |
| t_q2 | 0.2481 | 0.2334 | 0.9408 | 0.2069 | 0.8341 |
| t_q3 | 0.2481 | 0.3072 | 1.239 | 0.2237 | 0.9019 |
| t_q4 | 0.2558 | 0.3746 | 1.464 | 0.3897 | 1.523 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001954; p95 0.0003795; máx 0.001741
- Concentración: Gini 0.4441; masa en el 1 % / 5 % superior 0.1878 / 0.4101; entropía normalizada 0.8976
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.01832; con |Δu| -0.06215

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.008935; solape del 5 % superior 0.1635; masa de error en el 5 % superior del residuo 0.09143; masa de residuo en el 5 % superior del error 0.1228
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.939 (por parámetro 0.002176); número de condición 8.824e+300
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
