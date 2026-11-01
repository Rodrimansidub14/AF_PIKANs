# Ficha de diagnóstico — EXP-0011__rbfprad__s903__lbfgs_final__c4392a71

**Problema:** heat_1d · **Campaña:** EXP-0011 · **Brazo:** rbf+rad · **Semilla:** 903 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004/results/rad/seed-903/checkpoint-lbfgs-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 1.121e-05, u = 1.121e-05
- Error normalizado (RMS, todas las componentes): 1.121e-05; plano: 1.121e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5106; masa en el 5 % superior 0.3649; entropía normalizada 0.8992

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.004238 | 0.08474 | 0.04208 | 0.8413 |
| t_q1 | 0.2481 | 0.05675 | 0.2288 | 0.1659 | 0.6688 |
| t_q2 | 0.2481 | 0.1018 | 0.4103 | 0.1851 | 0.746 |
| t_q3 | 0.2481 | 0.2881 | 1.161 | 0.2344 | 0.9449 |
| t_q4 | 0.2558 | 0.5534 | 2.163 | 0.4146 | 1.621 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0002422; p95 0.0004536; máx 0.003021
- Concentración: Gini 0.4438; masa en el 1 % / 5 % superior 0.2287 / 0.4509; entropía normalizada 0.8858
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.007101; con |Δu| -0.1548

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.03277; solape del 5 % superior 0.07906; masa de error en el 5 % superior del residuo 0.05872; masa de residuo en el 5 % superior del error 0.081
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.256 (por parámetro 0.002263); número de condición 8.545e+300
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

- residual_misaligned

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
