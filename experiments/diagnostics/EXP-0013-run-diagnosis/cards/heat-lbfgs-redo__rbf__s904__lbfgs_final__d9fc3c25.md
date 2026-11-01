# Ficha de diagnóstico — heat-lbfgs-redo__rbf__s904__lbfgs_final__d9fc3c25

**Problema:** heat_1d · **Campaña:** heat-lbfgs-redo · **Brazo:** rbf · **Semilla:** 904 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0008-baseline-v2-competitive/checkpoints/heat-lbfgs-redo-20260930/results/seed-904/checkpoint-lbfgs-redo.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 1.784e-06, u = 1.784e-06
- Error normalizado (RMS, todas las componentes): 1.784e-06; plano: 1.784e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4144; masa en el 5 % superior 0.2619; entropía normalizada 0.9333

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.06707 | 1.341 | 0.04256 | 0.8511 |
| t_q1 | 0.2481 | 0.2551 | 1.028 | 0.1304 | 0.5256 |
| t_q2 | 0.2481 | 0.264 | 1.064 | 0.08536 | 0.3441 |
| t_q3 | 0.2481 | 0.2572 | 1.037 | 0.1113 | 0.4488 |
| t_q4 | 0.2558 | 0.2237 | 0.8744 | 0.6729 | 2.631 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001899; p95 0.0002074; máx 0.01143
- Concentración: Gini 0.5145; masa en el 1 % / 5 % superior 0.7266 / 0.8354; entropía normalizada 0.5833
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1379; con |Δu| -0.2071

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.03912; solape del 5 % superior 0.05673; masa de error en el 5 % superior del residuo 0.03988; masa de residuo en el 5 % superior del error 0.01849
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.383 (por parámetro 0.002024); número de condición 9.271e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- lbfgs_iterations: 1.5e+04
- lbfgs_seconds: 580.2
- meta_stop_reason: max_iterations

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
