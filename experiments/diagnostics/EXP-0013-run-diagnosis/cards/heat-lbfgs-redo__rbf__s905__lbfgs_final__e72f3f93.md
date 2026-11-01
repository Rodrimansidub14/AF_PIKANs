# Ficha de diagnóstico — heat-lbfgs-redo__rbf__s905__lbfgs_final__e72f3f93

**Problema:** heat_1d · **Campaña:** heat-lbfgs-redo · **Brazo:** rbf · **Semilla:** 905 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0008-baseline-v2-competitive/checkpoints/heat-lbfgs-redo-20260930/results/seed-905/checkpoint-lbfgs-redo.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 2.107e-06, u = 2.107e-06
- Error normalizado (RMS, todas las componentes): 2.107e-06; plano: 2.107e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4388; masa en el 5 % superior 0.2817; entropía normalizada 0.9254

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.01749 | 0.3498 | 0.05315 | 1.063 |
| t_q1 | 0.2481 | 0.1684 | 0.6788 | 0.1543 | 0.6222 |
| t_q2 | 0.2481 | 0.3866 | 1.558 | 0.1071 | 0.4319 |
| t_q3 | 0.2481 | 0.3153 | 1.271 | 0.1287 | 0.5189 |
| t_q4 | 0.2558 | 0.1297 | 0.5071 | 0.6098 | 2.384 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001511; p95 0.0001932; máx 0.008701
- Concentración: Gini 0.4784; masa en el 1 % / 5 % superior 0.6169 / 0.741; entropía normalizada 0.6526
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.09906; con |Δu| -0.06564

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.04186; solape del 5 % superior 0.04164; masa de error en el 5 % superior del residuo 0.03931; masa de residuo en el 5 % superior del error 0.1857
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.371 (por parámetro 0.00202); número de condición 8.617e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- lbfgs_iterations: 1.5e+04
- lbfgs_seconds: 557.2
- meta_stop_reason: max_iterations

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
