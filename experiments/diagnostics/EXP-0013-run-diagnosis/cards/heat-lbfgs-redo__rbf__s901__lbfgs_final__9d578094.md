# Ficha de diagnóstico — heat-lbfgs-redo__rbf__s901__lbfgs_final__9d578094

**Problema:** heat_1d · **Campaña:** heat-lbfgs-redo · **Brazo:** rbf · **Semilla:** 901 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0008-baseline-v2-competitive/checkpoints/heat-lbfgs-redo-20260930/results/seed-901/checkpoint-lbfgs-redo.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 1.945e-06, u = 1.945e-06
- Error normalizado (RMS, todas las componentes): 1.945e-06; plano: 1.945e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4326; masa en el 5 % superior 0.2976; entropía normalizada 0.9246

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.02487 | 0.4974 | 0.06727 | 1.345 |
| t_q1 | 0.2481 | 0.1442 | 0.5813 | 0.1751 | 0.706 |
| t_q2 | 0.2481 | 0.2996 | 1.208 | 0.1143 | 0.4609 |
| t_q3 | 0.2481 | 0.3076 | 1.24 | 0.1701 | 0.6859 |
| t_q4 | 0.2558 | 0.2486 | 0.972 | 0.5404 | 2.112 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001723; p95 0.0002398; máx 0.004092
- Concentración: Gini 0.5176; masa en el 1 % / 5 % superior 0.5805 / 0.7554; entropía normalizada 0.7314
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.07505; con |Δu| -0.2346

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.008849; solape del 5 % superior 0.07302; masa de error en el 5 % superior del residuo 0.055; masa de residuo en el 5 % superior del error 0.1042
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.638 (por parámetro 0.002094); número de condición 8.794e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- lbfgs_iterations: 1.26e+04
- lbfgs_seconds: 701.4
- meta_stop_reason: time_cap

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
