# Ficha de diagnóstico — heat-lbfgs-redo__rbf__s903__lbfgs_final__cd6d77e9

**Problema:** heat_1d · **Campaña:** heat-lbfgs-redo · **Brazo:** rbf · **Semilla:** 903 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0008-baseline-v2-competitive/checkpoints/heat-lbfgs-redo-20260930/results/seed-903/checkpoint-lbfgs-redo.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 1.634e-06, u = 1.634e-06
- Error normalizado (RMS, todas las componentes): 1.634e-06; plano: 1.634e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4275; masa en el 5 % superior 0.2967; entropía normalizada 0.9255

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.02662 | 0.5323 | 0.06288 | 1.257 |
| t_q1 | 0.2481 | 0.1454 | 0.5862 | 0.1564 | 0.6306 |
| t_q2 | 0.2481 | 0.2705 | 1.09 | 0.1841 | 0.7422 |
| t_q3 | 0.2481 | 0.2034 | 0.8201 | 0.1832 | 0.7385 |
| t_q4 | 0.2558 | 0.3807 | 1.488 | 0.4763 | 1.862 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001549; p95 0.000236; máx 0.002208
- Concentración: Gini 0.521; masa en el 1 % / 5 % superior 0.5131 / 0.7246; entropía normalizada 0.7687
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.0968; con |Δu| -0.2821

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.02281; solape del 5 % superior 0.1086; masa de error en el 5 % superior del residuo 0.06972; masa de residuo en el 5 % superior del error 0.09353
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.831 (por parámetro 0.002421); número de condición 8.446e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.3 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- lbfgs_iterations: 1.5e+04
- lbfgs_seconds: 445.5
- meta_stop_reason: max_iterations

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
