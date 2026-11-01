# Ficha de diagnóstico — heat-lbfgs-redo__rbf__s902__lbfgs_final__f4307c8f

**Problema:** heat_1d · **Campaña:** heat-lbfgs-redo · **Brazo:** rbf · **Semilla:** 902 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0008-baseline-v2-competitive/checkpoints/heat-lbfgs-redo-20260930/results/seed-902/checkpoint-lbfgs-redo.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 1.797e-06, u = 1.797e-06
- Error normalizado (RMS, todas las componentes): 1.797e-06; plano: 1.797e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4446; masa en el 5 % superior 0.3111; entropía normalizada 0.9196

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.02111 | 0.422 | 0.04165 | 0.8329 |
| t_q1 | 0.2481 | 0.1622 | 0.6539 | 0.1318 | 0.5313 |
| t_q2 | 0.2481 | 0.3189 | 1.286 | 0.1611 | 0.6494 |
| t_q3 | 0.2481 | 0.2626 | 1.059 | 0.1793 | 0.723 |
| t_q4 | 0.2558 | 0.2562 | 1.002 | 0.5278 | 2.063 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001661; p95 0.000237; máx 0.004229
- Concentración: Gini 0.5003; masa en el 1 % / 5 % superior 0.5472 / 0.723; entropía normalizada 0.7483
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.03573; con |Δu| -0.24

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.003843; solape del 5 % superior 0.09354; masa de error en el 5 % superior del residuo 0.06832; masa de residuo en el 5 % superior del error 0.1161
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.475 (por parámetro 0.002323); número de condición 8.806e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- lbfgs_iterations: 1.5e+04
- lbfgs_seconds: 539.1
- meta_stop_reason: max_iterations

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
