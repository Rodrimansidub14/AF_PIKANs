# Ficha de diagnóstico — EXP-0012__rbfpradadamuniformlbfgs__s904__lbfgs_final__53cbcff9

**Problema:** heat_1d · **Campaña:** EXP-0012 · **Brazo:** rbf+rad(adam)->uniform-lbfgs · **Semilla:** 904 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0012-heat-rad-uniform-lbfgs/results/run-20261004/results/seed-904/checkpoint-lbfgs-uniform.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 4.139e-06, u = 4.139e-06
- Error normalizado (RMS, todas las componentes): 4.139e-06; plano: 4.139e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4417; masa en el 5 % superior 0.2864; entropía normalizada 0.9242

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.0211 | 0.422 | 0.05685 | 1.137 |
| t_q1 | 0.2481 | 0.1584 | 0.6385 | 0.1814 | 0.7313 |
| t_q2 | 0.2481 | 0.1973 | 0.7954 | 0.1774 | 0.7153 |
| t_q3 | 0.2481 | 0.4338 | 1.749 | 0.2483 | 1.001 |
| t_q4 | 0.2558 | 0.2105 | 0.823 | 0.3929 | 1.536 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0002443; p95 0.0003807; máx 0.006508
- Concentración: Gini 0.4656; masa en el 1 % / 5 % superior 0.4227 / 0.6028; entropía normalizada 0.8096
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.02699; con |Δu| -0.2413

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.1145; solape del 5 % superior 0.005432; masa de error en el 5 % superior del residuo 0.02084; masa de residuo en el 5 % superior del error 0.01928
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 6.945 (por parámetro 0.001904); número de condición 9.471e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- lbfgs_iterations: 1.5e+04
- lbfgs_seconds: 622.5
- meta_stop_reason: max_iterations

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
