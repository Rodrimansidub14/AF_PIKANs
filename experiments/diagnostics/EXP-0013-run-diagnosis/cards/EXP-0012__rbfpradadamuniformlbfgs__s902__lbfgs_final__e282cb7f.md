# Ficha de diagnóstico — EXP-0012__rbfpradadamuniformlbfgs__s902__lbfgs_final__e282cb7f

**Problema:** heat_1d · **Campaña:** EXP-0012 · **Brazo:** rbf+rad(adam)->uniform-lbfgs · **Semilla:** 902 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0012-heat-rad-uniform-lbfgs/results/run-20261004/results/seed-902/checkpoint-lbfgs-uniform.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 4.011e-06, u = 4.011e-06
- Error normalizado (RMS, todas las componentes): 4.011e-06; plano: 4.011e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4703; masa en el 5 % superior 0.3358; entropía normalizada 0.9122

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.01639 | 0.3276 | 0.05999 | 1.2 |
| t_q1 | 0.2481 | 0.1052 | 0.4242 | 0.1898 | 0.765 |
| t_q2 | 0.2481 | 0.4542 | 1.831 | 0.2248 | 0.9062 |
| t_q3 | 0.2481 | 0.2963 | 1.195 | 0.2387 | 0.9621 |
| t_q4 | 0.2558 | 0.1442 | 0.5638 | 0.3468 | 1.356 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001746; p95 0.0003056; máx 0.002003
- Concentración: Gini 0.4362; masa en el 1 % / 5 % superior 0.2516 / 0.4369; entropía normalizada 0.8829
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.0285; con |Δu| -0.00892

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.024; solape del 5 % superior 0.03199; masa de error en el 5 % superior del residuo 0.03265; masa de residuo en el 5 % superior del error 0.03254
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.923 (por parámetro 0.002172); número de condición 8.822e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- lbfgs_iterations: 1.252e+04
- lbfgs_seconds: 700.4
- meta_stop_reason: time_cap

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
