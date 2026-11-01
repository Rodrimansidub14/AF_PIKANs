# Ficha de diagnóstico — EXP-0012__rbfpradadamuniformlbfgs__s903__lbfgs_final__a9643810

**Problema:** heat_1d · **Campaña:** EXP-0012 · **Brazo:** rbf+rad(adam)->uniform-lbfgs · **Semilla:** 903 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0012-heat-rad-uniform-lbfgs/results/run-20261004/results/seed-903/checkpoint-lbfgs-uniform.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 3.343e-06, u = 3.343e-06
- Error normalizado (RMS, todas las componentes): 3.343e-06; plano: 3.343e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4461; masa en el 5 % superior 0.2894; entropía normalizada 0.9234

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.01974 | 0.3946 | 0.06567 | 1.313 |
| t_q1 | 0.2481 | 0.1238 | 0.4989 | 0.2602 | 1.049 |
| t_q2 | 0.2481 | 0.2248 | 0.9064 | 0.173 | 0.6974 |
| t_q3 | 0.2481 | 0.4046 | 1.631 | 0.1753 | 0.7068 |
| t_q4 | 0.2558 | 0.2467 | 0.9646 | 0.3915 | 1.53 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001913; p95 0.0003128; máx 0.002816
- Concentración: Gini 0.4654; masa en el 1 % / 5 % superior 0.3594 / 0.5686; entropía normalizada 0.8376
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.03955; con |Δu| -0.1479

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.08583; solape del 5 % superior 0.02173; masa de error en el 5 % superior del residuo 0.02638; masa de residuo en el 5 % superior del error 0.0251
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.238 (por parámetro 0.002258); número de condición 8.598e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- lbfgs_iterations: 1.092e+04
- lbfgs_seconds: 700.5
- meta_stop_reason: time_cap

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
