# Ficha de diagnóstico — EXP-0012__rbfpradadamuniformlbfgs__s905__lbfgs_final__85e1c91c

**Problema:** heat_1d · **Campaña:** EXP-0012 · **Brazo:** rbf+rad(adam)->uniform-lbfgs · **Semilla:** 905 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/heat/EXP-0012-heat-rad-uniform-lbfgs/results/run-20261004/results/seed-905/checkpoint-lbfgs-uniform.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 3.49e-06, u = 3.49e-06
- Error normalizado (RMS, todas las componentes): 3.49e-06; plano: 3.49e-06 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4603; masa en el 5 % superior 0.329; entropía normalizada 0.9151

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.01438 | 0.2875 | 0.04798 | 0.9594 |
| t_q1 | 0.2481 | 0.127 | 0.512 | 0.2313 | 0.9323 |
| t_q2 | 0.2481 | 0.406 | 1.637 | 0.2475 | 0.9978 |
| t_q3 | 0.2481 | 0.3823 | 1.541 | 0.2553 | 1.029 |
| t_q4 | 0.2558 | 0.0847 | 0.3311 | 0.2659 | 1.04 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001447; p95 0.0002501; máx 0.001711
- Concentración: Gini 0.4351; masa en el 1 % / 5 % superior 0.2629 / 0.4382; entropía normalizada 0.881
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.003917; con |Δu| 0.02459

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.04896; solape del 5 % superior 0.01569; masa de error en el 5 % superior del residuo 0.02806; masa de residuo en el 5 % superior del error 0.02964
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.062 (por parámetro 0.001936); número de condición 8.513e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- lbfgs_iterations: 1.5e+04
- lbfgs_seconds: 621.3
- meta_stop_reason: max_iterations

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
