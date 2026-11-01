# Ficha de diagnóstico — piezo-daniel-recipe__mlp__s20260729__adam_final__dc95647f

**Problema:** piezo_2d · **Campaña:** piezo-daniel-recipe · **Brazo:** mlp · **Semilla:** 20260729 · **Etapa:** adam_final
**Checkpoint:** `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-seeds-20261004/results/mlp-seed-20260729/checkpoint-mlp-seed-20260729.pt` · **Familia de modelo:** piezo_daniel_upstream (mlp) · **Parámetros:** 5758 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0463, phi = 0.0463, u = 0.1275, v = 0.1777
- Error normalizado (RMS, todas las componentes): 0.2235; plano: 0.0463 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2815; masa en el 5 % superior 0.1125; entropía normalizada 0.9614

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| layer_bottom | 0.4762 | 0.4283 | 0.8994 | 0.4264 | 0.8954 |
| layer_top | 0.5238 | 0.5717 | 1.091 | 0.5736 | 1.095 |
| near_clamp | 0.0995 | 0.003332 | 0.03349 | 0.3514 | 3.531 |
| near_tip | 0.0995 | 0.1881 | 1.89 | 0.07114 | 0.7149 |
| steep5 | 0.05023 | 0.0959 | 1.909 | 0.03863 | 0.7691 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.5125; p95 0.9717; máx 1.353
- Concentración: Gini 0.3141; masa en el 1 % / 5 % superior 0.05691 / 0.2205; entropía normalizada 0.9379
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.4397; con |Δu| -0.5771

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.5417; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.0007459; masa de residuo en el 5 % superior del error 0.03739
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 1.742 (por parámetro 0.0003026); número de condición 7.822e+18

## 6. Señales de entrenamiento registradas

- seconds: 4079
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
