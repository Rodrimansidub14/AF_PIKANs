# Ficha de diagnóstico — piezo-daniel-recipe__rbfw16__s20260731__adam_final__bb3d387a

**Problema:** piezo_2d · **Campaña:** piezo-daniel-recipe · **Brazo:** rbf (ancho 16) · **Semilla:** 20260731 · **Etapa:** adam_final
**Checkpoint:** `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-seeds2-20261004/results/rbf16-seed-20260731/checkpoint-rbf-seed-20260731.pt` · **Familia de modelo:** piezo_daniel_upstream (rbf) · **Parámetros:** 5376 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.04679, phi = 0.04679, u = 0.09482, v = 0.08913
- Error normalizado (RMS, todas las componentes): 0.1383; plano: 0.04679 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3595; masa en el 5 % superior 0.2335; entropía normalizada 0.9283

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| layer_bottom | 0.4762 | 0.5646 | 1.186 | 0.39 | 0.819 |
| layer_top | 0.5238 | 0.4354 | 0.8312 | 0.61 | 1.165 |
| near_clamp | 0.0995 | 0.01086 | 0.1092 | 0.2357 | 2.369 |
| near_tip | 0.0995 | 0.329 | 3.306 | 0.09601 | 0.9649 |
| steep5 | 0.05023 | 0.1811 | 3.605 | 0.05355 | 1.066 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.4377; p95 0.8539; máx 1.658
- Concentración: Gini 0.2772; masa en el 1 % / 5 % superior 0.07763 / 0.2478; entropía normalizada 0.9411
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.2148; con |Δu| -0.03271

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.3995; solape del 5 % superior 0.0237; masa de error en el 5 % superior del residuo 0.01607; masa de residuo en el 5 % superior del error 0.05151
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 3.594 (por parámetro 0.0006686); número de condición 7.664e+18
- Bases (capa 0): efectivas 10.97; muertas 0.06818; aristas pequeñas 0
- ratio_struct_to_spacing_x = 9.92 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.399e+05 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- seconds: 9211
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
