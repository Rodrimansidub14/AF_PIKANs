# Ficha de diagnóstico — piezo-daniel-recipe__rbfw16__s20260728__adam_final__7e8830f1

**Problema:** piezo_2d · **Campaña:** piezo-daniel-recipe · **Brazo:** rbf (ancho 16) · **Semilla:** 20260728 · **Etapa:** adam_final
**Checkpoint:** `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-recipe-20261003/results/rbf16-seed-20260728/checkpoint-rbf-seed-20260728.pt` · **Familia de modelo:** piezo_daniel_upstream (rbf) · **Parámetros:** 5376 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.04849, phi = 0.04849, u = 0.09028, v = 0.09407
- Error normalizado (RMS, todas las componentes): 0.1391; plano: 0.04849 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2222; masa en el 5 % superior 0.1111; entropía normalizada 0.9724

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| layer_bottom | 0.4762 | 0.3289 | 0.6906 | 0.4647 | 0.9759 |
| layer_top | 0.5238 | 0.6711 | 1.281 | 0.5353 | 1.022 |
| near_clamp | 0.0995 | 0.01655 | 0.1663 | 0.1895 | 1.904 |
| near_tip | 0.0995 | 0.167 | 1.678 | 0.1049 | 1.054 |
| steep5 | 0.05023 | 0.08546 | 1.702 | 0.05937 | 1.182 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.6214; p95 1.002; máx 1.863
- Concentración: Gini 0.1927; masa en el 1 % / 5 % superior 0.05056 / 0.1729; entropía normalizada 0.9692
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.01765; con |Δu| -0.2513

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.1221; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.02584; masa de residuo en el 5 % superior del error 0.04696
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 3.302 (por parámetro 0.0006143); número de condición 7.594e+17
- Bases (capa 0): efectivas 10.97; muertas 0.06818; aristas pequeñas 0
- ratio_struct_to_spacing_x = 9.414 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.105e+05 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- seconds: 9222
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
