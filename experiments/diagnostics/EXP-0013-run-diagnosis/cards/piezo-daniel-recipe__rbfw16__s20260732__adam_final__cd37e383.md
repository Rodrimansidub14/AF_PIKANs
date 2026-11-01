# Ficha de diagnóstico — piezo-daniel-recipe__rbfw16__s20260732__adam_final__cd37e383

**Problema:** piezo_2d · **Campaña:** piezo-daniel-recipe · **Brazo:** rbf (ancho 16) · **Semilla:** 20260732 · **Etapa:** adam_final
**Checkpoint:** `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-seeds2-20261004/results/rbf16-seed-20260732/checkpoint-rbf-seed-20260732.pt` · **Familia de modelo:** piezo_daniel_upstream (rbf) · **Parámetros:** 5376 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.05209, phi = 0.05209, u = 0.08357, v = 0.1236
- Error normalizado (RMS, todas las componentes): 0.1581; plano: 0.05209 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.201; masa en el 5 % superior 0.09765; entropía normalizada 0.9769

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| layer_bottom | 0.4762 | 0.4534 | 0.9522 | 0.4233 | 0.8889 |
| layer_top | 0.5238 | 0.5466 | 1.043 | 0.5767 | 1.101 |
| near_clamp | 0.0995 | 0.01651 | 0.1659 | 0.2701 | 2.715 |
| near_tip | 0.0995 | 0.1757 | 1.765 | 0.09209 | 0.9255 |
| steep5 | 0.05023 | 0.08987 | 1.789 | 0.05622 | 1.119 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.4786; p95 0.881; máx 1.596
- Concentración: Gini 0.2893; masa en el 1 % / 5 % superior 0.0618 / 0.2268; entropía normalizada 0.9424
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.3025; con |Δu| -0.1195

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.4894; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.02062; masa de residuo en el 5 % superior del error 0.03676
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 3.419 (por parámetro 0.000636); número de condición 3.259e+18
- Bases (capa 0): efectivas 10.97; muertas 0.06818; aristas pequeñas 0
- ratio_struct_to_spacing_x = 9.221 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.187e+05 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- seconds: 9295
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
