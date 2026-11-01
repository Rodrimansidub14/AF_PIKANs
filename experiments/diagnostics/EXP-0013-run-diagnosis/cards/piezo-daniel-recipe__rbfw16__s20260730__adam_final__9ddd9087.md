# Ficha de diagnóstico — piezo-daniel-recipe__rbfw16__s20260730__adam_final__9ddd9087

**Problema:** piezo_2d · **Campaña:** piezo-daniel-recipe · **Brazo:** rbf (ancho 16) · **Semilla:** 20260730 · **Etapa:** adam_final
**Checkpoint:** `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-seeds-20261004/results/rbf16-seed-20260730/checkpoint-rbf-seed-20260730.pt` · **Familia de modelo:** piezo_daniel_upstream (rbf) · **Parámetros:** 5376 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.04639, phi = 0.04639, u = 0.06648, v = 0.04277
- Error normalizado (RMS, todas las componentes): 0.09166; plano: 0.04639 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2365; masa en el 5 % superior 0.1104; entropía normalizada 0.9695

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| layer_bottom | 0.4762 | 0.5709 | 1.199 | 0.3764 | 0.7903 |
| layer_top | 0.5238 | 0.4291 | 0.8193 | 0.6236 | 1.191 |
| near_clamp | 0.0995 | 0.01309 | 0.1315 | 0.2681 | 2.694 |
| near_tip | 0.0995 | 0.1652 | 1.66 | 0.07563 | 0.7601 |
| steep5 | 0.05023 | 0.08271 | 1.647 | 0.04105 | 0.8173 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.4774; p95 0.8811; máx 1.41
- Concentración: Gini 0.275; masa en el 1 % / 5 % superior 0.06355 / 0.2235; entropía normalizada 0.9473
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.3226; con |Δu| -0.1724

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.52; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.01457; masa de residuo en el 5 % superior del error 0.01237
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 3.409 (por parámetro 0.0006341); número de condición 1.317e+18
- Bases (capa 0): efectivas 10.97; muertas 0.06818; aristas pequeñas 0
- ratio_struct_to_spacing_x = 9.665 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.322e+05 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- seconds: 9749
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
