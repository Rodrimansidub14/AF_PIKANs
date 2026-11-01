# Ficha de diagnóstico — piezo-daniel-recipe__rbfw16__s20260729__adam_final__c5aebf86

**Problema:** piezo_2d · **Campaña:** piezo-daniel-recipe · **Brazo:** rbf (ancho 16) · **Semilla:** 20260729 · **Etapa:** adam_final
**Checkpoint:** `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-seeds-20261004/results/rbf16-seed-20260729/checkpoint-rbf-seed-20260729.pt` · **Familia de modelo:** piezo_daniel_upstream (rbf) · **Parámetros:** 5376 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.04577, phi = 0.04577, u = 0.124, v = 0.172
- Error normalizado (RMS, todas las componentes): 0.217; plano: 0.04577 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2764; masa en el 5 % superior 0.1239; entropía normalizada 0.9612

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| layer_bottom | 0.4762 | 0.4953 | 1.04 | 0.4296 | 0.9021 |
| layer_top | 0.5238 | 0.5047 | 0.9635 | 0.5704 | 1.089 |
| near_clamp | 0.0995 | 0.007184 | 0.0722 | 0.2837 | 2.851 |
| near_tip | 0.0995 | 0.2207 | 2.218 | 0.1003 | 1.008 |
| steep5 | 0.05023 | 0.1147 | 2.284 | 0.05812 | 1.157 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.4322; p95 0.8394; máx 1.354
- Concentración: Gini 0.3024; masa en el 1 % / 5 % superior 0.07253 / 0.2474; entropía normalizada 0.9355
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.1997; con |Δu| -0.2183

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.3737; solape del 5 % superior 0.07109; masa de error en el 5 % superior del residuo 0.02314; masa de residuo en el 5 % superior del error 0.06661
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 3.427 (por parámetro 0.0006374); número de condición 5.441e+18
- Bases (capa 0): efectivas 10.97; muertas 0.06818; aristas pequeñas 0
- ratio_struct_to_spacing_x = 9.169 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.15e+05 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- seconds: 9568
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
