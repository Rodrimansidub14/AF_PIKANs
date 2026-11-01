# Ficha de diagnóstico — piezo-daniel-recipe__mlp__s20260732__adam_final__130b6f1d

**Problema:** piezo_2d · **Campaña:** piezo-daniel-recipe · **Brazo:** mlp · **Semilla:** 20260732 · **Etapa:** adam_final
**Checkpoint:** `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-seeds2-20261004/results/mlp-seed-20260732/checkpoint-mlp-seed-20260732.pt` · **Familia de modelo:** piezo_daniel_upstream (mlp) · **Parámetros:** 5758 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.04555, phi = 0.04555, u = 0.1388, v = 0.1897
- Error normalizado (RMS, todas las componentes): 0.2394; plano: 0.04555 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3029; masa en el 5 % superior 0.1194; entropía normalizada 0.9566

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| layer_bottom | 0.4762 | 0.5139 | 1.079 | 0.4568 | 0.9593 |
| layer_top | 0.5238 | 0.4861 | 0.928 | 0.5432 | 1.037 |
| near_clamp | 0.0995 | 0.002568 | 0.0258 | 0.3538 | 3.556 |
| near_tip | 0.0995 | 0.202 | 2.03 | 0.06763 | 0.6796 |
| steep5 | 0.05023 | 0.103 | 2.052 | 0.03735 | 0.7437 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.5261; p95 0.9765; máx 1.432
- Concentración: Gini 0.3132; masa en el 1 % / 5 % superior 0.05813 / 0.219; entropía normalizada 0.9385
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.4922; con |Δu| -0.5775

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.5965; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.0008773; masa de residuo en el 5 % superior del error 0.03324
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 1.725 (por parámetro 0.0002997); número de condición 9.796e+17

## 6. Señales de entrenamiento registradas

- seconds: 3991
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
