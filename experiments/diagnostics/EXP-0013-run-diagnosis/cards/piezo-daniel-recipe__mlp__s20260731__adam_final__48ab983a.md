# Ficha de diagnóstico — piezo-daniel-recipe__mlp__s20260731__adam_final__48ab983a

**Problema:** piezo_2d · **Campaña:** piezo-daniel-recipe · **Brazo:** mlp · **Semilla:** 20260731 · **Etapa:** adam_final
**Checkpoint:** `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-seeds2-20261004/results/mlp-seed-20260731/checkpoint-mlp-seed-20260731.pt` · **Familia de modelo:** piezo_daniel_upstream (mlp) · **Parámetros:** 5758 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.04605, phi = 0.04605, u = 0.1312, v = 0.1717
- Error normalizado (RMS, todas las componentes): 0.221; plano: 0.04605 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3033; masa en el 5 % superior 0.1172; entropía normalizada 0.9567

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| layer_bottom | 0.4762 | 0.5032 | 1.057 | 0.4544 | 0.9543 |
| layer_top | 0.5238 | 0.4968 | 0.9484 | 0.5456 | 1.042 |
| near_clamp | 0.0995 | 0.00239 | 0.02402 | 0.2392 | 2.404 |
| near_tip | 0.0995 | 0.2009 | 2.019 | 0.1017 | 1.022 |
| steep5 | 0.05023 | 0.1026 | 2.043 | 0.05255 | 1.046 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.6772; p95 1.045; máx 1.352
- Concentración: Gini 0.1673; masa en el 1 % / 5 % superior 0.03432 / 0.1409; entropía normalizada 0.9783
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.2427; con |Δu| -0.3342

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.3781; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.009026; masa de residuo en el 5 % superior del error 0.05253
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 1.543 (por parámetro 0.0002679); número de condición 4.238e+18

## 6. Señales de entrenamiento registradas

- seconds: 3918
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
