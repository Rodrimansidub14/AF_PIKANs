# Ficha de diagnóstico — piezo-daniel-recipe__mlp__s20260728__adam_final__92a0f11c

**Problema:** piezo_2d · **Campaña:** piezo-daniel-recipe · **Brazo:** mlp · **Semilla:** 20260728 · **Etapa:** adam_final
**Checkpoint:** `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-recipe-20261003/results/mlp-seed-20260728/checkpoint-mlp-seed-20260728.pt` · **Familia de modelo:** piezo_daniel_upstream (mlp) · **Parámetros:** 5758 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.04682, phi = 0.04682, u = 0.1442, v = 0.1983
- Error normalizado (RMS, todas las componentes): 0.2496; plano: 0.04682 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2941; masa en el 5 % superior 0.1181; entropía normalizada 0.9587

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| layer_bottom | 0.4762 | 0.5369 | 1.127 | 0.4445 | 0.9334 |
| layer_top | 0.5238 | 0.4631 | 0.8842 | 0.5555 | 1.061 |
| near_clamp | 0.0995 | 0.002801 | 0.02815 | 0.344 | 3.458 |
| near_tip | 0.0995 | 0.1934 | 1.944 | 0.06868 | 0.6902 |
| steep5 | 0.05023 | 0.09851 | 1.961 | 0.03767 | 0.7499 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.533; p95 0.9845; máx 1.372
- Concentración: Gini 0.302; masa en el 1 % / 5 % superior 0.05274 / 0.2088; entropía normalizada 0.9427
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.482; con |Δu| -0.6037

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.5736; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.0007054; masa de residuo en el 5 % superior del error 0.03781
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 1.672 (por parámetro 0.0002904); número de condición 1.462e+18

## 6. Señales de entrenamiento registradas

- seconds: 3873
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
