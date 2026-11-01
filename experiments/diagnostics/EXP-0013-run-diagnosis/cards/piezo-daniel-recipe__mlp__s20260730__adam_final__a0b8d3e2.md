# Ficha de diagnóstico — piezo-daniel-recipe__mlp__s20260730__adam_final__a0b8d3e2

**Problema:** piezo_2d · **Campaña:** piezo-daniel-recipe · **Brazo:** mlp · **Semilla:** 20260730 · **Etapa:** adam_final
**Checkpoint:** `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-seeds-20261004/results/mlp-seed-20260730/checkpoint-mlp-seed-20260730.pt` · **Familia de modelo:** piezo_daniel_upstream (mlp) · **Parámetros:** 5758 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.04599, phi = 0.04599, u = 0.1398, v = 0.1944
- Error normalizado (RMS, todas las componentes): 0.2438; plano: 0.04599 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2975; masa en el 5 % superior 0.1122; entropía normalizada 0.9581

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| layer_bottom | 0.4762 | 0.5052 | 1.061 | 0.4375 | 0.9187 |
| layer_top | 0.5238 | 0.4948 | 0.9445 | 0.5625 | 1.074 |
| near_clamp | 0.0995 | 0.002498 | 0.0251 | 0.3415 | 3.432 |
| near_tip | 0.0995 | 0.2015 | 2.025 | 0.05778 | 0.5807 |
| steep5 | 0.05023 | 0.1032 | 2.055 | 0.03078 | 0.6128 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.5456; p95 1.014; máx 1.307
- Concentración: Gini 0.3138; masa en el 1 % / 5 % superior 0.0486 / 0.202; entropía normalizada 0.9405
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.5539; con |Δu| -0.6155

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.6447; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.0005397; masa de residuo en el 5 % superior del error 0.02884
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 1.608 (por parámetro 0.0002792); número de condición 2.467e+18

## 6. Señales de entrenamiento registradas

- seconds: 4144
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
