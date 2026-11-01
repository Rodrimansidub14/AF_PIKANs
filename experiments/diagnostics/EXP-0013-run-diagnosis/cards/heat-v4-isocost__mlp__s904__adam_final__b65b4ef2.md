# Ficha de diagnóstico — heat-v4-isocost__mlp__s904__adam_final__b65b4ef2

**Problema:** heat_1d · **Campaña:** heat-v4-isocost · **Brazo:** mlp · **Semilla:** 904 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs/tier2/seed-904/fnn/checkpoint-final.pt` · **Familia de modelo:** tutorial_fnn (mlp) · **Parámetros:** 921 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 2.085e-05, u = 2.085e-05
- Error normalizado (RMS, todas las componentes): 2.085e-05; plano: 2.085e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4674; masa en el 5 % superior 0.2982; entropía normalizada 0.9194

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.01323 | 0.2646 | 0.09209 | 1.841 |
| t_q1 | 0.2481 | 0.2835 | 1.143 | 0.4587 | 1.849 |
| t_q2 | 0.2481 | 0.2491 | 1.004 | 0.1359 | 0.5477 |
| t_q3 | 0.2481 | 0.3035 | 1.223 | 0.1658 | 0.6685 |
| t_q4 | 0.2558 | 0.1639 | 0.6409 | 0.2396 | 0.9368 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001725; p95 0.0003443; máx 0.0009219
- Concentración: Gini 0.4452; masa en el 1 % / 5 % superior 0.149 / 0.3723; entropía normalizada 0.9075
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.01093; con |Δu| 0.1076

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.06488; solape del 5 % superior 0.02776; masa de error en el 5 % superior del residuo 0.04592; masa de residuo en el 5 % superior del error 0.05249
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.777 (por parámetro 0.003015); número de condición 8.994e+300

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.149
- adam_loss_end: 2.829e-08
- adam_loss_start: 0.4829
- adam_n_logged: 484
- adam_rel_gain_last20pct: 0.3665

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
