# Ficha de diagnóstico — heat-v4-isocost__mlp__s903__adam_final__2bf36057

**Problema:** heat_1d · **Campaña:** heat-v4-isocost · **Brazo:** mlp · **Semilla:** 903 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs/tier2/seed-903/fnn/checkpoint-final.pt` · **Familia de modelo:** tutorial_fnn (mlp) · **Parámetros:** 921 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 2.363e-05, u = 2.363e-05
- Error normalizado (RMS, todas las componentes): 2.363e-05; plano: 2.363e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4018; masa en el 5 % superior 0.2299; entropía normalizada 0.9399

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.008111 | 0.1622 | 0.06381 | 1.276 |
| t_q1 | 0.2481 | 0.2849 | 1.148 | 0.4319 | 1.741 |
| t_q2 | 0.2481 | 0.2527 | 1.019 | 0.1529 | 0.6164 |
| t_q3 | 0.2481 | 0.2774 | 1.118 | 0.1557 | 0.6276 |
| t_q4 | 0.2558 | 0.185 | 0.7231 | 0.2595 | 1.014 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001907; p95 0.0003521; máx 0.001164
- Concentración: Gini 0.4282; masa en el 1 % / 5 % superior 0.1752 / 0.368; entropía normalizada 0.9085
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.09202; con |Δu| 0.1705

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.111; solape del 5 % superior 0.02354; masa de error en el 5 % superior del residuo 0.03498; masa de residuo en el 5 % superior del error 0.04131
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.644 (por parámetro 0.002871); número de condición 8.354e+300

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.05604
- adam_loss_end: 3.407e-08
- adam_loss_start: 0.443
- adam_n_logged: 493
- adam_rel_gain_last20pct: 0.9939

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
