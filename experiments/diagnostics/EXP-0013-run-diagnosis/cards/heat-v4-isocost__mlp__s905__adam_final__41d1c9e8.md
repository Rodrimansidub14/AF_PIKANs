# Ficha de diagnóstico — heat-v4-isocost__mlp__s905__adam_final__41d1c9e8

**Problema:** heat_1d · **Campaña:** heat-v4-isocost · **Brazo:** mlp · **Semilla:** 905 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs/tier2/seed-905/fnn/checkpoint-final.pt` · **Familia de modelo:** tutorial_fnn (mlp) · **Parámetros:** 921 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 1.268e-05, u = 1.268e-05
- Error normalizado (RMS, todas las componentes): 1.268e-05; plano: 1.268e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4338; masa en el 5 % superior 0.2938; entropía normalizada 0.9252

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.0424 | 0.8479 | 0.1167 | 2.333 |
| t_q1 | 0.2481 | 0.3298 | 1.33 | 0.3965 | 1.598 |
| t_q2 | 0.2481 | 0.2709 | 1.092 | 0.1548 | 0.6239 |
| t_q3 | 0.2481 | 0.2157 | 0.8696 | 0.1909 | 0.7694 |
| t_q4 | 0.2558 | 0.1836 | 0.7177 | 0.2579 | 1.008 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001455; p95 0.0002797; máx 0.001028
- Concentración: Gini 0.4328; masa en el 1 % / 5 % superior 0.1582 / 0.3774; entropía normalizada 0.9094
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1488; con |Δu| -0.1451

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.01474; solape del 5 % superior 0.111; masa de error en el 5 % superior del residuo 0.06479; masa de residuo en el 5 % superior del error 0.07456
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.74 (por parámetro 0.002975); número de condición 8.12e+300

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.1971
- adam_loss_end: 1.989e-08
- adam_loss_start: 0.5052
- adam_n_logged: 526
- adam_rel_gain_last20pct: 0.5908

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
