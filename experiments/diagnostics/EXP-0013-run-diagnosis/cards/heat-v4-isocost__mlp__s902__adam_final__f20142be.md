# Ficha de diagnóstico — heat-v4-isocost__mlp__s902__adam_final__f20142be

**Problema:** heat_1d · **Campaña:** heat-v4-isocost · **Brazo:** mlp · **Semilla:** 902 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs/tier2/seed-902/fnn/checkpoint-final.pt` · **Familia de modelo:** tutorial_fnn (mlp) · **Parámetros:** 921 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 1.02e-05, u = 1.02e-05
- Error normalizado (RMS, todas las componentes): 1.02e-05; plano: 1.02e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4149; masa en el 5 % superior 0.2546; entropía normalizada 0.9343

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.005704 | 0.1141 | 0.12 | 2.399 |
| t_q1 | 0.2481 | 0.1588 | 0.6402 | 0.3794 | 1.53 |
| t_q2 | 0.2481 | 0.2668 | 1.076 | 0.1772 | 0.7144 |
| t_q3 | 0.2481 | 0.1641 | 0.6617 | 0.1478 | 0.596 |
| t_q4 | 0.2558 | 0.4102 | 1.604 | 0.2955 | 1.155 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0001011; p95 0.0001886; máx 0.0006108
- Concentración: Gini 0.4346; masa en el 1 % / 5 % superior 0.1753 / 0.3962; entropía normalizada 0.9039
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.03623; con |Δu| -0.1745

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.07466; solape del 5 % superior 0.02173; masa de error en el 5 % superior del residuo 0.03; masa de residuo en el 5 % superior del error 0.05058
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.723 (por parámetro 0.002956); número de condición 8.648e+300

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.1356
- adam_loss_end: 3.173e-08
- adam_loss_start: 0.5795
- adam_n_logged: 474
- adam_rel_gain_last20pct: -0.9914

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned, plateau_adam

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
