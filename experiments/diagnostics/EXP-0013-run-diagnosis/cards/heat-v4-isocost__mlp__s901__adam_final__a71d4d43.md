# Ficha de diagnóstico — heat-v4-isocost__mlp__s901__adam_final__a71d4d43

**Problema:** heat_1d · **Campaña:** heat-v4-isocost · **Brazo:** mlp · **Semilla:** 901 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs/tier2/seed-901/fnn/checkpoint-final.pt` · **Familia de modelo:** tutorial_fnn (mlp) · **Parámetros:** 921 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0007349, u = 0.0007349
- Error normalizado (RMS, todas las componentes): 0.0007349; plano: 0.0007349 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4931; masa en el 5 % superior 0.264; entropía normalizada 0.9178

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.000109 | 0.002179 | 0.006279 | 0.1256 |
| t_q1 | 0.2481 | 0.01042 | 0.04202 | 0.08165 | 0.3291 |
| t_q2 | 0.2481 | 0.07525 | 0.3034 | 0.1491 | 0.6009 |
| t_q3 | 0.2481 | 0.2823 | 1.138 | 0.3205 | 1.292 |
| t_q4 | 0.2558 | 0.632 | 2.471 | 0.4488 | 1.754 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0009868; p95 0.001701; máx 0.002421
- Concentración: Gini 0.3762; masa en el 1 % / 5 % superior 0.05166 / 0.1948; entropía normalizada 0.9466
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.5311; con |Δu| 0.1034

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.7724; solape del 5 % superior 0.4888; masa de error en el 5 % superior del residuo 0.2026; masa de residuo en el 5 % superior del error 0.1604
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.591 (por parámetro 0.002813); número de condición 9.689e+300

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.1419
- adam_loss_end: 2.074e-08
- adam_loss_start: 0.3546
- adam_n_logged: 481
- adam_rel_gain_last20pct: 0.259

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_aligned

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
