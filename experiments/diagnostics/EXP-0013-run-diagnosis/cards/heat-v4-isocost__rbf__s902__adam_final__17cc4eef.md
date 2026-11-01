# Ficha de diagnóstico — heat-v4-isocost__rbf__s902__adam_final__17cc4eef

**Problema:** heat_1d · **Campaña:** heat-v4-isocost · **Brazo:** rbf · **Semilla:** 902 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs/tier2/seed-902/rbf/checkpoint-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0001833, u = 0.0001833
- Error normalizado (RMS, todas las componentes): 0.0001833; plano: 0.0001833 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4728; masa en el 5 % superior 0.1766; entropía normalizada 0.9291

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 7.556e-05 | 0.001511 | 0.01192 | 0.2384 |
| t_q1 | 0.2481 | 0.00879 | 0.03544 | 0.09066 | 0.3655 |
| t_q2 | 0.2481 | 0.22 | 0.8867 | 0.2607 | 1.051 |
| t_q3 | 0.2481 | 0.3802 | 1.533 | 0.2137 | 0.8614 |
| t_q4 | 0.2558 | 0.391 | 1.528 | 0.435 | 1.7 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0003932; p95 0.0007036; máx 0.005479
- Concentración: Gini 0.4264; masa en el 1 % / 5 % superior 0.1844 / 0.364; entropía normalizada 0.9021
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.1324; con |Δu| -0.06858

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.141; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.02011; masa de residuo en el 5 % superior del error 0.01174
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.495 (por parámetro 0.002329); número de condición 8.893e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.09624
- adam_loss_end: 2.328e-08
- adam_loss_start: 0.356
- adam_n_logged: 180
- adam_rel_gain_last20pct: 0.2468

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_misaligned

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
