# Ficha de diagnóstico — heat-v4-isocost__rbf__s903__adam_final__124bc7d5

**Problema:** heat_1d · **Campaña:** heat-v4-isocost · **Brazo:** rbf · **Semilla:** 903 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs/tier2/seed-903/rbf/checkpoint-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 2.344e-05, u = 2.344e-05
- Error normalizado (RMS, todas las componentes): 2.344e-05; plano: 2.344e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5141; masa en el 5 % superior 0.2678; entropía normalizada 0.9152

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.0003149 | 0.006297 | 0.03936 | 0.787 |
| t_q1 | 0.2481 | 0.009614 | 0.03876 | 0.1048 | 0.4225 |
| t_q2 | 0.2481 | 0.2424 | 0.9772 | 0.1835 | 0.7399 |
| t_q3 | 0.2481 | 0.3823 | 1.541 | 0.2084 | 0.8403 |
| t_q4 | 0.2558 | 0.3656 | 1.429 | 0.5032 | 1.967 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0002235; p95 0.000379; máx 0.003049
- Concentración: Gini 0.4804; masa en el 1 % / 5 % superior 0.3995 / 0.585; entropía normalizada 0.8225
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.02108; con |Δu| -0.313

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1112; solape del 5 % superior 0.1237; masa de error en el 5 % superior del residuo 0.0692; masa de residuo en el 5 % superior del error 0.0694
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.826 (por parámetro 0.002419); número de condición 8.504e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.3 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.07937
- adam_loss_end: 5.38e-07
- adam_loss_start: 0.4055
- adam_n_logged: 181
- adam_rel_gain_last20pct: -8.117

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
