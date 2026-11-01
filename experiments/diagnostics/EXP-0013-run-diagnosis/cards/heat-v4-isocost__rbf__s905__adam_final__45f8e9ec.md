# Ficha de diagnóstico — heat-v4-isocost__rbf__s905__adam_final__45f8e9ec

**Problema:** heat_1d · **Campaña:** heat-v4-isocost · **Brazo:** rbf · **Semilla:** 905 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs/tier2/seed-905/rbf/checkpoint-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 3.755e-05, u = 3.755e-05
- Error normalizado (RMS, todas las componentes): 3.755e-05; plano: 3.755e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.6; masa en el 5 % superior 0.4121; entropía normalizada 0.8738

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.0002702 | 0.005402 | 0.02867 | 0.5733 |
| t_q1 | 0.2481 | 0.003486 | 0.01405 | 0.09416 | 0.3796 |
| t_q2 | 0.2481 | 0.01496 | 0.06032 | 0.1283 | 0.5173 |
| t_q3 | 0.2481 | 0.1049 | 0.423 | 0.2042 | 0.8233 |
| t_q4 | 0.2558 | 0.8766 | 3.427 | 0.5733 | 2.241 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0002318; p95 0.0004054; máx 0.008619
- Concentración: Gini 0.4458; masa en el 1 % / 5 % superior 0.3301 / 0.5114; entropía normalizada 0.8205
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.2035; con |Δu| -0.1818

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3117; solape del 5 % superior 0.3217; masa de error en el 5 % superior del residuo 0.2072; masa de residuo en el 5 % superior del error 0.1442
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.386 (por parámetro 0.002025); número de condición 8.687e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.2634
- adam_loss_end: 9.959e-08
- adam_loss_start: 0.4138
- adam_n_logged: 181
- adam_rel_gain_last20pct: -1.183

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
