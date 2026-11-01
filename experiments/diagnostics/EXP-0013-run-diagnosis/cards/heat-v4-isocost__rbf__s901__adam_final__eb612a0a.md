# Ficha de diagnóstico — heat-v4-isocost__rbf__s901__adam_final__eb612a0a

**Problema:** heat_1d · **Campaña:** heat-v4-isocost · **Brazo:** rbf · **Semilla:** 901 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs/tier2/seed-901/rbf/checkpoint-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0002959, u = 0.0002959
- Error normalizado (RMS, todas las componentes): 0.0002959; plano: 0.0002959 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5879; masa en el 5 % superior 0.3262; entropía normalizada 0.8947

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 2.226e-05 | 0.0004451 | 0.01022 | 0.2044 |
| t_q1 | 0.2481 | 0.0002806 | 0.001131 | 0.03246 | 0.1309 |
| t_q2 | 0.2481 | 0.0169 | 0.06812 | 0.106 | 0.4275 |
| t_q3 | 0.2481 | 0.1819 | 0.7334 | 0.3018 | 1.216 |
| t_q4 | 0.2558 | 0.8009 | 3.131 | 0.5597 | 2.188 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0005275; p95 0.0009385; máx 0.004109
- Concentración: Gini 0.41; masa en el 1 % / 5 % superior 0.07876 / 0.2278; entropía normalizada 0.936
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.4053; con |Δu| -0.398

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.8042; solape del 5 % superior 0.3368; masa de error en el 5 % superior del residuo 0.2016; masa de residuo en el 5 % superior del error 0.1378
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.654 (por parámetro 0.002098); número de condición 8.862e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.2597
- adam_loss_end: 3.017e-07
- adam_loss_start: 0.42
- adam_n_logged: 181
- adam_rel_gain_last20pct: -7.164

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_aligned, plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
