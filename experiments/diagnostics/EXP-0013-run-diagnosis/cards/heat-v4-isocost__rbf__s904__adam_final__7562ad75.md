# Ficha de diagnóstico — heat-v4-isocost__rbf__s904__adam_final__7562ad75

**Problema:** heat_1d · **Campaña:** heat-v4-isocost · **Brazo:** rbf · **Semilla:** 904 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs/tier2/seed-904/rbf/checkpoint-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0003002, u = 0.0003002
- Error normalizado (RMS, todas las componentes): 0.0003002; plano: 0.0003002 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4654; masa en el 5 % superior 0.2234; entropía normalizada 0.9306

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 4.52e-05 | 0.0009038 | 0.00594 | 0.1188 |
| t_q1 | 0.2481 | 0.007298 | 0.02942 | 0.05603 | 0.2259 |
| t_q2 | 0.2481 | 0.1168 | 0.4707 | 0.184 | 0.7418 |
| t_q3 | 0.2481 | 0.3202 | 1.291 | 0.214 | 0.8627 |
| t_q4 | 0.2558 | 0.5557 | 2.172 | 0.546 | 2.134 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0006217; p95 0.001172; máx 0.01662
- Concentración: Gini 0.4453; masa en el 1 % / 5 % superior 0.2377 / 0.4206; entropía normalizada 0.8655
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.4351; con |Δu| -0.02787

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4784; solape del 5 % superior 0.4297; masa de error en el 5 % superior del residuo 0.1536; masa de residuo en el 5 % superior del error 0.1657
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.41 (por parámetro 0.002031); número de condición 9.353e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.05898
- adam_loss_end: 1.179e-06
- adam_loss_start: 0.4412
- adam_n_logged: 179
- adam_rel_gain_last20pct: 0.5972

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
