# Ficha de diagnóstico — S2-control-adam__rbf__s902__adam_final__c7130021

**Problema:** heat_1d · **Campaña:** S2-control-adam · **Brazo:** rbf · **Semilla:** 902 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2/runs/tier2/seed-902/rbf/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0001564, u = 0.0001564
- Error normalizado (RMS, todas las componentes): 0.0001564; plano: 0.0001564 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5811; masa en el 5 % superior 0.2839; entropía normalizada 0.8997

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 1.756e-05 | 0.0003512 | 0.01275 | 0.2549 |
| t_q1 | 0.2481 | 0.0005069 | 0.002044 | 0.04619 | 0.1862 |
| t_q2 | 0.2481 | 0.01864 | 0.07514 | 0.1512 | 0.6094 |
| t_q3 | 0.2481 | 0.232 | 0.9353 | 0.3529 | 1.423 |
| t_q4 | 0.2558 | 0.7488 | 2.927 | 0.4498 | 1.758 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0003201; p95 0.0005833; máx 0.004148
- Concentración: Gini 0.4562; masa en el 1 % / 5 % superior 0.2079 / 0.3949; entropía normalizada 0.892
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.2664; con |Δu| -0.2714

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5088; solape del 5 % superior 0.1865; masa de error en el 5 % superior del residuo 0.1117; masa de residuo en el 5 % superior del error 0.1282
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.474 (por parámetro 0.002323); número de condición 8.791e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.31 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.04547
- adam_loss_end: 1.182e-07
- adam_loss_start: 0.009208
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.8363
- lbfgs_loss_end: 1.421e-08
- lbfgs_loss_start: 1.426e-08
- lbfgs_n_logged: 2
- lbfgs_to_adam_loss_ratio: 0.1203

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_aligned

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
