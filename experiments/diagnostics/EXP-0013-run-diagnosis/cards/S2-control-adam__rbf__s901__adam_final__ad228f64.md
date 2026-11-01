# Ficha de diagnóstico — S2-control-adam__rbf__s901__adam_final__ad228f64

**Problema:** heat_1d · **Campaña:** S2-control-adam · **Brazo:** rbf · **Semilla:** 901 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2/runs/tier2/seed-901/rbf/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0002442, u = 0.0002442
- Error normalizado (RMS, todas las componentes): 0.0002442; plano: 0.0002442 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.508; masa en el 5 % superior 0.2231; entropía normalizada 0.9212

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 3.177e-05 | 0.0006352 | 0.01024 | 0.2047 |
| t_q1 | 0.2481 | 0.003196 | 0.01288 | 0.05309 | 0.214 |
| t_q2 | 0.2481 | 0.08588 | 0.3462 | 0.196 | 0.7903 |
| t_q3 | 0.2481 | 0.3641 | 1.468 | 0.2998 | 1.208 |
| t_q4 | 0.2558 | 0.5468 | 2.137 | 0.4511 | 1.763 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0004472; p95 0.0008835; máx 0.005501
- Concentración: Gini 0.4631; masa en el 1 % / 5 % superior 0.1685 / 0.3698; entropía normalizada 0.8987
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.2432; con |Δu| -0.03283

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3112; solape del 5 % superior 0.2915; masa de error en el 5 % superior del residuo 0.1048; masa de residuo en el 5 % superior del error 0.1536
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.638 (por parámetro 0.002094); número de condición 8.793e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.32 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.2558
- adam_loss_end: 2.015e-07
- adam_loss_start: 0.005612
- adam_n_logged: 200
- adam_rel_gain_last20pct: -5.869
- lbfgs_loss_end: 1.153e-08
- lbfgs_loss_start: 1.153e-08
- lbfgs_n_logged: 1
- lbfgs_to_adam_loss_ratio: 0.05723

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
