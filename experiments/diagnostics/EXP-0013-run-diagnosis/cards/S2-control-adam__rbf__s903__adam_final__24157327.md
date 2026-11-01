# Ficha de diagnóstico — S2-control-adam__rbf__s903__adam_final__24157327

**Problema:** heat_1d · **Campaña:** S2-control-adam · **Brazo:** rbf · **Semilla:** 903 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2/runs/tier2/seed-903/rbf/checkpoint-adam-final.pt` · **Familia de modelo:** af_sciml_model (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 5.124e-05, u = 5.124e-05
- Error normalizado (RMS, todas las componentes): 5.124e-05; plano: 5.124e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5715; masa en el 5 % superior 0.2965; entropía normalizada 0.9011

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| steep5 | 0.05001 | 0.0001119 | 0.002238 | 0.03055 | 0.6109 |
| t_q1 | 0.2481 | 0.0007202 | 0.002903 | 0.08642 | 0.3484 |
| t_q2 | 0.2481 | 0.08552 | 0.3448 | 0.1677 | 0.676 |
| t_q3 | 0.2481 | 0.2 | 0.8061 | 0.3745 | 1.51 |
| t_q4 | 0.2558 | 0.7138 | 2.79 | 0.3714 | 1.452 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0002318; p95 0.0004367; máx 0.002445
- Concentración: Gini 0.4723; masa en el 1 % / 5 % superior 0.2758 / 0.4749; entropía normalizada 0.8679
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.1796; con |Δu| -0.241

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2417; solape del 5 % superior 0.07001; masa de error en el 5 % superior del residuo 0.07203; masa de residuo en el 5 % superior del error 0.0499
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.83 (por parámetro 0.00242); número de condición 8.438e+300
- Bases (capa 0): efectivas 10.54; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 19.3 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 6.063 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.2016
- adam_loss_end: 4.326e-08
- adam_loss_start: 0.01039
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.8073
- lbfgs_log_slope_last10: -0.0002607
- lbfgs_loss_end: 1.214e-08
- lbfgs_loss_start: 1.305e-08
- lbfgs_n_logged: 156
- lbfgs_to_adam_loss_ratio: 0.2805

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
