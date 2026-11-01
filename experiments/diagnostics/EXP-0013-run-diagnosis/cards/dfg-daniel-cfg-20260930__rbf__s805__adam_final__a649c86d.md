# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s805__adam_final__a649c86d

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 805 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-rbf-seed-805/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.585, p = 0.8747, u = 0.5778, v = 0.7615
- Error normalizado (RMS, todas las componentes): 1.296; plano: 0.585 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3233; masa en el 5 % superior 0.4452; entropía normalizada 0.8884

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3625 | 5.381 | 0.5372 | 7.974 |
| steep5 | 0.05009 | 0.3422 | 6.832 | 0.4422 | 8.828 |
| wake | 0.15 | 0.1186 | 0.7909 | 0.03742 | 0.2495 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01496; p95 0.035; máx 0.09767
- Concentración: Gini 0.5876; masa en el 1 % / 5 % superior 0.2929 / 0.738; entropía normalizada 0.7671
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3216; con |Δu| 0.6164

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5304; solape del 5 % superior 0.5957; masa de error en el 5 % superior del residuo 0.3129; masa de residuo en el 5 % superior del error 0.5537
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 15.76 (por parámetro 0.00391); número de condición 2.018e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.25 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.488 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01237
- adam_loss_end: 0.00107
- adam_loss_start: 0.002516
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.03815
- adam_seconds: 238.1
- adam_steps: 2000
- lbfgs_loss_end: 0.0009426
- lbfgs_loss_start: 0.001064
- lbfgs_n_logged: 10
- lbfgs_seconds: 132.1
- lbfgs_to_adam_loss_ratio: 0.8813
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, residual_aligned

## 8. Límites y datos ausentes

- (ninguno)
