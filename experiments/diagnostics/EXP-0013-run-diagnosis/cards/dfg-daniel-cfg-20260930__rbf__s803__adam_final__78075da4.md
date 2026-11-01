# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s803__adam_final__78075da4

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 803 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-rbf-seed-803/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5657, p = 0.8658, u = 0.5582, v = 0.743
- Error normalizado (RMS, todas las componentes): 1.27; plano: 0.5657 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3215; masa en el 5 % superior 0.448; entropía normalizada 0.8873

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3665 | 5.441 | 0.5373 | 7.975 |
| steep5 | 0.05009 | 0.3491 | 6.969 | 0.4624 | 9.231 |
| wake | 0.15 | 0.1183 | 0.789 | 0.01967 | 0.1312 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01563; p95 0.03462; máx 0.1297
- Concentración: Gini 0.5884; masa en el 1 % / 5 % superior 0.3528 / 0.7747; entropía normalizada 0.748
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3528; con |Δu| 0.6249

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4961; solape del 5 % superior 0.6283; masa de error en el 5 % superior del residuo 0.3384; masa de residuo en el 5 % superior del error 0.6136
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 16.15 (por parámetro 0.004005); número de condición 1.741e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.272 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.454 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01796
- adam_loss_end: 0.00101
- adam_loss_start: 0.002689
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.05411
- adam_seconds: 237.8
- adam_steps: 2000
- lbfgs_loss_end: 0.0008973
- lbfgs_loss_start: 0.001002
- lbfgs_n_logged: 10
- lbfgs_seconds: 131.6
- lbfgs_to_adam_loss_ratio: 0.8882
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- (ninguno)
