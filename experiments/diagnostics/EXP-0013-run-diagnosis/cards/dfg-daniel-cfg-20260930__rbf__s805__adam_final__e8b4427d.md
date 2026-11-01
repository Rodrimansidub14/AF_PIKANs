# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s805__adam_final__e8b4427d

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 805 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-rbf-seed-805/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5475, p = 0.7522, u = 0.5416, v = 0.7575
- Error normalizado (RMS, todas las componentes): 1.197; plano: 0.5475 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3459; masa en el 5 % superior 0.4897; entropía normalizada 0.8703

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.4004 | 5.944 | 0.5322 | 7.9 |
| steep5 | 0.05009 | 0.3782 | 7.551 | 0.4348 | 8.679 |
| wake | 0.15 | 0.1073 | 0.7153 | 0.03302 | 0.2202 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0148; p95 0.03548; máx 0.0947
- Concentración: Gini 0.5992; masa en el 1 % / 5 % superior 0.2912 / 0.7355; entropía normalizada 0.7664
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3715; con |Δu| 0.5031

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4003; solape del 5 % superior 0.5935; masa de error en el 5 % superior del residuo 0.3335; masa de residuo en el 5 % superior del error 0.5499
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 15.9 (por parámetro 0.003944); número de condición 1.982e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.274 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.545 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.02513
- adam_loss_end: 0.001062
- adam_loss_start: 0.002521
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.04212
- adam_seconds: 248.6
- adam_steps: 2000
- lbfgs_loss_end: 0.0009312
- lbfgs_loss_start: 0.001048
- lbfgs_n_logged: 10
- lbfgs_seconds: 135.3
- lbfgs_to_adam_loss_ratio: 0.8769
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- (ninguno)
