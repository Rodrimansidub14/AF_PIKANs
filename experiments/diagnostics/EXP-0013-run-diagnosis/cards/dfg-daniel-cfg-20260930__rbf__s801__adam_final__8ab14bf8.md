# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s801__adam_final__8ab14bf8

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 801 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-rbf-seed-801/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5981, p = 0.8827, u = 0.5903, v = 0.8272
- Error normalizado (RMS, todas las componentes): 1.346; plano: 0.5981 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3307; masa en el 5 % superior 0.4452; entropía normalizada 0.886

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3853 | 5.719 | 0.451 | 6.694 |
| steep5 | 0.05009 | 0.3578 | 7.142 | 0.3307 | 6.603 |
| wake | 0.15 | 0.126 | 0.8405 | 0.08262 | 0.551 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01318; p95 0.03074; máx 0.07947
- Concentración: Gini 0.5082; masa en el 1 % / 5 % superior 0.2628 / 0.6437; entropía normalizada 0.8079
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4167; con |Δu| 0.5365

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4636; solape del 5 % superior 0.5891; masa de error en el 5 % superior del residuo 0.2823; masa de residuo en el 5 % superior del error 0.4708
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 15.53 (por parámetro 0.003852); número de condición 2.269e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.872 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.328 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01278
- adam_loss_end: 0.001252
- adam_loss_start: 0.002809
- adam_n_logged: 20
- adam_rel_gain_last20pct: -0.04181
- adam_seconds: 237.5
- adam_steps: 2000
- lbfgs_loss_end: 0.0009034
- lbfgs_loss_start: 0.001119
- lbfgs_n_logged: 10
- lbfgs_seconds: 131.5
- lbfgs_to_adam_loss_ratio: 0.7214
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, plateau_adam

## 8. Límites y datos ausentes

- (ninguno)
