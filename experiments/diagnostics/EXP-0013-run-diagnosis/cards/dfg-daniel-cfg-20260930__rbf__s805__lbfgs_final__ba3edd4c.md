# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s805__lbfgs_final__ba3edd4c

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 805 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-rbf-seed-805/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5259, p = 0.7168, u = 0.521, v = 0.6673
- Error normalizado (RMS, todas las componentes): 1.109; plano: 0.5259 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3405; masa en el 5 % superior 0.47; entropía normalizada 0.8753

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.351 | 5.21 | 0.6293 | 9.341 |
| steep5 | 0.05009 | 0.3521 | 7.029 | 0.584 | 11.66 |
| wake | 0.15 | 0.1022 | 0.6813 | 0.01323 | 0.08821 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01554; p95 0.03232; máx 0.1612
- Concentración: Gini 0.6624; masa en el 1 % / 5 % superior 0.4143 / 0.8149; entropía normalizada 0.7174
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3567; con |Δu| 0.4295

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4335; solape del 5 % superior 0.5; masa de error en el 5 % superior del residuo 0.3509; masa de residuo en el 5 % superior del error 0.6019
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 17 (por parámetro 0.004216); número de condición 2.35e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 1.969 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.029 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

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
