# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s805__lbfgs_final__86cceb43

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 805 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-rbf-seed-805/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5688, p = 0.8519, u = 0.5625, v = 0.6704
- Error normalizado (RMS, todas las componentes): 1.221; plano: 0.5688 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3134; masa en el 5 % superior 0.4193; entropía normalizada 0.8961

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3149 | 4.675 | 0.6438 | 9.556 |
| steep5 | 0.05009 | 0.313 | 6.248 | 0.5883 | 11.74 |
| wake | 0.15 | 0.1174 | 0.7826 | 0.01466 | 0.09779 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01546; p95 0.03224; máx 0.1673
- Concentración: Gini 0.624; masa en el 1 % / 5 % superior 0.4267 / 0.8182; entropía normalizada 0.7167
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3203; con |Δu| 0.5274

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5305; solape del 5 % superior 0.5109; masa de error en el 5 % superior del residuo 0.3138; masa de residuo en el 5 % superior del error 0.6118
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 17.31 (por parámetro 0.004294); número de condición 2.442e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 1.95 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.984 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

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
