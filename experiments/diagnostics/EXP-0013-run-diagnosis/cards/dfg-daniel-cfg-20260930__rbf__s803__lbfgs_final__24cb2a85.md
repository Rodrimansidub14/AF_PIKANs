# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s803__lbfgs_final__24cb2a85

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 803 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-rbf-seed-803/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5531, p = 0.8457, u = 0.5466, v = 0.6534
- Error normalizado (RMS, todas las componentes): 1.2; plano: 0.5531 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3129; masa en el 5 % superior 0.4219; entropía normalizada 0.8957

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3182 | 4.723 | 0.6519 | 9.677 |
| steep5 | 0.05009 | 0.3161 | 6.31 | 0.6164 | 12.3 |
| wake | 0.15 | 0.1176 | 0.7843 | 0.01305 | 0.08705 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01599; p95 0.03186; máx 0.1934
- Concentración: Gini 0.6322; masa en el 1 % / 5 % superior 0.4758 / 0.8296; entropía normalizada 0.7014
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3124; con |Δu| 0.5502

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.6086; solape del 5 % superior 0.4848; masa de error en el 5 % superior del residuo 0.3171; masa de residuo en el 5 % superior del error 0.6494
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 17.32 (por parámetro 0.004295); número de condición 2.479e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 1.92 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.031 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

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

- concentrated_error, residual_aligned

## 8. Límites y datos ausentes

- (ninguno)
