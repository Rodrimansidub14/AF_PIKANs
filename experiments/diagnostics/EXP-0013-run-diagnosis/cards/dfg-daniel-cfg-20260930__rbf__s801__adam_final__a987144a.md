# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s801__adam_final__a987144a

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 801 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-rbf-seed-801/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5665, p = 0.7594, u = 0.5604, v = 0.8143
- Error normalizado (RMS, todas las componentes): 1.246; plano: 0.5665 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3501; masa en el 5 % superior 0.4856; entropía normalizada 0.8694

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.4223 | 6.268 | 0.4737 | 7.031 |
| steep5 | 0.05009 | 0.3919 | 7.824 | 0.3489 | 6.965 |
| wake | 0.15 | 0.1153 | 0.7686 | 0.07475 | 0.4985 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01347; p95 0.0324; máx 0.07678
- Concentración: Gini 0.563; masa en el 1 % / 5 % superior 0.255 / 0.6678; entropía normalizada 0.7937
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2827; con |Δu| 0.459

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4835; solape del 5 % superior 0.5935; masa de error en el 5 % superior del residuo 0.302; masa de residuo en el 5 % superior del error 0.4867
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 15.24 (por parámetro 0.003779); número de condición 2.348e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.748 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.319 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.02228
- adam_loss_end: 0.001109
- adam_loss_start: 0.002813
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.07046
- adam_seconds: 249.8
- adam_steps: 2000
- lbfgs_loss_end: 0.0008919
- lbfgs_loss_start: 0.001097
- lbfgs_n_logged: 10
- lbfgs_seconds: 136.6
- lbfgs_to_adam_loss_ratio: 0.804
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- (ninguno)
