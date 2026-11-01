# Ficha de diagnóstico — dfg-long-20261001__rbf__s803__adam_final__3abe2f67

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** rbf · **Semilla:** 803 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-rbf-seed-803/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.4576, p = 0.6333, u = 0.4532, v = 0.5809
- Error normalizado (RMS, todas las componentes): 0.9715; plano: 0.4576 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3363; masa en el 5 % superior 0.4794; entropía normalizada 0.8752

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3397 | 5.042 | 0.67 | 9.945 |
| steep5 | 0.05009 | 0.3419 | 6.826 | 0.662 | 13.22 |
| wake | 0.15 | 0.09879 | 0.6588 | 0.01408 | 0.0939 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01458; p95 0.02544; máx 0.2121
- Concentración: Gini 0.689; masa en el 1 % / 5 % superior 0.5878 / 0.8618; entropía normalizada 0.6636
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4556; con |Δu| 0.5039

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3994; solape del 5 % superior 0.5783; masa de error en el 5 % superior del residuo 0.3798; masa de residuo en el 5 % superior del error 0.6866
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 23.95 (por parámetro 0.00594); número de condición 5.082e+06
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 1.387 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.733 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 7.277e-05
- adam_loss_end: 0.0007668
- adam_loss_start: 0.002698
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.03402
- adam_seconds: 2493
- adam_steps: 2e+04
- lbfgs_iterations: 4025
- lbfgs_log_slope_last10: -0.009257
- lbfgs_loss_end: 0.0002728
- lbfgs_loss_start: 0.0007588
- lbfgs_n_logged: 161
- lbfgs_seconds: 600.2
- lbfgs_to_adam_loss_ratio: 0.3557
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error

## 8. Límites y datos ausentes

- (ninguno)
