# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s804__adam_final__f7160029

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 804 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-rbf-seed-804/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.543, p = 0.7507, u = 0.5365, v = 0.79
- Error normalizado (RMS, todas las componentes): 1.215; plano: 0.543 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3468; masa en el 5 % superior 0.496; entropía normalizada 0.8676

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.4206 | 6.244 | 0.5151 | 7.646 |
| steep5 | 0.05009 | 0.3924 | 7.834 | 0.3997 | 7.98 |
| wake | 0.15 | 0.1092 | 0.728 | 0.0554 | 0.3695 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01437; p95 0.03298; máx 0.09417
- Concentración: Gini 0.5864; masa en el 1 % / 5 % superior 0.2875 / 0.6846; entropía normalizada 0.7806
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3575; con |Δu| 0.5346

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.482; solape del 5 % superior 0.55; masa de error en el 5 % superior del residuo 0.2966; masa de residuo en el 5 % superior del error 0.506
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 14.58 (por parámetro 0.003615); número de condición 2.273e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.374 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.502 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01565
- adam_loss_end: 0.001098
- adam_loss_start: 0.002627
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.1181
- adam_seconds: 250.3
- adam_steps: 2000
- lbfgs_loss_end: 0.0009302
- lbfgs_loss_start: 0.001057
- lbfgs_n_logged: 10
- lbfgs_seconds: 137
- lbfgs_to_adam_loss_ratio: 0.8468
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- (ninguno)
