# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s803__lbfgs_final__31ed0396

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 803 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-mlp-seed-803/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9227, p = 1.053, u = 0.9195, v = 1.033
- Error normalizado (RMS, todas las componentes): 1.739; plano: 0.9227 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3046; masa en el 5 % superior 0.3252; entropía normalizada 0.9221

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2819 | 4.184 | 0.2832 | 4.204 |
| steep5 | 0.05009 | 0.2553 | 5.097 | 0.2648 | 5.287 |
| wake | 0.15 | 0.1591 | 1.061 | 0.01684 | 0.1123 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005791; p95 0.01443; máx 0.03782
- Concentración: Gini 0.5651; masa en el 1 % / 5 % superior 0.2149 / 0.6338; entropía normalizada 0.8027
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2306; con |Δu| 0.275

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3285; solape del 5 % superior 0.6435; masa de error en el 5 % superior del residuo 0.2455; masa de residuo en el 5 % superior del error 0.5097
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.485 (por parámetro 0.0003125); número de condición 1.277e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.005717
- adam_loss_end: 0.002491
- adam_loss_start: 0.00295
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.03006
- adam_seconds: 75.04
- adam_steps: 2000
- lbfgs_loss_end: 0.001604
- lbfgs_loss_start: 0.002444
- lbfgs_n_logged: 10
- lbfgs_seconds: 48.35
- lbfgs_to_adam_loss_ratio: 0.6437
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
