# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s801__adam_final__2384e2ca

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 801 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-mlp-seed-801/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9703, p = 1.009, u = 0.9693, v = 1.01
- Error normalizado (RMS, todas las componentes): 1.726; plano: 0.9703 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2925; masa en el 5 % superior 0.3436; entropía normalizada 0.9208

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3216 | 4.774 | 0.2432 | 3.61 |
| steep5 | 0.05009 | 0.2945 | 5.88 | 0.2378 | 4.748 |
| wake | 0.15 | 0.163 | 1.087 | 0.03513 | 0.2343 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005634; p95 0.01302; máx 0.02159
- Concentración: Gini 0.4212; masa en el 1 % / 5 % superior 0.1128 / 0.3833; entropía normalizada 0.8918
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3905; con |Δu| 0.2094

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2581; solape del 5 % superior 0.563; masa de error en el 5 % superior del residuo 0.212; masa de residuo en el 5 % superior del error 0.2647
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.549 (por parámetro 0.0003205); número de condición 1.363e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.008032
- adam_loss_end: 0.002436
- adam_loss_start: 0.002907
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.02675
- adam_seconds: 70.04
- adam_steps: 2000
- lbfgs_loss_end: 0.001616
- lbfgs_loss_start: 0.002127
- lbfgs_n_logged: 10
- lbfgs_seconds: 46.1
- lbfgs_to_adam_loss_ratio: 0.6634
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
