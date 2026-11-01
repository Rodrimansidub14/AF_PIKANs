# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s803__adam_final__e809081c

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 803 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-mlp-seed-803/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9668, p = 1.015, u = 0.9656, v = 1.01
- Error normalizado (RMS, todas las componentes): 1.727; plano: 0.9668 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2921; masa en el 5 % superior 0.3449; entropía normalizada 0.9205

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3214 | 4.771 | 0.2055 | 3.05 |
| steep5 | 0.05009 | 0.2944 | 5.876 | 0.2157 | 4.307 |
| wake | 0.15 | 0.1624 | 1.083 | 0.01866 | 0.1244 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.00624; p95 0.0161; máx 0.03229
- Concentración: Gini 0.4733; masa en el 1 % / 5 % superior 0.1963 / 0.6135; entropía normalizada 0.8239
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3629; con |Δu| 0.3393

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2846; solape del 5 % superior 0.4913; masa de error en el 5 % superior del residuo 0.1701; masa de residuo en el 5 % superior del error 0.3567
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.365 (por parámetro 0.0002973); número de condición 1.4e+17

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
