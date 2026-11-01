# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s804__lbfgs_final__fa8ad4eb

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 804 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-mlp-seed-804/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.8656, p = 1.01, u = 0.862, v = 0.995
- Error normalizado (RMS, todas las componentes): 1.659; plano: 0.8656 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3028; masa en el 5 % superior 0.3578; entropía normalizada 0.9146

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3183 | 4.725 | 0.1789 | 2.656 |
| steep5 | 0.05009 | 0.2919 | 5.826 | 0.155 | 3.095 |
| wake | 0.15 | 0.1515 | 1.011 | 0.02057 | 0.1372 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005673; p95 0.01094; máx 0.04067
- Concentración: Gini 0.4815; masa en el 1 % / 5 % superior 0.3517 / 0.7511; entropía normalizada 0.7699
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.6736; con |Δu| 0.5371

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3325; solape del 5 % superior 0.6739; masa de error en el 5 % superior del residuo 0.2299; masa de residuo en el 5 % superior del error 0.5904
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.529 (por parámetro 0.000318); número de condición 1.404e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.004615
- adam_loss_end: 0.00322
- adam_loss_start: 0.002851
- adam_n_logged: 20
- adam_rel_gain_last20pct: -0.2804
- adam_seconds: 71.17
- adam_steps: 2000
- lbfgs_loss_end: 0.001658
- lbfgs_loss_start: 0.002201
- lbfgs_n_logged: 10
- lbfgs_seconds: 47.36
- lbfgs_to_adam_loss_ratio: 0.5147
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
