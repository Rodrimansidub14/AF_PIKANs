# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s804__adam_final__133a471a

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 804 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-mlp-seed-804/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9027, p = 1.085, u = 0.896, v = 1.24
- Error normalizado (RMS, todas las componentes): 1.875; plano: 0.9027 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2264; masa en el 5 % superior 0.3064; entropía normalizada 0.9387

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.282 | 4.185 | 0.06478 | 0.9616 |
| steep5 | 0.05009 | 0.2576 | 5.142 | 0.03879 | 0.7744 |
| wake | 0.15 | 0.1564 | 1.043 | 0.229 | 1.527 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0127; p95 0.02089; máx 0.02505
- Concentración: Gini 0.2499; masa en el 1 % / 5 % superior 0.03609 / 0.1569; entropía normalizada 0.9646
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.5476; con |Δu| -0.1599

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1787; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.0311; masa de residuo en el 5 % superior del error 0.04114
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.459 (por parámetro 0.0003092); número de condición 1.376e+17

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

- null_state, concentrated_error, residual_misaligned, low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
