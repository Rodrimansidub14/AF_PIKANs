# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s802__lbfgs_final__62856b0f

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 802 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-mlp-seed-802/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.8618, p = 1.013, u = 0.858, v = 0.9986
- Error normalizado (RMS, todas las componentes): 1.661; plano: 0.8618 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3008; masa en el 5 % superior 0.3574; entropía normalizada 0.9147

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3185 | 4.727 | 0.2327 | 3.454 |
| steep5 | 0.05009 | 0.2916 | 5.822 | 0.2125 | 4.243 |
| wake | 0.15 | 0.1523 | 1.016 | 0.03217 | 0.2145 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005571; p95 0.0126; máx 0.03352
- Concentración: Gini 0.5287; masa en el 1 % / 5 % superior 0.2651 / 0.6828; entropía normalizada 0.7962
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.7177; con |Δu| 0.4842

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2916; solape del 5 % superior 0.6587; masa de error en el 5 % superior del residuo 0.2471; masa de residuo en el 5 % superior del error 0.5226
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.665 (por parámetro 0.0003351); número de condición 1.255e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.004478
- adam_loss_end: 0.002525
- adam_loss_start: 0.00303
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.01614
- adam_seconds: 71.36
- adam_steps: 2000
- lbfgs_loss_end: 0.001657
- lbfgs_loss_start: 0.002479
- lbfgs_n_logged: 10
- lbfgs_seconds: 47.64
- lbfgs_to_adam_loss_ratio: 0.6561
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
