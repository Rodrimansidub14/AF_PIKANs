# Ficha de diagnóstico — ns2d-cg-20260930__mlp__s703__adam_final__d7659ac6

**Problema:** ns2d_cg · **Campaña:** ns2d-cg-20260930 · **Brazo:** mlp · **Semilla:** 703 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/ns2d-cg-20260930/ns/mlp-seed-703/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 41003 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1021, p = 0.3942, u = 0.08019, v = 0.5026
- Error normalizado (RMS, todas las componentes): 0.6438; plano: 0.1021 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3915; masa en el 5 % superior 0.4168; entropía normalizada 0.8975

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_corner | 0.171 | 0.2802 | 1.638 | 0.7955 | 4.651 |
| steep5 | 0.05004 | 0.09194 | 1.837 | 0.4991 | 9.975 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.03778; p95 0.05514; máx 0.374
- Concentración: Gini 0.4408; masa en el 1 % / 5 % superior 0.4269 / 0.7222; entropía normalizada 0.776
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.6336; con |Δu| 0.4767

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.2016; solape del 5 % superior 0.1674; masa de error en el 5 % superior del residuo 0.1345; masa de residuo en el 5 % superior del error 0.1504
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 14.8 (por parámetro 0.0003609); número de condición 2.059e+09

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.03643
- adam_loss_end: 0.001057
- adam_loss_start: 0.141
- adam_n_logged: 200
- adam_rel_gain_last20pct: -3.578
- adam_seconds: 685.6
- adam_steps: 2e+04
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
