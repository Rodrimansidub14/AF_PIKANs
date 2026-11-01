# Ficha de diagnóstico — ac-rerun-20260930__mlp__s603__lbfgs_final__4de47e24

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** mlp · **Semilla:** 603 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/mlp-seed-603/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 921 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0178, u = 0.0178
- Error normalizado (RMS, todas las componentes): 0.0178; plano: 0.0178 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.8877; masa en el 5 % superior 0.9899; entropía normalizada 0.6053

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.9999 | 1.627 | 0.7997 | 1.301 |
| steep5 | 0.05005 | 0.001116 | 0.0223 | 0.1857 | 3.711 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0005307; p95 0.001079; máx 0.00695
- Concentración: Gini 0.4858; masa en el 1 % / 5 % superior 0.1825 / 0.4495; entropía normalizada 0.8787
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2533; con |Δu| 0.3083

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3978; solape del 5 % superior 0.1596; masa de error en el 5 % superior del residuo 0.3382; masa de residuo en el 5 % superior del error 0.1115
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 19.48 (por parámetro 0.02115); número de condición 6.282e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.00368
- adam_loss_end: 0.002188
- adam_loss_start: 0.5336
- adam_n_logged: 400
- adam_rel_gain_last20pct: 0.5005
- adam_seconds: 346.9
- adam_steps: 4e+04
- lbfgs_iterations: 1.34e+04
- lbfgs_seconds: 300.7
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
