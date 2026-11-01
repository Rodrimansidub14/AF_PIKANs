# Ficha de diagnóstico — cohort-B-lbfgs__mlp__s501__lbfgs_final__9d519d2e

**Problema:** dfg_2d · **Campaña:** cohort-B-lbfgs · **Brazo:** mlp · **Semilla:** 501 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/training/DFG-Tanh-MLP/seed-501/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_B (mlp) · **Parámetros:** 17027 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.01936, p = 0.0265, u = 0.01908, v = 0.03209
- Error normalizado (RMS, todas las componentes): 0.04578; plano: 0.01936 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3778; masa en el 5 % superior 0.4186; entropía normalizada 0.8826

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3545 | 5.262 | 0.8506 | 12.63 |
| steep5 | 0.05009 | 0.2669 | 5.328 | 0.727 | 14.51 |
| wake | 0.15 | 0.1709 | 1.14 | 0.02365 | 0.1577 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.02994; p95 0.03231; máx 1.548
- Concentración: Gini 0.6288; masa en el 1 % / 5 % superior 0.8148 / 0.9278; entropía normalizada 0.4474
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1984; con |Δu| 0.2861

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4389; solape del 5 % superior 0.3761; masa de error en el 5 % superior del residuo 0.2607; masa de residuo en el 5 % superior del error 0.6905
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 32.74 (por parámetro 0.001923); número de condición 6.586e+07

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
