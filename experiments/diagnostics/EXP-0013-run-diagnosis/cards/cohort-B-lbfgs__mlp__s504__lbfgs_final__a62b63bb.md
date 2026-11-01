# Ficha de diagnóstico — cohort-B-lbfgs__mlp__s504__lbfgs_final__a62b63bb

**Problema:** dfg_2d · **Campaña:** cohort-B-lbfgs · **Brazo:** mlp · **Semilla:** 504 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/training/DFG-Tanh-MLP/seed-504/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_B (mlp) · **Parámetros:** 17027 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.01682, p = 0.03062, u = 0.01635, v = 0.03093
- Error normalizado (RMS, todas las componentes): 0.04649; plano: 0.01682 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2768; masa en el 5 % superior 0.3337; entropía normalizada 0.9256

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2919 | 4.334 | 0.8075 | 11.99 |
| steep5 | 0.05009 | 0.1865 | 3.724 | 0.6623 | 13.22 |
| wake | 0.15 | 0.2509 | 1.674 | 0.0277 | 0.1847 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.02537; p95 0.03099; máx 1.039
- Concentración: Gini 0.6142; masa en el 1 % / 5 % superior 0.7408 / 0.8951; entropía normalizada 0.5215
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2726; con |Δu| 0.2832

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4555; solape del 5 % superior 0.4087; masa de error en el 5 % superior del residuo 0.2087; masa de residuo en el 5 % superior del error 0.5313
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 31.1 (por parámetro 0.001827); número de condición 1.035e+08

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
