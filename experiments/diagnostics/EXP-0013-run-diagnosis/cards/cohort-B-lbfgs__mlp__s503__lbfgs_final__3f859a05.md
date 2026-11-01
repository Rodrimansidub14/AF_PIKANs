# Ficha de diagnóstico — cohort-B-lbfgs__mlp__s503__lbfgs_final__3f859a05

**Problema:** dfg_2d · **Campaña:** cohort-B-lbfgs · **Brazo:** mlp · **Semilla:** 503 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/training/DFG-Tanh-MLP/seed-503/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_B (mlp) · **Parámetros:** 17027 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1798, p = 0.5077, u = 0.1707, v = 0.1399
- Error normalizado (RMS, todas las componentes): 0.5536; plano: 0.1798 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.234; masa en el 5 % superior 0.1805; entropía normalizada 0.9666

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1373 | 2.037 | 0.7842 | 11.64 |
| steep5 | 0.05009 | 0.1222 | 2.439 | 0.7529 | 15.03 |
| wake | 0.15 | 0.2136 | 1.425 | 0.04116 | 0.2745 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.07211; p95 0.09025; máx 2.689
- Concentración: Gini 0.5208; masa en el 1 % / 5 % superior 0.7359 / 0.8579; entropía normalizada 0.5168
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2811; con |Δu| 0.1833

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3896; solape del 5 % superior 0.5152; masa de error en el 5 % superior del residuo 0.1397; masa de residuo en el 5 % superior del error 0.5972
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 27.08 (por parámetro 0.00159); número de condición 9.402e+08

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
