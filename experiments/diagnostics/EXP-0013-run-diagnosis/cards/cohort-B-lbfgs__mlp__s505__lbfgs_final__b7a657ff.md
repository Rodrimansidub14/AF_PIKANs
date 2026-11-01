# Ficha de diagnóstico — cohort-B-lbfgs__mlp__s505__lbfgs_final__b7a657ff

**Problema:** dfg_2d · **Campaña:** cohort-B-lbfgs · **Brazo:** mlp · **Semilla:** 505 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/training/DFG-Tanh-MLP/seed-505/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_B (mlp) · **Parámetros:** 17027 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.01412, p = 0.0229, u = 0.01371, v = 0.03003
- Error normalizado (RMS, todas las componentes): 0.04017; plano: 0.01412 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3012; masa en el 5 % superior 0.4443; entropía normalizada 0.8884

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3977 | 5.903 | 0.8505 | 12.62 |
| steep5 | 0.05009 | 0.2415 | 4.821 | 0.7276 | 14.52 |
| wake | 0.15 | 0.205 | 1.367 | 0.01793 | 0.1196 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.03059; p95 0.03279; máx 1.573
- Concentración: Gini 0.6147; masa en el 1 % / 5 % superior 0.8035 / 0.9229; entropía normalizada 0.4651
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3091; con |Δu| 0.2972

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3254; solape del 5 % superior 0.4413; masa de error en el 5 % superior del residuo 0.2866; masa de residuo en el 5 % superior del error 0.7067
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 33.62 (por parámetro 0.001975); número de condición 1.828e+08

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
