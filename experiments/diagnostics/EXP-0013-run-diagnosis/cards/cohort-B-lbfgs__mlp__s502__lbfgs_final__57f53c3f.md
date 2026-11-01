# Ficha de diagnóstico — cohort-B-lbfgs__mlp__s502__lbfgs_final__57f53c3f

**Problema:** dfg_2d · **Campaña:** cohort-B-lbfgs · **Brazo:** mlp · **Semilla:** 502 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/training/DFG-Tanh-MLP/seed-502/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_B (mlp) · **Parámetros:** 17027 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.01495, p = 0.02588, u = 0.01457, v = 0.02725
- Error normalizado (RMS, todas las componentes): 0.04031; plano: 0.01495 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2634; masa en el 5 % superior 0.3429; entropía normalizada 0.9265

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2785 | 4.135 | 0.8911 | 13.23 |
| steep5 | 0.05009 | 0.1924 | 3.842 | 0.8108 | 16.19 |
| wake | 0.15 | 0.1977 | 1.319 | 0.01677 | 0.1118 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.03; p95 0.02854; máx 1.848
- Concentración: Gini 0.5784; masa en el 1 % / 5 % superior 0.8552 / 0.9336; entropía normalizada 0.3633
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3376; con |Δu| 0.3404

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.328; solape del 5 % superior 0.4087; masa de error en el 5 % superior del residuo 0.2118; masa de residuo en el 5 % superior del error 0.8085
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 34.19 (por parámetro 0.002008); número de condición 1.403e+08

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
