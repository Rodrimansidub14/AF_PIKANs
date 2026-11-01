# Ficha de diagnóstico — cohort-B-lbfgs__rbf__s503__lbfgs_final__4b3ef378

**Problema:** dfg_2d · **Campaña:** cohort-B-lbfgs · **Brazo:** rbf · **Semilla:** 503 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/training/DFG-RBF-PIKAN/seed-503/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_B (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0784, p = 0.224, u = 0.07421, v = 0.07369
- Error normalizado (RMS, todas las componentes): 0.2472; plano: 0.0784 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2457; masa en el 5 % superior 0.1989; entropía normalizada 0.9627

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1551 | 2.303 | 0.8265 | 12.27 |
| steep5 | 0.05009 | 0.1372 | 2.739 | 0.7829 | 15.63 |
| wake | 0.15 | 0.2039 | 1.36 | 0.02669 | 0.178 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.05299; p95 0.05298; máx 2.921
- Concentración: Gini 0.535; masa en el 1 % / 5 % superior 0.816 / 0.9063; entropía normalizada 0.4218
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1779; con |Δu| 0.2631

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4351; solape del 5 % superior 0.4457; masa de error en el 5 % superior del residuo 0.1359; masa de residuo en el 5 % superior del error 0.7394
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 48.32 (por parámetro 0.01198); número de condición 9.631e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3639 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7471 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
