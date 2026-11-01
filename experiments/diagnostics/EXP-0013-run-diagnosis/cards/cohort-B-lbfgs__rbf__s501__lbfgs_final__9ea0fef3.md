# Ficha de diagnóstico — cohort-B-lbfgs__rbf__s501__lbfgs_final__9ea0fef3

**Problema:** dfg_2d · **Campaña:** cohort-B-lbfgs · **Brazo:** rbf · **Semilla:** 501 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/training/DFG-RBF-PIKAN/seed-501/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_B (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0765, p = 0.2728, u = 0.06968, v = 0.06956
- Error normalizado (RMS, todas las componentes): 0.29; plano: 0.0765 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.193; masa en el 5 % superior 0.1416; entropía normalizada 0.9776

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1135 | 1.685 | 0.7573 | 11.24 |
| steep5 | 0.05009 | 0.09772 | 1.951 | 0.6705 | 13.39 |
| wake | 0.15 | 0.2052 | 1.368 | 0.03293 | 0.2196 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.04354; p95 0.0549; máx 1.956
- Concentración: Gini 0.5032; masa en el 1 % / 5 % superior 0.6941 / 0.8479; entropía normalizada 0.5558
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1892; con |Δu| 0.1896

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2796; solape del 5 % superior 0.4261; masa de error en el 5 % superior del residuo 0.09867; masa de residuo en el 5 % superior del error 0.5221
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 54 (por parámetro 0.01339); número de condición 4.825e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3683 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7443 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
