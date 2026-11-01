# Ficha de diagnóstico — cohort-A-lbfgs__rbf__s303__lbfgs_final__efffa6b6

**Problema:** dfg_2d · **Campaña:** cohort-A-lbfgs · **Brazo:** rbf · **Semilla:** 303 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-rbf-current-cohort-refinement/dfg-current-cohort-refinement/run-1790606331368753261/training/seed-303/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_A (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.09785, p = 0.2922, u = 0.09205, v = 0.09165
- Error normalizado (RMS, todas las componentes): 0.3198; plano: 0.09785 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2092; masa en el 5 % superior 0.1711; entropía normalizada 0.972

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1356 | 2.013 | 0.8343 | 12.38 |
| steep5 | 0.05009 | 0.1189 | 2.374 | 0.7981 | 15.93 |
| wake | 0.15 | 0.1993 | 1.329 | 0.02745 | 0.183 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.06047; p95 0.06194; máx 3.087
- Concentración: Gini 0.5241; masa en el 1 % / 5 % superior 0.8 / 0.897; entropía normalizada 0.4423
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.164; con |Δu| 0.2097

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3124; solape del 5 % superior 0.5087; masa de error en el 5 % superior del residuo 0.1293; masa de residuo en el 5 % superior del error 0.6198
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 47.94 (por parámetro 0.01189); número de condición 7.489e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3707 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7508 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
