# Ficha de diagnóstico — cohort-B-lbfgs__rbf__s502__lbfgs_final__d34e7023

**Problema:** dfg_2d · **Campaña:** cohort-B-lbfgs · **Brazo:** rbf · **Semilla:** 502 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/training/DFG-RBF-PIKAN/seed-502/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_B (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.07726, p = 0.2291, u = 0.07276, v = 0.07186
- Error normalizado (RMS, todas las componentes): 0.2509; plano: 0.07726 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2106; masa en el 5 % superior 0.1793; entropía normalizada 0.9711

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1347 | 1.999 | 0.7774 | 11.54 |
| steep5 | 0.05009 | 0.1181 | 2.358 | 0.7234 | 14.44 |
| wake | 0.15 | 0.1987 | 1.325 | 0.02229 | 0.1487 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.04852; p95 0.05419; máx 2.291
- Concentración: Gini 0.524; masa en el 1 % / 5 % superior 0.7835 / 0.89; entropía normalizada 0.4805
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1853; con |Δu| 0.1293

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.268; solape del 5 % superior 0.437; masa de error en el 5 % superior del residuo 0.1238; masa de residuo en el 5 % superior del error 0.5841
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 51.84 (por parámetro 0.01286); número de condición 4.789e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3594 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7457 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
