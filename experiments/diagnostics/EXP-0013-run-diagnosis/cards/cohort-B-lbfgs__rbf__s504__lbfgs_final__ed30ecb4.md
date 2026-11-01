# Ficha de diagnóstico — cohort-B-lbfgs__rbf__s504__lbfgs_final__ed30ecb4

**Problema:** dfg_2d · **Campaña:** cohort-B-lbfgs · **Brazo:** rbf · **Semilla:** 504 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/training/DFG-RBF-PIKAN/seed-504/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_B (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.07059, p = 0.237, u = 0.06516, v = 0.05908
- Error normalizado (RMS, todas las componentes): 0.2528; plano: 0.07059 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.21; masa en el 5 % superior 0.1626; entropía normalizada 0.9731

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1364 | 2.025 | 0.7904 | 11.73 |
| steep5 | 0.05009 | 0.1172 | 2.34 | 0.7399 | 14.77 |
| wake | 0.15 | 0.1988 | 1.326 | 0.02342 | 0.1562 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.04485; p95 0.04648; máx 2.619
- Concentración: Gini 0.4929; masa en el 1 % / 5 % superior 0.7706 / 0.8823; entropía normalizada 0.4591
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1932; con |Δu| 0.1733

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.318; solape del 5 % superior 0.5109; masa de error en el 5 % superior del residuo 0.1232; masa de residuo en el 5 % superior del error 0.6273
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 45.37 (por parámetro 0.01125); número de condición 9.003e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3634 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7378 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
