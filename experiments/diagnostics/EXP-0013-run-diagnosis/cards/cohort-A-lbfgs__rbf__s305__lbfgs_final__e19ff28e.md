# Ficha de diagnóstico — cohort-A-lbfgs__rbf__s305__lbfgs_final__e19ff28e

**Problema:** dfg_2d · **Campaña:** cohort-A-lbfgs · **Brazo:** rbf · **Semilla:** 305 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-rbf-current-cohort-refinement/dfg-current-cohort-refinement/run-1790606331368753261/training/seed-305/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_A (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.09159, p = 0.3001, u = 0.08485, v = 0.08619
- Error normalizado (RMS, todas las componentes): 0.3236; plano: 0.09159 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.216; masa en el 5 % superior 0.1494; entropía normalizada 0.9731

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1248 | 1.852 | 0.7581 | 11.25 |
| steep5 | 0.05009 | 0.1032 | 2.061 | 0.6826 | 13.63 |
| wake | 0.15 | 0.224 | 1.494 | 0.03981 | 0.2655 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.04992; p95 0.06014; máx 1.971
- Concentración: Gini 0.5128; masa en el 1 % / 5 % superior 0.7188 / 0.8554; entropía normalizada 0.5568
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1384; con |Δu| 0.1614

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3221; solape del 5 % superior 0.4609; masa de error en el 5 % superior del residuo 0.1095; masa de residuo en el 5 % superior del error 0.3785
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 57.05 (por parámetro 0.01415); número de condición 2.939e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3712 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7533 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
