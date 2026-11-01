# Ficha de diagnóstico — cohort-A-lbfgs__rbf__s302__lbfgs_final__a885a499

**Problema:** dfg_2d · **Campaña:** cohort-A-lbfgs · **Brazo:** rbf · **Semilla:** 302 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-rbf-current-cohort-refinement/dfg-current-cohort-refinement/run-1790606331368753261/training/seed-302/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_A (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.08601, p = 0.2403, u = 0.08167, v = 0.07848
- Error normalizado (RMS, todas las componentes): 0.2656; plano: 0.08601 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2264; masa en el 5 % superior 0.1806; entropía normalizada 0.9675

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1463 | 2.172 | 0.7608 | 11.29 |
| steep5 | 0.05009 | 0.1249 | 2.493 | 0.702 | 14.01 |
| wake | 0.15 | 0.2034 | 1.356 | 0.03209 | 0.214 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.04814; p95 0.05702; máx 1.599
- Concentración: Gini 0.5359; masa en el 1 % / 5 % superior 0.7394 / 0.8707; entropía normalizada 0.5458
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1743; con |Δu| 0.2166

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4176; solape del 5 % superior 0.413; masa de error en el 5 % superior del residuo 0.1265; masa de residuo en el 5 % superior del error 0.3741
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 51.51 (por parámetro 0.01278); número de condición 4.334e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3733 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7519 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
