# Ficha de diagnóstico — cohort-B-pilot-adam__mlp__s504__adam_final__a00919b0

**Problema:** dfg_2d · **Campaña:** cohort-B-pilot-adam · **Brazo:** mlp · **Semilla:** 504 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch-competitive-pilots/dfg/dfg-re20-t4x2/campaign/DFG-Tanh-MLP/seed-504/checkpoint-final.pt` · **Familia de modelo:** dfg_cohort_B (mlp) · **Parámetros:** 17027 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1943, p = 0.4407, u = 0.1888, v = 0.1403
- Error normalizado (RMS, todas las componentes): 0.4995; plano: 0.1943 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2472; masa en el 5 % superior 0.2657; entropía normalizada 0.9502

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.184 | 2.731 | 0.6083 | 9.029 |
| steep5 | 0.05009 | 0.1724 | 3.441 | 0.6135 | 12.25 |
| wake | 0.15 | 0.1699 | 1.133 | 0.009566 | 0.06379 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.05278; p95 0.1028; máx 1.576
- Concentración: Gini 0.5884; masa en el 1 % / 5 % superior 0.5677 / 0.83; entropía normalizada 0.6452
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3143; con |Δu| 0.2192

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.248; solape del 5 % superior 0.5891; masa de error en el 5 % superior del residuo 0.222; masa de residuo en el 5 % superior del error 0.614
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 30.82 (por parámetro 0.00181); número de condición 1.009e+08

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
