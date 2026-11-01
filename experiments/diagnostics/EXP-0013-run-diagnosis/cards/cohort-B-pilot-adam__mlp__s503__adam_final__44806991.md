# Ficha de diagnóstico — cohort-B-pilot-adam__mlp__s503__adam_final__44806991

**Problema:** dfg_2d · **Campaña:** cohort-B-pilot-adam · **Brazo:** mlp · **Semilla:** 503 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch-competitive-pilots/dfg/dfg-re20-t4x2/campaign/DFG-Tanh-MLP/seed-503/checkpoint-final.pt` · **Familia de modelo:** dfg_cohort_B (mlp) · **Parámetros:** 17027 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.7944, p = 1.05, u = 0.7908, v = 0.7085
- Error normalizado (RMS, todas las componentes): 1.493; plano: 0.7944 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2893; masa en el 5 % superior 0.3658; entropía normalizada 0.9168

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.299 | 4.438 | 0.5051 | 7.498 |
| steep5 | 0.05009 | 0.2863 | 5.715 | 0.4544 | 9.072 |
| wake | 0.15 | 0.1427 | 0.9514 | 0.005486 | 0.03659 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.2; p95 0.3818; máx 1.513
- Concentración: Gini 0.7602; masa en el 1 % / 5 % superior 0.3584 / 0.85; entropía normalizada 0.7093
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4521; con |Δu| 0.5389

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5001; solape del 5 % superior 0.8022; masa de error en el 5 % superior del residuo 0.3506; masa de residuo en el 5 % superior del error 0.8197
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 17.52 (por parámetro 0.001029); número de condición 6.524e+08

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, residual_aligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
