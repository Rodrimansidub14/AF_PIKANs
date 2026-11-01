# Ficha de diagnóstico — cohort-B-pilot-adam__mlp__s505__adam_final__c86c326f

**Problema:** dfg_2d · **Campaña:** cohort-B-pilot-adam · **Brazo:** mlp · **Semilla:** 505 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch-competitive-pilots/dfg/dfg-re20-t4x2/campaign/DFG-Tanh-MLP/seed-505/checkpoint-final.pt` · **Familia de modelo:** dfg_cohort_B (mlp) · **Parámetros:** 17027 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1883, p = 0.3941, u = 0.1839, v = 0.1415
- Error normalizado (RMS, todas las componentes): 0.4574; plano: 0.1883 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2517; masa en el 5 % superior 0.2824; entropía normalizada 0.946

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1919 | 2.848 | 0.5627 | 8.352 |
| steep5 | 0.05009 | 0.1803 | 3.599 | 0.5353 | 10.69 |
| wake | 0.15 | 0.1628 | 1.086 | 0.009988 | 0.06661 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0513; p95 0.0927; máx 1.148
- Concentración: Gini 0.6305; masa en el 1 % / 5 % superior 0.5512 / 0.848; entropía normalizada 0.664
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3221; con |Δu| 0.3052

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.384; solape del 5 % superior 0.663; masa de error en el 5 % superior del residuo 0.2385; masa de residuo en el 5 % superior del error 0.5763
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 32.3 (por parámetro 0.001897); número de condición 2.655e+08

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
