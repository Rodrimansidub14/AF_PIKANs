# EXP-0028 — atribución de fallos (guía de lectura)

Esta carpeta conserva los archivos con los nombres con que se citan en `registry/` y en las decisiones, por eso no se movieron. Este índice dice qué es cada cosa y en qué orden leerla.

## Lo firmado

| Archivo | Qué es |
|---|---|
| `spec.md` | Pre-registro v1.1 (firmado el 2026-10-23). Hipótesis, brazos J, F, E, I frente a B, y reglas. |
| `decision.md` | Decisión de EXP-0028 (`accepted_exploratory`). |
| `EXP-0029_spec.md`, `decision-EXP-0029.md` | Spec y decisión de EXP-0029, que vive dentro de esta carpeta (`not_supported`). |
| `predictions.md` | Predicciones para el bimorfo (H-PRED-1), firmadas el 2026-10-25. |

## Por fase de ejecución

| Fase | Presupuesto | Lanzador | Análisis | Notas |
|---|---|---|---|---|
| 1 | `budget.md`, `budget_vector.json`, `measure_budget_vector.py` | `run_phase1.py`, `kaggle_block.py` | `analyze_phase1.py` | `missing_arms.md` |
| 1c | `budget_1c.md` | `run_phase1c.py`, `kaggle_block_1c.py` | `analyze_phase1c.py` | `missing_arms_1c.md` |
| 1c DFG | | `run_phase1c_dfg.py`, `kaggle_block_1c_dfg.py` | `analyze_phase1c_dfg.py` | `dfg_blockers.md` |
| 1d | `budget_1d.md` | `run_phase1d.py`, `kaggle_block_1d.py` | `analyze_phase1d.py` | |
| 1e | `budget_1e.md`, `budget_1e_calc.py` | `run_phase1e.py`, `kaggle_block_1e.py` | `analyze_phase1e.py` | |
| rebanada | `budget_slice.md` | | | `artifact_slice/` |

Los datos de cada fase están en `results/` y se registran en `registry/runs.jsonl`. Los paquetes enviados a Kaggle y a Colab están en `kaggle/` y `colab/`.

## Diseño del mecanismo y análisis de piezo (de más viejo a más nuevo)

1. `DISENO_MECANISMO_FINAL.md`: primer borrador, **superado** (lo dice su cabecera).
2. `ANALISIS_PIEZO_PROPIEDADES.md`: análisis de propiedades con las 24 corridas de piezo, sin cómputo nuevo.
3. `REVISION_LITERATURA_PIEZO.md`: revisión de literatura y de la matemática de la PDE.
4. `HIPOTESIS_PARTES_QUE_FALTAN.md`: hipótesis de trabajo, ninguna probada.
5. `DISENO_ARANA.md`: propuesta «araña».
6. `DISENO_FINAL.md`: propuesta que sustituye a la anterior en lo que dice sobre pesos.
7. `DISENO_EJECUCION.html`: versión visual del diseño de ejecución.
8. `LECCIONES_EXP0029.md`: lo aprendido de EXP-0029.

## Otras carpetas

- `fase0/`: verificación de reproducibilidad de B y mapas de palancas por problema.
- `fase1c_cpu/`, `fase1c_dfg_cpu/`: verificaciones y tiempos en CPU de NS y DFG.
- `fase2/`: scripts que miden propiedades y arman las tablas.
- `piezo/`: documentos de procedencia, alineación con el artículo y plan de recursos del bimorfo, y el kit de la nube.
- `analisis_piezo/`, `cruce/`: análisis de las corridas de piezo y su cruce con propiedades.
- `artifact_slice/`: rebanada de artefactos (scripts de análisis y de bloque).
- `wandb_tools/`: reconstrucción de corridas en W&B desde JSON.
- `_borradores/`: borradores del agente, **no son decisiones**.
