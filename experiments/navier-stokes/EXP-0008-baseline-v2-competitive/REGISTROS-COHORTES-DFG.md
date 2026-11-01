# Registros de las cohortes del cilindro (DFG, Re = 20)

Este documento reemplaza el apartado «Registros de las cohortes del cilindro» del anexo de la tesis. Todas las rutas son relativas a la raíz de este repositorio.

## Catálogo

`experiments/diagnostics/EXP-0013-run-diagnosis/results/inventory.csv` (versionado desde la fase 1) identifica, por campaña, brazo, semilla y etapa, la ruta y la firma de cada estado. Permite localizar sus configuraciones originales.

## Dónde está cada cohorte

| Cohorte | Campañas | Ubicación |
|---|---|---|
| A | `cohort-A-r3r4-adam`, `cohort-A-lbfgs` | Ver `inventory.csv`. |
| B | `cohort-B-pilot-adam`, `cohort-B-lbfgs` | Ver `inventory.csv`. |
| B, continuación L-BFGS de la RBF-PIKAN | `run-1790606331368753261` | `experiments/baseline-v2/EXP-0008-baseline-v2-competitive/agent-runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-rbf-current-cohort-refinement/` |
| B, continuación L-BFGS del perceptrón | `run-1790581871213606372` | `experiments/baseline-v2/EXP-0008-baseline-v2-competitive/agent-runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-lbfgs-refinement/` |
| C | `dfg-long-20261001` | `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/results/dfg-long-20261001/` |

## Qué hay y qué no

En `agent-runs/` están los manifiestos, aprobaciones, puntuaciones y resúmenes de texto de cada campaña. Los puntos de control (`*.pt`) no se versionan. Siguen en el almacenamiento local `var/agent_runs/` del autor y se distribuyen aparte con suma de verificación, como se explica en `docs/REPRODUCIR.md`.
