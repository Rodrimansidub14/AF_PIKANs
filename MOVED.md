# MOVED — rutas viejas → nuevas

## Paso 1 (2026-10-01): experiments/ y notebooks/

| Antes | Ahora |
|---|---|
| `experiments/EXP-0001-heat1d-baseline` | `experiments/heat/EXP-0001-heat1d-baseline` |
| `experiments/EXP-0007-heat-stepmatched-deepxde` | `experiments/heat/EXP-0007-heat-stepmatched-deepxde` |
| `experiments/EXP-0002-heat-burgers-sobol` | `experiments/heat/EXP-0002-heat-burgers-sobol` |
| `experiments/EXP-0006-burgers-sobol-rad` | `experiments/burgers/EXP-0006-burgers-sobol-rad` |
| `experiments/EXP-0003-allen-cahn-rbf-sobol` | `experiments/allen-cahn/EXP-0003-allen-cahn-rbf-sobol` |
| `experiments/EXP-0004-dfg-re20-rbf-sobol` | `experiments/navier-stokes/EXP-0004-dfg-re20-rbf-sobol` |
| `experiments/EXP-0008-baseline-v2-competitive` | `experiments/baseline-v2/EXP-0008-baseline-v2-competitive` |
| `notebooks/experiments/allen-cahn-rbf-sobol-exploratory.ipynb` | `notebooks/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/allen-cahn-rbf-sobol-exploratory.ipynb` |
| `notebooks/experiments/allen-cahn-rbf-sobol-results.ipynb` | `notebooks/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/allen-cahn-rbf-sobol-results.ipynb` |
| `notebooks/experiments/exp0008-baseline-v2-fast-track-signals.ipynb` | `notebooks/baseline-v2/EXP-0008-baseline-v2-competitive/exp0008-baseline-v2-fast-track-signals.ipynb` |
| `notebooks/experiments/heat-burgers-architecture-screen-bspline_cubic-bundle-logscale-v1-results.ipynb` | `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-bspline_cubic-bundle-logscale-v1-results.ipynb` |
| `notebooks/experiments/heat-burgers-architecture-screen-bspline_cubic-bundle-logscale-v1.ipynb` | `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-bspline_cubic-bundle-logscale-v1.ipynb` |
| `notebooks/experiments/heat-burgers-architecture-screen-bspline_cubic.ipynb` | `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-bspline_cubic.ipynb` |
| `notebooks/experiments/heat-burgers-architecture-screen-comparison.ipynb` | `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-comparison.ipynb` |
| `notebooks/experiments/heat-burgers-architecture-screen-mlp_tanh-bundle-logscale-v1-results.ipynb` | `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-mlp_tanh-bundle-logscale-v1-results.ipynb` |
| `notebooks/experiments/heat-burgers-architecture-screen-mlp_tanh-bundle-logscale-v1.ipynb` | `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-mlp_tanh-bundle-logscale-v1.ipynb` |
| `notebooks/experiments/heat-burgers-architecture-screen-mlp_tanh.ipynb` | `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-mlp_tanh.ipynb` |
| `notebooks/experiments/heat-burgers-architecture-screen-rbf_fixed-bundle-logscale-v1-results.ipynb` | `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-rbf_fixed-bundle-logscale-v1-results.ipynb` |
| `notebooks/experiments/heat-burgers-architecture-screen-rbf_fixed-bundle-logscale-v1.ipynb` | `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-rbf_fixed-bundle-logscale-v1.ipynb` |
| `notebooks/experiments/heat-burgers-architecture-screen-rbf_fixed.ipynb` | `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-rbf_fixed.ipynb` |
| `notebooks/experiments/heat-burgers-baseline-t4x2-results.ipynb` | `notebooks/heat/EXP-0002-heat-burgers-sobol/heat-burgers-baseline-t4x2-results.ipynb` |
| `notebooks/experiments/heat-burgers-baseline-t4x2.ipynb` | `notebooks/heat/EXP-0002-heat-burgers-sobol/heat-burgers-baseline-t4x2.ipynb` |
| `notebooks/PIKAN-Warmup` | `notebooks/warmup` |

Referencias de texto actualizadas (paso 1): `tests/test_allen_cahn_campaign.py`, `experiments/.../build_bundle.py` (ROOT = parents[3]), `manifest.json` de EXP-0003, `signals-and-records.md` y `spec.md` de EXP-0008, y menciones en `docs/`.

**Desviación respecto a la propuesta:** los notebooks NO viven dentro de `experiments/`, porque `context/research-ontology.toml` (protegido) fija `notebooks = "notebooks"` y el auditor de notebooks recorre esa carpeta. Se espeja la estructura: `notebooks/<problema>/<EXP>/`.

## Paso 2 (2026-10-01): docs/ y raíz

| Antes | Ahora |
|---|---|
| `docs/brief-avance-investigacion-2026-09-23.docx` | `docs/briefs/brief-avance-investigacion-2026-09-23.docx` |
| `docs/brief-avance-investigacion-2026-09-24.docx` | `docs/briefs/brief-avance-investigacion-2026-09-24.docx` |
| `docs/campaign-plan-exploration-to-sota.md` | `docs/plans/campaign-plan-exploration-to-sota.md` |
| `docs/sota-readiness-roadmap.md` | `docs/roadmaps/sota-readiness-roadmap.md` |
| `docs/analysis-dashboard.md` | `docs/reference/analysis-dashboard.md` |
| `docs/rbf-signal-baseline-sota.md` | `docs/reference/rbf-signal-baseline-sota.md` |
| `wandb/` | `var/scratch/wandb/` |
| `PINN/`, `Praticas/` (vacíos) | eliminados |
| `test_localsystem.txt` ("I have run") | eliminado |
| `tmp/` | se queda: `src/harness_loop/run_experiment.py` lo usa como salida por defecto (`tmp/harness_runs`) |

## Paso 3 (2026-10-01): resultados promovidos (copiados, `var/agent_runs/` queda como registro crudo)

Campañas ac-rerun, ns2d-cg, dfg-daniel-cfg, dfg-long, heat-lbfgs-redo → `experiments/<problema>/EXP-0008-baseline-v2-competitive/{results,checkpoints}/<campaña>/` (checkpoints con `index.json` sha256). Aún no hay figuras: se generan en la fase de análisis.

## Limpieza de var/agent_runs/ (2026-10-03, aprobada por el investigador)

Eliminados (salidas desechables de pytest, renders intermedios de los briefs, directorios temporales): pytest-afpikans-tests pytest-afpikans-tests-final pytest-dfg-dtype-red pytest-dfg-sciml pytest-dfg-sciml-final pytest-dfg-sciml-regression brief-avance-20260924-render brief-avance-20260924-render2 brief-avance-20260924-render3 brief-avance-20260924-render4 setup-tmp tests test-dfg-final-20260927 test-dfg-full-20260927 test-dfg-sciml-20260927 test-dfg-sciml-20260927-green. Se conservan `brief-avance-20260924-render5`, los entornos (`kaggle-cli-env`, `sciml-env`, `uv-cache`) y todas las campañas.

## Paso 2 (2026-10-26): nombres de notebooks de calentamiento

Todos pasan a minúsculas con guiones (convención en `specs/conventions/convencion-nombres.md`). `notebooks/warmup/tora_B2C.pdf` pasa a `docs/reference/` (no es un notebook).

| Antes | Ahora |
|---|---|
| `notebooks/warmup/HEAT/heat_PIKAN (1).ipynb` | `notebooks/warmup/heat/heat-pikan-copia-1.ipynb` |
| `notebooks/warmup/HEAT/heat_PIKAN.ipynb` | `notebooks/warmup/heat/heat-pikan.ipynb` |
| `notebooks/warmup/HEAT/heat_PINN2.ipynb` | `notebooks/warmup/heat/heat-pinn-2.ipynb` |
| `notebooks/warmup/NAVIER STOKES/Copy_of_NS_PINN_geo (1).ipynb` | `notebooks/warmup/navier-stokes/ns-pinn-geo-copia-colab.ipynb` |
| `notebooks/warmup/NAVIER STOKES/NS_KAN_variants.ipynb` | `notebooks/warmup/navier-stokes/ns-kan-variants.ipynb` |
| `notebooks/warmup/NAVIER STOKES/NS_PIKANS_BASE.ipynb` | `notebooks/warmup/navier-stokes/ns-pikans-base.ipynb` |
| `notebooks/warmup/NAVIER STOKES/NS_PIKANS_BASE_pre_presentation_cleanup_20260506.ipynb` | `notebooks/warmup/navier-stokes/ns-pikans-base-pre-presentation-cleanup-20260506.ipynb` |
| `notebooks/warmup/NAVIER STOKES/NS_PIKANS_geo_(1).ipynb` | `notebooks/warmup/navier-stokes/ns-pikans-geo-copia-1.ipynb` |
| `notebooks/warmup/NAVIER STOKES/NS_PIKANS_geo_BASELINE.ipynb` | `notebooks/warmup/navier-stokes/ns-pikans-geo-baseline.ipynb` |
| `notebooks/warmup/NAVIER STOKES/NS_PIKANS_geo_BASELINE_pre_pipeline_refactor_20260505.ipynb` | `notebooks/warmup/navier-stokes/ns-pikans-geo-baseline-pre-pipeline-refactor-20260505.ipynb` |
