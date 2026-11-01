# Traza de experimentos

Generado por `tools/build_trace.py` desde `experiments/*/EXP-*/{spec,decision}.md`, `registry/runs.jsonl` y los notebooks. **No editar a mano.**

| EXP | Problema | Pregunta | Estado | Decisión | Bloques en el registro | Notebooks |
|---|---|---|---|---|---|---|
| EXP-0001 | heat | Baseline reproducible de PIKAN sobre Heat 1D | decided | superseded | 0 | 0 |
| EXP-0002 | heat |  | — | closed_exploratory_record | 0 | 2 |
| EXP-0003 | allen-cahn | Allen–Cahn con RBF-PIKAN | — | closed_exploratory_record | 0 | 2 |
| EXP-0004 | navier-stokes | DFG 2D-1 Re=20 con RBF-PIKAN | — | — | 0 | 0 |
| EXP-0006 | burgers | Burgers: Sobol frente a RAD | — | rejected | 0 | 0 |
| EXP-0007 | heat | Heat step-matched comparison (v5) | — | closed_exploratory_record | 0 | 0 |
| EXP-0008 | baseline-v2 | Baseline competitivo v2 (receta de optimización + capacidad + evaluadores) | decided | closed_superseded_by_baselines | 8 | 1 |
| EXP-0009 | piezo | Piezo (bimorfo PVDF, formulación indirecta): problema, referencia y baseline | decided | closed_baseline_recorded | 10 | 10 |
| EXP-0011 | heat | RAD sobre el control RBF-PIKAN de Heat (primer experimento de mecanismo) | decided | rejected | 1 | 1 |
| EXP-0012 | heat | ¿El déficit de RAD tras L-BFGS venía de los puntos de L-BFGS? (seguimiento de EXP-0011) | decided | rejected | 1 | 1 |
| EXP-0013 | diagnostics | Diagnóstico de qué hace cada corrida, para definir un mecanismo de *awareness* | decided | accepted | 3 | 0 |
| EXP-0014 | mechanisms | Muestreador autogobernado: primera iteración de un mecanismo de *awareness* | draft | — | 0 | 0 |
| EXP-0015 | mechanisms | ¿Cantidad o ubicación? Prueba de techo sobre las bases de la capa 0 | preregistered | — | 2 | 0 |
| EXP-0016 | mechanisms | Focalización guiada por el asentamiento de la representación interna | decided | — | 8 | 0 |
| EXP-0017 | mechanisms | ¿Qué parte de la representación puede guiar la focalización? Batería en lazo abierto | scored | — | 1 | 0 |
| EXP-0018 | mechanisms | Techo causal del foco por responsabilidad (A* oráculo) | scored | — | 3 | 0 |
| EXP-0022 | mechanisms | ¿Qué estructura del problema de optimización frena a la PIKAN-PDE? Diagnóstico de condicio | preregistered | — | 0 | 0 |
| EXP-0023 | mechanisms | Precondicionamiento por la estructura de la representación | preregistered | — | 5 | 0 |
| EXP-0024 | mechanisms | Túnel de viento: separar descubrir de confirmar | preregistered | — | 4 | 0 |
| EXP-0025 | mechanisms | Observador con sondeo: un controlador que elige palanca durante el entrenamiento | preregistered | — | 1 | 0 |
| EXP-0026 | mechanisms | ¿Por qué E ayuda en Burgers en un bloque y daña en otro? Cadencia × recálculo al arrancar  | preregistered | — | 1 | 0 |
| EXP-0027 | mechanisms | ¿Predice el estado en la primera intervención qué semillas dañan a E en Burgers? | — | — | 0 | 0 |
| EXP-0028 | mechanisms | ¿Qué componente del mecanismo, qué propiedad del problema y qué parte del entrenamiento ha | preregistered | accepted_exploratory | 14 | 2 |
| EXP-0029 | mechanisms | Mecanismo final M en piezo (spec y firma en EXP-0029_spec.md) | scored (spec dentro de EXP-0028) | not_supported | 2 | 0 |
| EXP-0030 | mechanisms | Caracterización de los resultados del mecanismo M en piezo | preregistered | inconclusive | 1 | 0 |

## Bloques de corridas (de `registry/runs.jsonl`)

| run_id | EXP | Fecha | Semillas | Backend | Estado | Resultados |
|---|---|---|---|---|---|---|
| ac-rerun-20260930 | EXP-0008 | 2026-09-30 | 601-605 | kaggle_T4x2_private | COMPLETE | `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/results/ac-rerun-20260930` |
| dfg-daniel-cfg-20260930 | EXP-0008 | 2026-09-30 | 801-805 | kaggle_T4x2_private | COMPLETE | `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/results/dfg-daniel-cfg-20260930` |
| heat-lbfgs-redo-20260930 | EXP-0008 | 2026-09-30 | 901-905 | kaggle_T4x2_private | COMPLETE | `experiments/heat/EXP-0008-baseline-v2-competitive/results/heat-lbfgs-redo-20260930` |
| ns2d-cg-20260930 | EXP-0008 | 2026-09-30 | 701-705 | kaggle_T4x2_private | COMPLETE | `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/results/ns2d-cg-20260930` |
| dfg-long-20261001 | EXP-0008 | 2026-10-01 | 801-805 | kaggle_T4x2_private | COMPLETE | `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/results/dfg-long-20261001` |
| piezo-long1-20261001 | EXP-0009 | 2026-10-01 | 1001 | kaggle_T4x2_private | COMPLETE | `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/long1-20261001` |
| piezo-probe-20261001 | EXP-0009 | 2026-10-01 | 1001 | kaggle_T4x2_private | COMPLETE | `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/probe-20261001` |
| piezo-vsplit-20261001 | EXP-0009 | 2026-10-01 | 1001 | kaggle_T4x2_private | COMPLETE | `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/vsplit-20261001` |
| burgers-big-20261003 | EXP-0008 | 2026-10-03 | 301-305 | kaggle_T4x2_private | COMPLETE | `experiments/burgers/EXP-0008-baseline-v2-competitive/results/burgers-big-20261003` |
| burgers-close-20261003 | EXP-0008 | 2026-10-03 | 301-305 | kaggle_T4x2_private | COMPLETE | `experiments/burgers/EXP-0008-baseline-v2-competitive/results/burgers-close-20261003` |
| burgers-f64-lbfgs-20261003 | EXP-0008 | 2026-10-03 | 301-305 | kaggle_T4x2_private | COMPLETE | `experiments/burgers/EXP-0008-baseline-v2-competitive/results/burgers-f64-lbfgs-20261003` |
| piezo-daniel-recipe-20261003 | EXP-0009 | 2026-10-03 | 20260728 | kaggle_T4x2_private | COMPLETE | `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-recipe-20261003` |
| piezo-layered-20261003 | EXP-0009 | 2026-10-03 | 1001 | kaggle_T4x2_private | COMPLETE | `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/layered-20261003` |
| piezo-rbf-lr-20261003 | EXP-0009 | 2026-10-03 | 1001 | kaggle_T4x2_private | COMPLETE | `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/rbf-lr-20261003` |
| piezo-sweep-20261003 | EXP-0009 | 2026-10-03 | 1001 | kaggle_T4x2_private | COMPLETE | `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/sweep-20261003` |
| piezo-sweep2-20261003 | EXP-0009 | 2026-10-03 | 1001 | kaggle_T4x2_private | COMPLETE | `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/sweep2-20261003` |
| heat-rad-exp0011-20261004 | EXP-0011 | 2026-10-04 | 901-905 | kaggle_T4x2_private | COMPLETE | `experiments/heat/EXP-0011-heat-rad-adam/results/run-20261004` |
| heat-rad-exp0012-20261004 | EXP-0012 | 2026-10-04 | 901-905 | kaggle_T4x2_private | COMPLETE | `experiments/heat/EXP-0012-heat-rad-uniform-lbfgs/results/run-20261004` |
| piezo-daniel-seeds-20261004 | EXP-0009 | 2026-10-04 | 20260729-20260730 | kaggle_T4x2_private | COMPLETE | `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-seeds-20261004` |
| piezo-daniel-seeds2-20261004 | EXP-0009 | 2026-10-04 | 20260731-20260732 | kaggle_T4x2_private | COMPLETE | `experiments/piezo/EXP-0009-piezo-indirect-rbf/results/daniel-seeds2-20261004` |
| exp0013-analysis-20261005 | EXP-0013 | 2026-10-05 |  | local_cpu | COMPLETE | `experiments/diagnostics/EXP-0013-run-diagnosis/results/analysis` |
| exp0013-signals-20261005 | EXP-0013 | 2026-10-05 |  | local_cpu | COMPLETE | `experiments/diagnostics/EXP-0013-run-diagnosis/results` |
| exp0013-step0-inventory-20261005 | EXP-0013 | 2026-10-05 | all | local_cpu | COMPLETE | `experiments/diagnostics/EXP-0013-run-diagnosis/results/inventory.csv` |
| exp0015-step0-seed1-20261018 | EXP-0015 | 2026-10-18 | 1 | cloud_cpu_2cores | COMPLETE | `experiments/mechanisms/EXP-0015-layer0-basis-refinement/step0/results` |
| exp0015-step0-seeds23-20261018 | EXP-0015 | 2026-10-18 | 2,3 | local_cpu_4core | COMPLETE | `experiments/mechanisms/EXP-0015-layer0-basis-refinement/step0/results` |
| exp0016-executor-20261019 | EXP-0016 | 2026-10-19 | 301, 601 (humo) | local_cpu_4core | COMPLETE | `tests/test_stall_handoff_sweep.py` |
| exp0016-precheck-bundle-20261019 | EXP-0016 | 2026-10-19 | 301, 601 | none (bundle preparado, no subido) | PREPARED | `experiments/mechanisms/EXP-0016-stall-handoff/kaggle` |
| exp0016-precheck-gpu-20261019 | EXP-0016 | 2026-10-19 | 301, 601 (receta recortada: 4000 pasos de Adam, L-BFGS 400 it.) | kaggle_t4 | COMPLETE | `experiments/mechanisms/EXP-0016-stall-handoff/results/precheck-gpu` |
| exp0016-stageA-20261019 | EXP-0016 | 2026-10-19 | 301-305, 601-603 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0016-stall-handoff/results/stage-a` |
| exp0016-stageB-20261019 | EXP-0016 | 2026-10-19 | 301-305, 601-603 (registros de la etapa A) | local_cpu_4core | COMPLETE | `experiments/mechanisms/EXP-0016-stall-handoff/results.md` |
| exp0016-tier01-20261019 | EXP-0016 | 2026-10-19 | 301-305, 601-605, 901-905 (historias archivadas) | local_cpu_4core | COMPLETE | `experiments/mechanisms/EXP-0016-stall-handoff` |
| exp0016-v2-tier01-20261019 | EXP-0016 | 2026-10-19 | 301, 601 (humo); 901-905 (vista previa) | local_cpu_4core | COMPLETE | `experiments/mechanisms/EXP-0016-stall-handoff` |
| exp0016-warmup-check-gpu-20261019 | EXP-0016 | 2026-10-19 | 301 (receta recortada: 4000 pasos de Adam, L-BFGS 400 it.) | kaggle_t4 | COMPLETE | `experiments/mechanisms/EXP-0016-stall-handoff/results/warmup-check-gpu` |
| exp0017-battery-20261020 | EXP-0017 | 2026-10-20 | 301-305, 601-603 (40 pares de estados del EXP-0016 etapa A) | local_cpu_4core | COMPLETE | `experiments/mechanisms/EXP-0017-representation-battery/results.md` |
| exp0018-ceiling-20261020 | EXP-0018 | 2026-10-20 | 301-305, 601-605 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0018-responsibility-ceiling/results.md` |
| exp0018-test7-t4-20261020 | EXP-0018 | 2026-10-20 | 301, 601 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0018-responsibility-ceiling/results/precheck7` |
| exp0018-tier01-20261020 | EXP-0018 | 2026-10-20 | 301, 601 (A* contra EXP-0017) | local_cpu_4core | COMPLETE | `experiments/mechanisms/EXP-0018-responsibility-ceiling` |
| exp0024-regrid-cpu-vs-gpu-20261020 | EXP-0024 | 2026-10-20 | 1 | kaggle_t4 + cpu | COMPLETE | `experiments/mechanisms/EXP-0024-wind-tunnel-protocol/results/regrid-check-gpu` |
| exp0024-t1-block-20261020 | EXP-0024 | 2026-10-20 | 1-5 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0024-wind-tunnel-protocol/round1-T1.md` |
| exp0024-t1-d1-diagnostic-20261020 | EXP-0024 | 2026-10-20 | 1 | kaggle_t4 + local_cpu | COMPLETE | `experiments/mechanisms/EXP-0024-wind-tunnel-protocol/results` |
| exp0024-t1-shorttest-gpu-20261020 | EXP-0024 | 2026-10-20 | 1 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0024-wind-tunnel-protocol/results/t1-shorttest-gpu` |
| exp0023-t1-block-20261021 | EXP-0023 | 2026-10-21 | 1-5 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0023-gram-preconditioning/results/t1-block` |
| exp0023-v-block-20261021 | EXP-0023 | 2026-10-21 | 101-105 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0023-gram-preconditioning/results/v-block` |
| exp0023-c1-P-20261022 | EXP-0023 | 2026-10-22 | 11-20 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0023-gram-preconditioning/results/c1-P` |
| exp0023-c1-T1-20261022 | EXP-0023 | 2026-10-22 | 6-10 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0023-gram-preconditioning/results/c1-T1` |
| exp0023-c1-V-20261022 | EXP-0023 | 2026-10-22 | 106-110 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0023-gram-preconditioning/results/c1-V` |
| exp0025-block-T1-20261022 | EXP-0025 | 2026-10-22 | 2001-2005 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0025-observer-v0/results/block-T1` |
| exp0026-diag-20261022 | EXP-0026 | 2026-10-22 | 3001-3010 (mas semillas de control) | kaggle_t4x2 | PARTIAL | `experiments/mechanisms/EXP-0026-e-burgers-diagnostic/results/exp0026` |
| exp0028-phase1a | EXP-0028 | 2026-10-24 | 4001-4010 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/phase1a` |
| exp0028-phase1b | EXP-0028 | 2026-10-24 | 4001-4010 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/phase1b` |
| exp0028-phase1c-dfg | EXP-0028 | 2026-10-24 | 4001-4010 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/phase1c-dfg` |
| exp0028-phase1c-t1 | EXP-0028 | 2026-10-24 | 4001-4010 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/phase1c-t1` |
| exp0028-phase1d-ac | EXP-0028 | 2026-10-24 | 4101-4105 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/phase1d-ac` |
| exp0028-phase1d-bu1 | EXP-0028 | 2026-10-24 | 4101-4102 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/phase1d-bu1` |
| exp0028-phase1d-kpp | EXP-0028 | 2026-10-24 | 4101-4105 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/phase1d-kpp` |
| exp0028-phase1e-ns1 | EXP-0028 | 2026-10-24 | 4101-4103 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/phase1e-ns1` |
| exp0028-phase1e-acf | EXP-0028 | 2026-10-25 | 4101-4105 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/phase1e-acf` |
| exp0028-phase1e-ns2 | EXP-0028 | 2026-10-25 | 4104-4105 | kaggle_t4x2 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/phase1e-ns2` |
| exp0028-piezo-modal-run | EXP-0028 | 2026-10-25 | 20260728-20260732 | modal_h100 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/piezo-modal-run` |
| exp0028-piezo-modal-smoke | EXP-0028 | 2026-10-25 | 20260728 | modal_h100 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/piezo-modal-smoke` |
| exp0028-piezo-s1ab | EXP-0028 | 2026-10-25 | 20260728 | kaggle_t4 | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/piezo-s1a, piezo-s1b` |
| exp0028-piezo-stage0 | EXP-0028 | 2026-10-25 | 20260728 | local_cpu + kaggle | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/piezo-stage0` |
| exp0029-piezo-S-t1t2-20261025 | EXP-0029 | 2026-10-25 | 20260728-20260730 (y 4001) | modal + kaggle | COMPLETE | `experiments/mechanisms/EXP-0028-failure-attribution/results/piezo-exp0029, piezo-exp0029-t2` |
| exp0029-piezo-final-20261026 | EXP-0029 | 2026-10-26 | 20260728-20260732 | modal_a100 | PARTIAL | `experiments/mechanisms/EXP-0028-failure-attribution/results/piezo-exp0029-final` |
| exp0030-caracterizacion-M-20261026 | EXP-0030 | 2026-10-26 | 20260728-20260732 | local_cpu | COMPLETE | `experiments/mechanisms/EXP-0030-characterization-M/results` |

## Notebooks

| Notebook | Rol (metadata.research) |
|---|---|
| `experiments/heat/EXP-0011-heat-rad-adam/kaggle/notebook.ipynb` | — |
| `experiments/heat/EXP-0012-heat-rad-uniform-lbfgs/kaggle/notebook.ipynb` | — |
| `experiments/mechanisms/EXP-0028-failure-attribution/colab/blocks_colab.ipynb` | — |
| `experiments/mechanisms/EXP-0028-failure-attribution/colab/piezo_colab.ipynb` | — |
| `experiments/piezo/EXP-0009-piezo-indirect-rbf/kaggle/piezo-daniel-recipe-20261003/notebook.ipynb` | — |
| `experiments/piezo/EXP-0009-piezo-indirect-rbf/kaggle/piezo-daniel-seeds-20261004/notebook.ipynb` | — |
| `experiments/piezo/EXP-0009-piezo-indirect-rbf/kaggle/piezo-daniel-seeds2-20261004/notebook.ipynb` | — |
| `experiments/piezo/EXP-0009-piezo-indirect-rbf/kaggle/piezo-layered-20261003/notebook.ipynb` | — |
| `experiments/piezo/EXP-0009-piezo-indirect-rbf/kaggle/piezo-long1-20261001/notebook.ipynb` | — |
| `experiments/piezo/EXP-0009-piezo-indirect-rbf/kaggle/piezo-probe-20261001/notebook.ipynb` | — |
| `experiments/piezo/EXP-0009-piezo-indirect-rbf/kaggle/piezo-rbf-lr-20261003/notebook.ipynb` | — |
| `experiments/piezo/EXP-0009-piezo-indirect-rbf/kaggle/piezo-sweep-20261003/notebook.ipynb` | — |
| `experiments/piezo/EXP-0009-piezo-indirect-rbf/kaggle/piezo-sweep2-20261003/notebook.ipynb` | — |
| `experiments/piezo/EXP-0009-piezo-indirect-rbf/kaggle/piezo-vsplit-20261001/notebook.ipynb` | — |
| `notebooks/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/allen-cahn-rbf-sobol-exploratory.ipynb` | exploration |
| `notebooks/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/allen-cahn-rbf-sobol-results.ipynb` | exploration |
| `notebooks/baseline-v2/EXP-0008-baseline-v2-competitive/exp0008-baseline-v2-fast-track-signals.ipynb` | — |
| `notebooks/heat/EXP-0002-heat-burgers-sobol/heat-burgers-baseline-t4x2-results.ipynb` | exploration |
| `notebooks/heat/EXP-0002-heat-burgers-sobol/heat-burgers-baseline-t4x2.ipynb` | exploration |
| `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-bspline_cubic-bundle-logscale-v1-results.ipynb` | exploration |
| `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-bspline_cubic-bundle-logscale-v1.ipynb` | exploration |
| `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-bspline_cubic.ipynb` | exploration |
| `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-comparison.ipynb` | exploration |
| `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-mlp_tanh-bundle-logscale-v1-results.ipynb` | exploration |
| `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-mlp_tanh-bundle-logscale-v1.ipynb` | exploration |
| `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-mlp_tanh.ipynb` | exploration |
| `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-rbf_fixed-bundle-logscale-v1-results.ipynb` | exploration |
| `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-rbf_fixed-bundle-logscale-v1.ipynb` | exploration |
| `notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-rbf_fixed.ipynb` | exploration |
| `notebooks/warmup/heat/heat-pikan-copia-1.ipynb` | — |
| `notebooks/warmup/heat/heat-pikan.ipynb` | — |
| `notebooks/warmup/heat/heat-pinn-2.ipynb` | — |
| `notebooks/warmup/navier-stokes/ns-kan-variants.ipynb` | — |
| `notebooks/warmup/navier-stokes/ns-pikans-base-pre-presentation-cleanup-20260506.ipynb` | — |
| `notebooks/warmup/navier-stokes/ns-pikans-base.ipynb` | — |
| `notebooks/warmup/navier-stokes/ns-pikans-geo-baseline-pre-pipeline-refactor-20260505.ipynb` | — |
| `notebooks/warmup/navier-stokes/ns-pikans-geo-baseline.ipynb` | — |
| `notebooks/warmup/navier-stokes/ns-pikans-geo-copia-1.ipynb` | — |
| `notebooks/warmup/navier-stokes/ns-pinn-geo-copia-colab.ipynb` | — |
