# Progreso de cierre competitivo — 2026-09-28

Registro recuperable del plan `plan-closure-20260928.md`. `scientific_accepted=false` en todo momento.

## Inicio y preflight

- La sesión anterior fue interrumpida por reinicio del PC cerca de 18:15Z antes de lanzar kernels. Se reutilizan el venv de Kaggle, `tools/kaggle_exec.py`, `s1/kernel-metadata.json`, los bundles y las salidas ya descargadas.
- `kaggle kernels list --mine` ejecutado con `tools/kaggle_exec.py` el 2026-09-28 antes de cualquier push. No aparece ningún kernel de S1/S2/S3; por tanto no hay sesión del plan que retomar en Kaggle.
- `kaggle kernels quota` no existe en el CLI 2.2.4. Consulta correcta: `kaggle quota`.
- Cuota previa a S1: 14.10 h usadas / 15.90 h disponibles de 30.00 h; refresco 2026-10-03T00:00:00Z. Cap S1=0.50 h; saldo proyectado=15.40 h (reserva mínima 12.00 h satisfecha).
- No se ha hecho push ni iniciado cómputo de este cierre. No se registran credenciales.

## Sesiones

| Orden | Slug / URL | Push UTC | Cap | Estado | Duración real | Descarga |
|---|---|---:|---:|---|---:|---|
| S1 | `rodrimandub/af-pikans-competitive-baseline-s1-20260928` | Kaggle listed 2026-09-28 19:23:33.963 UTC; started 19:23:38 UTC | 0.50 h | PARTIAL (`s1a=candidate_complete`; DFG worker launcher failed) | 432.335 s | downloaded 19:33:55 UTC to `kaggle-outputs/closure-s1-20260928/` |
| S1b-retry | `rodrimandub/af-pikans-competitive-baseline-s1b-retry-20260928` | CLI push start 2026-09-28 19:41:23.301 UTC; returned 19:41:27.281 UTC | 0.50 h (infrastructure reserve) | FAILED before compute (base64 import missing) | 7.51 s | downloaded 19:42:37 UTC to kaggle-outputs/closure-s1b-retry-20260928/ |
| S1b-retry2 | `rodrimandub/af-pikans-competitive-baseline-s1b-retry2-20260928` | CLI push start 2026-09-28 19:46:11.820 UTC; returned 19:46:15.253 UTC | 0.50 h (remaining infrastructure reserve) | PARTIAL (`gmsh` Python module unavailable; no FEM levels) | 86.563 s | downloaded to `kaggle-outputs/closure-s1b-retry2-20260928/` |
| S2 | `rodrimandub/af-pikans-heat-step-matched-s2-20260928` | 2026-09-28 20:33:09 UTC | 1.25 h | PARTIAL (5/10 Adam targets; 10/10 scored) | 3472.068 s | downloaded 2026-09-28 21:33:12 UTC to `kaggle-outputs/closure-s2-20260928/` |
| S3-attempt1 | `rodrimandub/af-pikans-ac-lbfgs-continuation-s3-20260928` | 2026-09-28 21:35:24 UTC | stage allowance 2700 s | PARTIAL (worker config KeyError; no training/scoring) | 25.359 s | downloaded 2026-09-28 21:38:15 UTC to `kaggle-outputs/closure-s3-20260928/` |

## Cambios de estado

- 2026-09-28 preflight: `kernels list --mine` confirma que no hay kernel del plan ya lanzado.
- 2026-09-28 preflight: `quota` confirma 15.90 h disponibles; `kernels quota` fue un comando no reconocido y no afectó la cuota.
- 2026-09-28 preparación S1: ETDRK4, worker FEM Taylor-Hood, rescoring y manifiesto quedaron listos localmente en `s1/`; no se ejecutó código local ni se hizo push.
- 2026-09-28 19:20:51–19:21:18 UTC S1 push attempt: Kaggle returned `400 Client Error: Bad Request ... SaveKernel` with “Your kernel title does not resolve to the specified id.” No kernel was created and no GPU quota was consumed.
- 2026-09-28 19:21:18 UTC: title correction required; recheck `kernels list --mine` and quota before the next push.

- 2026-09-28 19:25:02 UTC S1 status changed to RUNNING. Kaggle list lastRunTime=2026-09-28 19:23:33.963000 UTC; use as the server-confirmed launch timestamp because the CLI response capture was interrupted by the local progress-row format mismatch. Cap 0.50 h; quota before launch 15.90 h; no credentials recorded.

- 2026-09-28 19:31:43 UTC S1 status changed to COMPLETE. Updating ledger before output download; cap 0.50 h, launch timestamp 19:23:33.963 UTC; actual runtime will be read from artifact metadata.

- 2026-09-28 19:33:55 UTC S1 output bundle downloaded to kaggle-outputs/closure-s1-20260928/ (19 files, including S1a/DFG JSON and checkpoints).

- 2026-09-28 19:34:55 UTC S1 artifact metadata records actual_session_seconds=432.335 s; s1a candidate_complete, scientific_accepted=false. S1b status no_worker_record: supervisor could not find /kaggle/src/closure_s1.py; no FEM reference was run. Do not accept or promote reference.

- 2026-09-28 19:41:27.281 UTC S1b infrastructure retry push returned exit=0; refreshed quota before push: 15.78 h available, projected 15.28 h after 0.50 h cap (>=12.00 h). Retry bundle s1b-retry/ stages solver and mesh under Kaggle working and skips S1a.

- 2026-09-28 19:41:58 UTC S1b retry status changed from push-accepted to RUNNING; no duplicate kernel launched. Cap 0.50 h, quota before launch 15.78 h (projected after cap 15.28 h).

- 2026-09-28 19:43:13 UTC S1b retry status changed to COMPLETE. Recording before output download; artifact runtime pending.

- 2026-09-28 19:43:37 UTC S1b retry output bundle downloaded to kaggle-outputs/closure-s1b-retry-20260928/ before artifact inspection.

- 2026-09-28 19:44:13 UTC S1b retry artifact: 7.51 s actual; source staging stopped with NameError: name 'base64' is not defined before running micromamba or FEM. Output bundle downloaded to kaggle-outputs/closure-s1b-retry-20260928/. No scientific output.

- 2026-09-28 19:46:15.253 UTC S1b retry2 push returned exit=0; refreshed quota before push: 15.78 h available, projected 15.28 h after 0.50 h cap (>=12.00 h). Fixed missing base64 import; no S1a rerun.

- 2026-09-28 19:46:41 UTC S1b retry2 status changed to RUNNING; no duplicate kernel launched. Cap 0.50 h, quota before launch 15.78 h (projected after cap 15.28 h).

- 2026-09-28 19:48:50 UTC S1b retry2 status changed to COMPLETE. Recording before output download; runtime pending from artifact metadata.

- 2026-09-28 19:49:17 UTC S1b retry2 bundle downloaded to kaggle-outputs/closure-s1b-retry2-20260928/ before artifact inspection.

- 2026-09-28 19:50:28 UTC S1b retry2 artifacts: actual_session_seconds=86.563; micromamba created DOLFINx 0.11/Gmsh 4.15 environment, then solve_dfg.py failed before FEM with ModuleNotFoundError: No module named 'gmsh' during environment version collection. No levels or DFG scores; reference_accepted=false.
- S2 push slug=rodrimandub/af-pikans-heat-step-matched-s2-20260928; start_utc=2026-09-28T20:33:09.479Z; returned_utc=2026-09-28T20:33:13.117Z; cap=4500s (1.25h); quota_before=15.75h; projected_after_cap=14.50h; exit=0.
- S2 status changed to RUNNING at 2026-09-28T20:33:47.519Z; no duplicate launch; cap=4500s; push slug=rodrimandub/af-pikans-heat-step-matched-s2-20260928.
- 2026-09-28: repaired six missing line breaks between trailing current-cohort DFG scorecard rows; replaced one unreadable +/- character. Existing rows preserved.
- 2026-09-28: S3 T4x2 package prepared while S2 ran; it reuses S1a ETDRK4 plus the existing AC checkpoints/pools and verifies their source hashes remotely. Pending S2 completion; not pushed.

- S2 status changed to COMPLETE at 2026-09-28T21:31:55Z; output download pending; cap=4500s.
- S2 output bundle downloaded at 2026-09-28T21:33:12Z to kaggle-outputs/closure-s2-20260928/; artifact runtime inspection pending.

- S2 downloaded manifest: Kaggle status=partial; actual_session_seconds=3472.068; all 10 checkpoints scored; only 5/10 reached 20,000 Adam updates (all B-spline runs stopped at 7,777-7,834 updates under the 680s per-run cap). No replacement run.
- S3 push slug=rodrimandub/af-pikans-ac-lbfgs-continuation-s3-20260928; start_utc=2026-09-28T21:35:24.437Z; returned_utc=2026-09-28T21:35:27.911Z; cap=2700s (0.75h); quota check skipped per explicit researcher override; exit=0.

- S3 status changed to RUNNING at 2026-09-28T21:35:41Z; no duplicate launch. Cap=2700s. Quota check skipped per explicit researcher instruction.

- S3 status changed to COMPLETE at 2026-09-28T21:37:33Z; output download pending; inspect worker/preflight status before any decision.
- S3 output bundle downloaded at 2026-09-28T21:38:15Z to kaggle-outputs/closure-s3-20260928/; manifest/runtime inspection pending.

- S3 attempt 1 output: actual_session_seconds=25.359; source hashes all matched, but workers exited on KeyError config["seeds"] before training. No score or continuation checkpoint; using only remaining S3 stage cap on a corrected retry.
| S3-retry | `rodrimandub/af-pikans-ac-lbfgs-continuation-s3-retry-20260928` | 2026-09-28 21:43:29 UTC | 2674.641 s (remaining S3 stage cap) | COMPLETE; outputs downloaded | 1276.378 s | downloaded 2026-09-28 22:07:33 UTC to `kaggle-outputs/closure-s3-retry-20260928/` |

- S3 retry package corrected: worker config now carries seeds/arms; per-run cap remains 240 s; kernel session cap=2674.641 s so S3-attempt1 (25.359 s) + retry cap stays within the original 2700 s S3 allowance. No quota query.
- S3-retry push slug=rodrimandub/af-pikans-ac-lbfgs-continuation-s3-retry-20260928; start_utc=2026-09-28T21:43:29.254Z; returned_utc=2026-09-28T21:43:33.060Z; cap=2674.641s (S3 cumulative allowance remains 2700s); quota check skipped per explicit researcher override; exit=0.

- S3-retry status changed to RUNNING at 2026-09-28T21:43:49Z; cap=2674.641s; no quota query per explicit researcher instruction.
- Scorecard updated: appended S1a reference/convergence/rescoring rows, S1b no-field failure row, and S2 RBF/B-spline/paired-difference rows; previous rows preserved.

- S3-retry status changed to COMPLETE at 2026-09-28T22:06:24Z; output download pending; cap=2674.641s.

- S3-retry output bundle downloaded at 2026-09-28T22:07:33Z to kaggle-outputs/closure-s3-retry-20260928/; artifact runtime/results inspection pending.

- S3-retry artifact inspection: COMPLETE; actual_session_seconds=1276.378235803 on 2x Tesla T4; all 10/10 arm-seed continuations reached their per-run cap and all 10/10 scores succeeded. Output manifest SHA256=bbc9ed81d1b4c44fcce57849e4c878881c2a6a62511014e220e96f9518a169fb; download checkpoint recorded 2026-09-28T22:07:33Z. scientific_accepted=false.

- S3 score aggregation recorded in scorecard-current.csv: RBF and MLP means/sample SDs (n=5) plus paired RBF-minus-MLP differences for FD8192 and ETDRK4; closure-gate-evidence.md and competitive-baseline-execution.md now report completion and retain screening_only / scientific_accepted=false.
- Cuota final: no consultada por instruccion explicita del investigador. Ultimo valor verificado antes de S2=15.75 h disponibles; consumo real S2+S3=4,773.805 s (1.326057 h); estimacion posterior=14.423943 h si no hubo otro consumo de cuenta; no verificada por CLI.

## Follow-up 2026-09-29

- Researcher approved closure gaps 2 (DFG FEM) and 3 (Heat B-spline) together. No quota command will be run; planning estimate supplied by researcher: 14.42 h available. Follow-up hard cap=8,280 session seconds (2.30 h), split DFG=1,800 s and Heat<=6,480 s.
- Kaggle `kernels list --mine` via tools/kaggle_exec.py completed before new pushes; no kernel from this follow-up is present. Existing completed S1/S1b/S2/S3 kernels remain listed; none is being relaunched.
- Preflight snapshot at 2026-09-29 06:28 UTC: DFG and Heat had not yet been pushed at that point; subsequent launch, status, and download records follow below. The existing S2 row above records PARTIAL (5/10 Adam targets; 10/10 scored; 3472.068 s). `scientific_accepted=false`.
- 2026-09-29 06:28:01 UTC: Follow-up packages prepared for static pre-push review. DFG cap=1800s (worker cap=1600s), fixed 221x42 grid/outlet gauge is in the pre-scoring and final manifest; removed the Python `gmsh` import probe and kept Gmsh binary mesh generation plus DOLFINx `.msh` loading. Heat cap=6480s, per-seed cap=1930s, B-spline seeds 901–905 in parallel 2/2/1 waves, 20,000 Adam updates and saved pools/L-BFGS contract unchanged; 512x256 confirmation retained. Optional DeepXDE reference skipped due no residual cap. Static source read only; no local training, tests, validators, or quota query.
- DFG push initiated: slug=rodrimandub/af-pikans-dfg-fem-field-followup-20260929; push_started_utc=2026-09-29T06:29:37.409Z; cap=1800s. Kaggle CLI process loads credentials from repo .env; credentials are not echoed.
- DFG push returned successfully: slug=rodrimandub/af-pikans-dfg-fem-field-followup-20260929; push_started_utc=2026-09-29T06:29:37.409Z; push_returned_utc=2026-09-29T06:29:42.258Z; cap=1800s (0.50h); private Kaggle T4x2.
- DFG status observed at 2026-09-29T06:30:17.388Z: unknown (CLI exit=1); cap=1800s; slug=rodrimandub/af-pikans-dfg-fem-field-followup-20260929.
- Kaggle listed the created DFG kernel under canonical slug rodrimandub/af-pikans-dfg-fem-field-follow-up-20260929 (title slug normalized; no duplicate push). Status observed at 2026-09-29T06:31:48.520Z: COMPLETE (CLI exit=0); cap=1800s.
- DFG output bundle downloaded: canonical slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-20260929; downloaded_at_utc=2026-09-29T06:32:29.775Z; destination=var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-fem-field-followup-20260929; cap=1800s; CLI exit=0.
- DFG attempt 1 artifact inspection at 2026-09-29T06:35:08.057Z: status=partial; actual_session_seconds=77.523 of 1800s cap; environment created, but worker still ran the old embedded solve_dfg.py and failed with ModuleNotFoundError: No module named 'gmsh'. Root cause is packaging replacement using AST line numbers as string offsets; field reference/scores absent. No training occurred. Remaining DFG allocation=1722.477s; retry only within that remaining cap after correcting asset replacement.
- DFG retry static pre-push read at 2026-09-29T06:39:37.671Z: embedded solver source was decoded for inspection; imports include subprocess/sys/time, DOLFINx gmshio, and no Python import gmsh; environment reports the gmsh binary version. DFG retry cap=1722.477s; worker cap=1550s; score loop leaves a final 15s reserve. Manifest declares fixed grid and outlet gauge before scoring; source keys for 301-305/501-505 and two source kernels are present. No local kernel execution/tests/validators.
- DFG retry push initiated: slug=rodrimandub/af-pikans-dfg-fem-field-followup-retry-20260929; push_started_utc=2026-09-29T06:39:37.746Z; cap=1722.477s (remaining DFG allowance after attempt 1).
- DFG retry push returned successfully: slug=rodrimandub/af-pikans-dfg-fem-field-followup-retry-20260929; push_started_utc=2026-09-29T06:39:37.746Z; push_returned_utc=2026-09-29T06:39:42.667Z; cap=1722.477s; private Kaggle T4x2.
- DFG retry canonical slug confirmed/status observed at 2026-09-29T06:40:27.827Z: RUNNING (CLI exit=0); cap=1722.477s; slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-20260929.
- DFG retry status observed at 2026-09-29T06:43:03.518Z: COMPLETE (CLI exit=0); cap=1722.477s; slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-20260929.
- DFG retry output downloaded at 2026-09-29T06:43:29.940Z: canonical slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-20260929; destination=var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-fem-field-followup-retry-20260929; cap=1722.477s; CLI exit=0.
- DFG retry attempt 2 inspected at 2026-09-29T06:51:50.051Z: actual_session_seconds=85.572; binary Gmsh 4.15.2 and version probe worked, but DOLFINx gmshio.read_from_msh still required its Python gmsh module and failed before solving; only L0 mesh was generated, no levels/QoIs/field scores. PyPI official page confirmed gmsh 4.15.2 has a Linux wheel and Python API. Cumulative DFG session use so far=163.095s; remaining allocation=1636.905s; retry2 now installs gmsh==4.15.2 inside micromamba and records the imported version.
- DFG retry2 static pre-push read at 2026-09-29T06:53:06.145Z: confirmed the embedded solve_dfg.py source has required DOLFINx/Gmsh imports, Gmsh binary version reporting, and gmsh_python version import after the pinned gmsh==4.15.2 installation in the same micromamba Python environment. The previous failing Python import key is absent. Session cap=1636.905s (remaining DFG allowance), worker cap=1500s; pre-scoring manifest fixes the 221x42 evaluation grid and outlet gauge; scoring checkpoint IDs 301-305 and 501-505; no local execution/tests/validators.
- DFG retry2 push initiated: slug=rodrimandub/af-pikans-dfg-fem-field-followup-retry2-20260929; push_started_utc=2026-09-29T06:53:06.195Z; cap=1636.905s (remaining DFG allocation).
- DFG retry2 push returned successfully: slug=rodrimandub/af-pikans-dfg-fem-field-followup-retry2-20260929; push_started_utc=2026-09-29T06:53:06.195Z; push_returned_utc=2026-09-29T06:53:10.328Z; cap=1636.905s; private Kaggle T4x2.
- DFG retry2 canonical slug listed/status observed at 2026-09-29T06:53:46.822Z: RUNNING; cap=1636.905s; slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-2-20260929 (CLI exit=0).
- DFG retry2 status changed/checked at 2026-09-29T06:57:24.135Z: COMPLETE (CLI exit=0); cap=1636.905s; slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-2-20260929.
- DFG retry2 output downloaded at 2026-09-29T06:57:49.262Z: canonical slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-2-20260929; destination=var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-fem-field-followup-retry2-20260929; cap=1636.905s; CLI exit=0.
- DFG retry2 artifact inspection at 2026-09-29T06:59:34.774Z: actual_session_seconds=89.985; Python gmsh import and binary version both confirmed at 4.15.2; solver failed before solving with AttributeError: module 'dolfinx.fem' has no attribute 'petsc'. No levels/QoIs/field scores. Cumulative DFG session use=253.080s; remaining DFG allocation=1546.920s for one corrected retry.

- 2026-09-29 07:08 UTC: DFG retry3 ruling: DOLFINx 0.11 documents the PETSc SNES NonlinearProblem under optional dolfinx.fem.petsc; explicitly import that submodule as fem_petsc and use fem_petsc.NonlinearProblem. This remains Newton via snes_type=newtonls. The kernel cap is the exact remaining DFG allocation 1546.920s; worker cap=1350s leaves scoring/finalization time. Reference: https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.fem.petsc.html. No quota query or local execution of solver/kernel.
- DFG retry3 push initiated: requested slug=rodrimandub/af-pikans-dfg-fem-field-followup-retry3-20260929; push_started_utc=2026-09-29 07:11:25 UTC; cap=1546.920s (remaining DFG allocation). Kaggle CLI child process loads .env credentials; no quota query.
- DFG retry3 push returned at 2026-09-29 07:11:34 UTC: requested slug=rodrimandub/af-pikans-dfg-fem-field-followup-retry3-20260929; cap=1546.920s; private Kaggle T4x2; CLI exit=1; canonical slug/status pending list.
- DFG retry3 push command-path correction at 2026-09-29 07:12:14 UTC: the first local invocation exited 1 before Kaggle CLI startup because its relative venv path was resolved from the kernel folder. No Kaggle request/kernel/GPU time; retry from repository root with repo-relative wrapper path.
- DFG retry3 Kaggle push retry initiated: requested slug=rodrimandub/af-pikans-dfg-fem-field-followup-retry3-20260929; push_started_utc=2026-09-29 07:12:19 UTC; cap=1546.920s. Kaggle CLI child process loads .env credentials; no quota query.
- DFG retry3 Kaggle push returned at 2026-09-29 07:12:34 UTC: requested slug=rodrimandub/af-pikans-dfg-fem-field-followup-retry3-20260929; cap=1546.920s; private Kaggle T4x2; CLI exit=0; canonical slug/status pending list.
- DFG retry3 canonical kernel observed at 2026-09-29 07:13:08 UTC: rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-3-20260929; status listed below; cap=1546.920s. Kernel push warning: title slug did not exactly resolve to requested id, but Kaggle accepted version 1 and exposed this canonical URL.
- DFG retry3 status observed: RUNNING; cap=1546.920s; canonical slug rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-3-20260929.
- Heat B-spline follow-up package refreshed for exact title-derived canonical slug rodrimandub/af-pikans-heat-b-spline-completion-follow-up-20260929. Static review: private T4x2, cap=6480s, B-spline seeds=901-905 in 2/2/1 two-GPU waves, RUN_CAP=1930s, 20,000 Adam target retained, DeepXDE 1.12.2 L-BFGS contract and 512x256 confirmation retained; optional alpha=0.1 reference skipped.
- DFG retry3 status poll at 2026-09-29 07:18:29 UTC: COMPLETE; cap=1546.920s; kernel has already been pushed once and will not be relaunched.
- DFG retry3 output download started at 2026-09-29 07:19:01 UTC; canonical slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-3-20260929; destination=kaggle-outputs/dfg-fem-field-followup-retry3-20260929; cap=1546.920s.
- DFG retry3 output download returned at 2026-09-29 07:19:22 UTC: canonical slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-3-20260929; destination=kaggle-outputs/dfg-fem-field-followup-retry3-20260929; cap=1546.920s; CLI exit=0.
- DFG retry3 artifacts inspected: actual_session_seconds=86.359/1546.920; Gmsh Python+binaries 4.15.2 and DOLFINx 0.11.0 imported; first mesh generated, then solve failed before levels with NotImplementedError: Cannot take length of non-vector expression. Bundle downloaded; no QoI/field scores. Cumulative DFG actual=339.439s; remaining DFG allowance=1460.561s.
- 2026-09-29 DFG retry4 ruling: DOLFINx 0.11 Function.collapse() returns a Function; only FunctionSpace.collapse() returns (space,map). Removed tuple unpacking for both vector velocity and scalar pressure. Official API: https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.fem.html. Retry cap=1460.561s (remaining of the 1800s DFG allowance after 339.439s actual); worker cap=1300s; optional reference remains skipped.
- DFG retry4 push initiated: canonical title-derived slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-4-20260929; push_started_utc=2026-09-29 07:24:18 UTC; cap=1460.561s (remaining DFG allowance); private T4x2; no quota query.
- DFG retry4 push returned at 2026-09-29 07:24:31 UTC: slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-4-20260929; cap=1460.561s; private Kaggle T4x2; CLI exit=0; status pending.
- DFG retry4 status poll at 2026-09-29 07:25:12 UTC: RUNNING; cap=1460.561s; no duplicate launch.
- DFG retry4 status poll at 2026-09-29 07:26:02 UTC: RUNNING; cap=1460.561s; no duplicate launch.
- DFG retry4 status poll at 2026-09-29 07:27:00 UTC: RUNNING; cap=1460.561s; no duplicate launch.
- DFG retry4 status poll at 2026-09-29 07:28:06 UTC: COMPLETE; cap=1460.561s; no duplicate launch.
- DFG retry4 output download started at 2026-09-29 07:28:30 UTC; slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-4-20260929; destination=kaggle-outputs/dfg-fem-field-followup-retry4-20260929; cap=1460.561s.
- DFG retry4 output download returned at 2026-09-29 07:29:08 UTC: slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-4-20260929; destination=kaggle-outputs/dfg-fem-field-followup-retry4-20260929; cap=1460.561s; CLI exit=undefined.
- DFG retry4 download completion clarification: CLI exceeded the first 30s wait and returned no exit code to the polling cell; a follow-up filesystem read found the output files present and no Kaggle CLI process remains. Artifact inspection proceeds from the downloaded destination.
- DFG retry4 downloaded bundle inspection: 30 files present including final manifest, FEM fields/QoIs/coefficients for L0-L2 and full checkpoint score JSON/CSV. All artifacts remain candidate evidence only.

- 2026-09-29 07:50:27 UTC: DFG retry5 static pre-push review passed after correcting reference_path to DFG-L3; contract/manifest already declared DFG-L3, fixed outlet gauge, grid 221x42, checkpoint cohorts 301-305 and 501-505; entrypoint embeds solve_dfg.py and dfg_model.py; source hashes match manifest. Kernel list showed neither retry5 nor Heat follow-up present. A system-Python package-builder invocation failed before script execution (missing encodings); package was generated using the existing Kaggle venv Python. No local training/tests/validators or quota query.
- DFG retry5 push initiated: slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929; push_started_utc=2026-09-29 07:50:27 UTC; cap=1281.024s (remaining DFG allowance); private Kaggle T4x2.
- DFG retry5 push returned at 2026-09-29 07:50:38 UTC: requested slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929; cap=1281.024s; private Kaggle T4x2; CLI exit=0; status pending confirmation.
- DFG retry5 status checked at 2026-09-29 07:51:04 UTC: slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929; cap=1281.024s; CLI exit=0; status output recorded below.
- DFG retry5 status poll at 2026-09-29 07:54:00 UTC: slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929; cap=1281.024s; CLI exit=0; [REDACTED]/af-pikans-dfg-fem-field-follow-up-retry-5-20260929 has status "KernelWorkerStatus.RUNNING".
- DFG retry5 status poll at 2026-09-29 07:56:31 UTC: slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929; cap=1281.024s; CLI exit=0; [REDACTED]/af-pikans-dfg-fem-field-follow-up-retry-5-20260929 has status "KernelWorkerStatus.RUNNING".
- DFG retry5 status poll at 2026-09-29 07:58:14 UTC: slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929; cap=1281.024s; CLI exit=0; [REDACTED]/af-pikans-dfg-fem-field-follow-up-retry-5-20260929 has status "KernelWorkerStatus.RUNNING".
- DFG retry5 status poll at 2026-09-29 08:00:23 UTC: slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929; cap=1281.024s; CLI exit=0; [REDACTED]/af-pikans-dfg-fem-field-follow-up-retry-5-20260929 has status "KernelWorkerStatus.RUNNING".
- DFG retry5 status poll at 2026-09-29 08:02:35 UTC: slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929; cap=1281.024s; CLI exit=0; [REDACTED]/af-pikans-dfg-fem-field-follow-up-retry-5-20260929 has status "KernelWorkerStatus.RUNNING".
- DFG retry5 status poll at 2026-09-29 08:04:25 UTC: slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929; cap=1281.024s; CLI exit=0; [REDACTED]/af-pikans-dfg-fem-field-follow-up-retry-5-20260929 has status "KernelWorkerStatus.COMPLETE".
- DFG retry5 output download started at 2026-09-29 08:04:57 UTC: canonical slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929; destination=var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-fem-field-followup-retry5-20260929; cap=1281.024s.
- DFG retry5 output download returned at 2026-09-29 08:05:33 UTC: canonical slug=rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929; destination=var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-fem-field-followup-retry5-20260929; cap=1281.024s; CLI exit=undefined; .
- DFG retry5 download completion confirmed at 2026-09-29 08:08:59 UTC: destination now has 32 files (84,070,296 bytes), including final manifest, worker status, all three mesh levels and checkpoint score JSON/CSV. The initial CLI wait returned no exit code; files arrived asynchronously, so no duplicate download was started.
- DFG retry5 artifacts inspected at 2026-09-29 08:24:18 UTC: actual_session_seconds=753.916 of 1281.024; status=candidate_complete; DOLFINx 0.11.0, Gmsh binary/Python 4.15.2, PETSc 3.25.5; T4x2 verified. Three levels saved. FEATFLOW errors: Cd relative=2.88793e-5, Cl absolute=6.41082e-8, delta_p relative=1.20946e-4 (all pass). Finest L2->L3 changes: Cd relative=9.04228e-5 and delta_p relative=1.06461e-4 (pass); Cl relative=6.64849e-3 (fails 0.2%). All 15 checkpoint scores failed because exported L3 velocity has shape (9203,) rather than (9203,2); xy=(9203,2); saved P2 vector coefficients are available. No scores are reported. Cumulative DFG use=1272.892s; remaining item-2 allowance=527.108s. A score-only postprocessing repair is planned within that remainder; no FEM retraining.
- DFG score-repair package statically reviewed at 2026-09-29 08:37:41 UTC: private T4x2; cap=527.108s (remaining DFG allowance); reads retry5 L3 mesh/field/coefficient outputs plus checkpoint sources; reconstructs P2 u/v and P1 p; scores 301-305 and 501-505; partial JSON written before setup/scoring and after each checkpoint; scientific_accepted=false. No local training/tests/validators or quota read.
- DFG field-score-repair push initiated: slug=rodrimandub/af-pikans-dfg-field-score-repair-20260929; push_started_utc=2026-09-29 08:38:17 UTC; cap=527.108s (remaining DFG allowance); private Kaggle T4x2. Fresh kernels list --mine confirmed this slug is absent; quota not queried.
- DFG field-score-repair push returned at 2026-09-29 08:38:28 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-20260929; cap=527.108s; private Kaggle T4x2; CLI exit=0; status pending.
- DFG field-score-repair status poll at 2026-09-29 08:38:46 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-20260929; cap=527.108s; CLI exit=0; [REDACTED]/af-pikans-dfg-field-score-repair-20260929 has status "KernelWorkerStatus.RUNNING".
- DFG field-score-repair status poll at 2026-09-29 08:40:04 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-20260929; cap=527.108s; CLI exit=0; [REDACTED]/af-pikans-dfg-field-score-repair-20260929 has status "KernelWorkerStatus.COMPLETE".
- DFG field-score-repair output download started at 2026-09-29 08:40:36 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-20260929; destination=var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-field-score-repair-20260929; cap=527.108s.
- DFG field-score-repair output download completed at 2026-09-29 08:41:04 UTC: CLI exit=0; destination=var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-field-score-repair-20260929; cap=527.108s; downloaded output manifest, partial ledger, status, setup log and Kaggle kernel log.
- DFG field-score-repair attempt 1 inspected at 2026-09-29 08:44:35 UTC: actual_session_seconds=16.750/527.108; status=failed_partial before field reconstruction/scoring. meshio installed successfully; exact rounded-coordinate dictionary could not map 25,224 velocity dof coordinates to mesh nodes. No FEM solve/retraining or checkpoint score occurred. Remaining DFG allowance=510.358s (cumulative DFG actual=1289.642s). Retry will use nearest-node coordinate matching with a bounded distance and one-to-one mapping check.
- 2026-09-29 08:51:02 UTC DFG score-repair retry static pre-push reread: private Kaggle T4x2 metadata, cap=510.358s (remaining DFG allotment), new retry slug confirmed absent from a fresh kernels list --mine; manifest/source SHA256 agree. Required imports, IDs 301-305 and 501-505, fixed 221x42 grid/outlet gauge, coordinate-map tolerance/uniqueness diagnostics, partial JSON writes before scoring and after every checkpoint, and false acceptance flags are present. No local kernel execution, training, tests, or validators; quota not queried.

- DFG score-repair retry push initiated: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-20260929; push_started_utc=2026-09-29 08:52:18.047 UTC; cap=510.358s (remaining DFG allotment); private Kaggle T4x2; quota not queried.
- DFG score-repair retry push returned: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-20260929; push_started_utc=2026-09-29 08:52:18.047 UTC; push_returned_utc=2026-09-29 08:52:21.954 UTC; cap=510.358s; private Kaggle T4x2; CLI exit=0.

- DFG score-repair retry status observed at 2026-09-29 08:52:51.927 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-20260929 has status "KernelWorkerStatus.RUNNING"; cap=510.358s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-20260929; CLI exit=0; no duplicate kernel launched.

- DFG score-repair retry status observed at 2026-09-29 08:55:52.733 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-20260929 has status "KernelWorkerStatus.COMPLETE"; cap=510.358s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-20260929; CLI exit=0.

- DFG score-repair retry output download started: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-20260929; download_started_utc=2026-09-29 08:56:15.626 UTC; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry-20260929; cap=510.358s.
- DFG score-repair retry output download returned at 2026-09-29 08:56:21.575 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-20260929; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry-20260929; cap=510.358s; CLI exit=0.
- 2026-09-29 09:06:01 UTC DFG score-repair retry1 artifact inspection: actual_session_seconds=13.648/510.358; status=failed_partial before reconstruction/scoring. The nearest map was one-to-one for 25,231 DOFs but max distance=0.001497843m and p99=0.001324244m, exceeding the strict floating-point tolerance; no checkpoint scored. Retry2 first maps through saved DOLFINx geometry, then allows at most 0.0016m direct nearest-node distance with independent <0.5% reconstruction agreement against both stored velocity and pressure traces. Cumulative DFG use=1303.290s; remaining of 1800s DFG allocation=496.710s.

- 2026-09-29 09:07:04.824 UTC fresh kernels list --mine before DFG score-repair retry2 push: target slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-2-20260929 present=False; CLI exit=0; no quota command issued.

- DFG score-repair retry2 push initiated: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-2-20260929; push_started_utc=2026-09-29 09:07:20.691 UTC; cap=496.710s (remaining DFG allotment); private Kaggle T4x2; quota not queried.
- DFG score-repair retry2 push returned: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-2-20260929; push_started_utc=2026-09-29 09:07:20.691 UTC; push_returned_utc=2026-09-29 09:07:24.998 UTC; cap=496.710s; private Kaggle T4x2; CLI exit=0.

- DFG score-repair retry2 status observed at 2026-09-29 09:07:39.811 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-2-20260929 has status "KernelWorkerStatus.RUNNING"; cap=496.710s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-2-20260929; CLI exit=0.

- DFG score-repair retry2 status observed at 2026-09-29 09:08:30.504 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-2-20260929 has status "KernelWorkerStatus.COMPLETE"; cap=496.710s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-2-20260929; CLI exit=0.

- DFG score-repair retry2 output download started: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-2-20260929; download_started_utc=2026-09-29 09:08:45.879 UTC; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry2-20260929; cap=496.710s.
- DFG score-repair retry2 output download returned at 2026-09-29 09:08:52.016 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-2-20260929; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry2-20260929; cap=496.710s; CLI exit=0.
- 2026-09-29 09:08:52 UTC DFG score-repair retry2 artifact inspection: actual_session_seconds=18.062/496.710; status=failed_partial before checkpoint scoring. The map succeeded, but the best of the two vector coefficient layouts and two velocity components still mismatched the stored FEM scalar velocity trace by relative L2=1.0087242; no scores were emitted. Cumulative DFG use=1321.352s; remaining of 1800s DFG allowance=478.648s. Retry3 will test all six permutations of the three quadratic edge-node positions, retaining the 0.5% velocity and pressure trace checks.

- 2026-09-29 09:13:32 UTC DFG score-repair retry3 static reread: cap=478.648s; source/manifest SHA256=ce6c07da7eaf98449e9c4d2ee403db8dbcf14a4be4ab208cc11581fb6619395b; private T4x2 metadata; saved DOLFINx geometry route, nearest-distance check, all six quadratic edge-node permutations, two coefficient layouts and two scalar components tested against stored FEM trace; failed-candidate diagnostics written before exit; score partial JSON persists before scoring and after each checkpoint. Heat package static hash also agrees (cd80dcda1a4aeb6f815dbcb0154558be4411ad8cbf4cd99991ccd013ab33e737). No local execution/tests/validators; quota not queried.

- 2026-09-29 09:13:46.458 UTC fresh kernels list --mine before DFG score-repair retry3 push: target slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-3-20260929 present=False; CLI exit=0; quota not queried.

- DFG score-repair retry3 push initiated: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-3-20260929; push_started_utc=2026-09-29 09:14:00.372 UTC; cap=478.648s (remaining DFG allotment); private Kaggle T4x2; quota not queried.
- DFG score-repair retry3 push returned: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-3-20260929; push_started_utc=2026-09-29 09:14:00.372 UTC; push_returned_utc=2026-09-29 09:14:04.289 UTC; cap=478.648s; private Kaggle T4x2; CLI exit=0.

- DFG score-repair retry3 status observed at 2026-09-29 09:14:17.417 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-3-20260929 has status "KernelWorkerStatus.QUEUED"; cap=478.648s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-3-20260929; CLI exit=0.

- DFG score-repair retry3 status observed at 2026-09-29 09:15:53.974 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-3-20260929 has status "KernelWorkerStatus.QUEUED"; cap=478.648s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-3-20260929; CLI exit=0.

- Runtime precision correction: DFG score-repair retry2 final manifest records 18.061s (status JSON rounded to 18.062s); cumulative DFG actual=1321.351s and exact remaining=478.649s. Retry3 cap=478.648s, retaining 0.001s headroom.

- DFG score-repair retry3 status observed at 2026-09-29 09:18:39.368 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-3-20260929 has status "KernelWorkerStatus.COMPLETE"; cap=478.648s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-3-20260929; CLI exit=0.

- DFG score-repair retry3 output download started: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-3-20260929; download_started_utc=2026-09-29 09:18:52.963 UTC; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry3-20260929; cap=478.648s.
- DFG score-repair retry3 output download returned at 2026-09-29 09:18:58.513 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-3-20260929; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry3-20260929; cap=478.648s; CLI exit=0.

- 2026-09-29 09:18:58 UTC DFG score-repair retry3 artifact inspection: final manifest actual_session_seconds=37.557/478.648 (score status JSON rounds to 37.558); status=failed_partial during failure-diagnostic serialization before scoring. All six edge permutations and vector layouts failed the 0.5% trace threshold; diagnostic candidate list contained NaN and violated allow_nan=false, obscuring the numeric comparison. Cumulative DFG actual=1358.908s; remaining of 1800s allowance=441.092s; retry4 cap=441.091s.

- 2026-09-29 09:23:47 UTC DFG score-repair retry4 static reread: cap=441.091s; source/manifest SHA256=82bbc538ada3b9e388e976b9bc2a045122652a27805ace4523722d20cb80dd05; private T4x2 metadata; exact saved DOLFINx geometry mapping then constrained nearest fallback; six quadratic edge-node permutations, two coefficient layouts and both velocity components; non-finite candidates encoded as null and candidate/node-coverage diagnostics persist under allow_nan=false; <0.5% stored velocity/pressure trace gates remain. No local execution/training/tests/validators; quota not queried.

- 2026-09-29 09:24:01.850 UTC fresh kernels list --mine before DFG score-repair retry4 push: target slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-4-20260929 present=False; CLI exit=0; no quota command.

- DFG score-repair retry4 push initiated: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-4-20260929; push_started_utc=2026-09-29 09:24:14.973 UTC; cap=441.091s (remaining DFG allotment); private Kaggle T4x2; quota not queried.
- DFG score-repair retry4 push returned: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-4-20260929; push_started_utc=2026-09-29 09:24:14.973 UTC; push_returned_utc=2026-09-29 09:24:19.375 UTC; cap=441.091s; private Kaggle T4x2; CLI exit=0.

- DFG score-repair retry4 status observed at 2026-09-29 09:24:33.759 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-4-20260929 has status "KernelWorkerStatus.QUEUED"; cap=441.091s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-4-20260929; CLI exit=0.

- DFG score-repair retry4 status observed at 2026-09-29 09:26:19.946 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-4-20260929 has status "KernelWorkerStatus.QUEUED"; cap=441.091s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-4-20260929; CLI exit=0.

- DFG score-repair retry4 status observed at 2026-09-29 09:28:09.181 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-4-20260929 has status "KernelWorkerStatus.RUNNING"; cap=441.091s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-4-20260929; CLI exit=0.

- DFG score-repair retry4 status observed at 2026-09-29 09:29:17.348 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-4-20260929 has status "KernelWorkerStatus.COMPLETE"; cap=441.091s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-4-20260929; CLI exit=0.

- DFG score-repair retry4 output download started: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-4-20260929; download_started_utc=2026-09-29 09:29:31.158 UTC; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry4-20260929; cap=441.091s.
- DFG score-repair retry4 output download returned at 2026-09-29 09:29:38.694 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-4-20260929; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry4-20260929; cap=441.091s; CLI exit=0.

- 2026-09-29 09:29:38 UTC DFG score-repair retry4 artifact inspection: actual_session_seconds=39.124/441.091; status=failed_partial, no checkpoint scores. Every coefficient-layout/edge-order candidate generated non-finite predictions, so no finite relative error could be selected. The catch handler overwrote the detailed candidate/mapping partial JSON with its generic traceback; retry5 will preserve prior diagnostics. Cumulative DFG actual=1398.032s; remaining of 1800s allowance=401.968s.

- 2026-09-29 09:32:17 UTC DFG score-repair retry5 static reread: cap=401.968s; source/manifest SHA256=dbc2d9b5bf133d6de8ce7e0d3545eea722c41caf309124e8a4ca50f92ec7953c; private T4x2; preserves detailed coordinate/layout/permutation candidate diagnostics through outer exception handler; no non-finite JSON values; no retraining, partial-before-score and after-checkpoint behavior retained. No local execution/training/tests/validators; quota not queried.

- 2026-09-29 09:32:32.678 UTC fresh kernels list --mine before DFG score-repair retry5 push: target slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-5-20260929 present=False; CLI exit=0; no quota command.

- DFG score-repair retry5 push initiated: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-5-20260929; push_started_utc=2026-09-29 09:32:47.570 UTC; cap=401.968s (remaining DFG allotment); private Kaggle T4x2; quota not queried.
- DFG score-repair retry5 push returned: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-5-20260929; push_started_utc=2026-09-29 09:32:47.570 UTC; push_returned_utc=2026-09-29 09:32:51.512 UTC; cap=401.968s; private Kaggle T4x2; CLI exit=0.

- DFG score-repair retry5 status observed at 2026-09-29 09:33:04.804 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-5-20260929 has status "KernelWorkerStatus.QUEUED"; cap=401.968s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-5-20260929; CLI exit=0.

- 2026-09-29 09:29:38 UTC DFG retry4 root-cause review: output mesh DFG-L3 has 395,252 geometry nodes, while repair's 25,231 nearest-mapped DOFs match DFG-L1. ind_one() selected the lexically first L1 fields/coefficients but the fixed L3 mesh. Retry5 pins both ields_on_common_grid.npz and em_coefficients.npz to /dfg/DFG-L3/; no FEM re-solve or checkpoint retraining.

- 2026-09-29 09:36:28 UTC DFG score-repair retry5 final static reread: L3-only asset filters for fields/coefficients/mesh, source paths recorded in output; cap=401.968s, private T4x2, source/manifest SHA256=03dd436f77eb2698395540e8c797628a3b1e7b241d534e40f1ea6bbc06453167; earlier candidate diagnostics preserved on exceptions and non-finite values serialized as null. No local execution/training/tests/validators; quota not queried.

- 2026-09-29 09:36:44.574 UTC fresh kernels list --mine before corrected DFG retry5 push: target slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-5-20260929 present=True; CLI exit=0; quota not queried.

- DFG score-repair retry5 pre-fix pushed version status observed at 2026-09-29 09:37:34.768 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-5-20260929 has status "KernelWorkerStatus.RUNNING"; cap=401.968s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-5-20260929; CLI exit=0; local L3-only patch applies to a new version only after this run completes.

- DFG score-repair retry5 pre-fix version status observed at 2026-09-29 09:38:14.645 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-5-20260929 has status "KernelWorkerStatus.COMPLETE"; cap=401.968s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-5-20260929; CLI exit=0.

- DFG score-repair retry5 pre-fix version output download started: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-5-20260929; download_started_utc=2026-09-29 09:38:29.602 UTC; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry5-20260929; cap=401.968s.
- DFG score-repair retry5 pre-fix version output download returned at 2026-09-29 09:38:36.625 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-5-20260929; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry5-20260929; cap=401.968s; CLI exit=0.

- 2026-09-29 09:38:36 UTC DFG score-repair retry5 pre-fix artifact inspection: manifest actual_session_seconds=28.469/401.968; failed before scoring. Detailed partial confirms L3 mesh/triangle6 each have 395,252 nodes but selected L1 coefficient file has only 25,231 velocity DOFs and 6,405 pressure DOFs; 18,406 non-finite velocity-grid values resulted from unmapped L3 nodes. Local retry6 selects fields, coefficients, and mesh only from /dfg/DFG-L3/. Cumulative DFG actual=1426.501s; remaining of 1800s allowance=373.499s.

- 2026-09-29 09:40:49 UTC DFG score-repair retry6 static reread: all FEM artifact searches require the DFG-L3 directory; path recording, geometry map and trace-check diagnostics are present; cap=373.499s (remaining DFG budget); source/manifest SHA256=05dabe7776f9d0b5a7c2fd0e6cf046ce7f776bdfa7a1ce76da62e48e4899cf87; private T4x2. No local execution/training/tests/validators; quota not queried.

- 2026-09-29 09:41:05.692 UTC fresh kernels list --mine before corrected DFG retry6 push: target slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929 present=False; CLI exit=0; quota not queried.

- DFG score-repair retry6 push initiated: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; push_started_utc=2026-09-29 09:41:22.542 UTC; cap=373.499s (remaining DFG allotment); private Kaggle T4x2; quota not queried.
- DFG score-repair retry6 push returned: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; push_started_utc=2026-09-29 09:41:22.542 UTC; push_returned_utc=2026-09-29 09:41:26.874 UTC; cap=373.499s; private Kaggle T4x2; CLI exit=0.

- DFG score-repair retry6 status observed at 2026-09-29 09:41:42.114 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-6-20260929 has status "KernelWorkerStatus.QUEUED"; cap=373.499s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; CLI exit=0.

- DFG score-repair retry6 status observed at 2026-09-29 09:43:30.830 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-6-20260929 has status "KernelWorkerStatus.QUEUED"; cap=373.499s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; CLI exit=0.

- 2026-09-29 09:45:08 UTC Heat B-spline completion static reread: private T4x2 metadata and two-GPU runtime check; source/manifest SHA256=f577ebf3dacf0c928316c9aa45b5ad4c6a9529306be9f1d09e20ef29632d170f; cap=6480s, per-seed cap=1930s; B-spline only, seeds=901-905, 20,000 Adam steps, DeepXDE 1.12.2 L-BFGS settings, explicit float64, 512x256 confirmation; early/partial ledgers and seed checkpoint paths retained. Optional alpha=0.1 reference skipped. No local execution/training/tests/validators; quota not queried.

- DFG score-repair retry6 status observed at 2026-09-29 09:45:22.960 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-6-20260929 has status "KernelWorkerStatus.QUEUED"; cap=373.499s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; CLI exit=0.

- DFG score-repair retry6 status observed at 2026-09-29 09:46:47.964 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-6-20260929 has status "KernelWorkerStatus.QUEUED"; cap=373.499s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; CLI exit=0.

- DFG score-repair retry6 status observed at 2026-09-29 09:47:49.139 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-6-20260929 has status "KernelWorkerStatus.QUEUED"; cap=373.499s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; CLI exit=0.

- DFG score-repair retry6 status observed at 2026-09-29 09:49:14.246 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-6-20260929 has status "KernelWorkerStatus.QUEUED"; cap=373.499s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; CLI exit=0.

- DFG score-repair retry6 status observed at 2026-09-29 09:50:26.759 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-6-20260929 has status "KernelWorkerStatus.QUEUED"; cap=373.499s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; CLI exit=0.

- 2026-09-29 09:51:21.253 UTC fresh kernels list --mine before Heat B-spline push: target slug=rodrimandub/af-pikans-heat-b-spline-completion-follow-up-20260929 present=False; CLI exit=0; quota not queried.

- Heat B-spline completion push initiated: slug=rodrimandub/af-pikans-heat-b-spline-completion-follow-up-20260929; push_started_utc=2026-09-29 09:51:35.495 UTC; cap=6480s (1.80h); private Kaggle T4x2; no quota query.
- Heat B-spline completion push returned: slug=rodrimandub/af-pikans-heat-b-spline-completion-follow-up-20260929; push_started_utc=2026-09-29 09:51:35.495 UTC; push_returned_utc=2026-09-29 09:51:38.177 UTC; cap=6480s (1.80h); private Kaggle T4x2; CLI exit=1.

- 2026-09-29 09:52:23.696 UTC post-failure kernels list --mine after Heat SaveKernel 400: target slug=rodrimandub/af-pikans-heat-b-spline-completion-follow-up-20260929 present=False; CLI exit=0; no quota query.

- 2026-09-29 09:55:35 UTC Heat SaveKernel retry preparation: first long-title/internet-enabled slug was absent after 400 (no kernel/GPU run). New canonical title/id are shorter and internet is disabled because the workload uses only attached pool data; the T4x2, 6480s/1930s caps, float64, 20k Adam, L-BFGS contract, 512x256 grid, and early ledgers are unchanged. Source/manifest SHA256=2a6d94eff4ca9fac2e2260afad1437143169d072063d0be00e3066e9a0ec96fc; static reread complete; no local execution/tests/validators or quota query.

- 2026-09-29 09:55:48.497 UTC fresh kernels list --mine before shortened-title Heat push: target slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929 present=False; CLI exit=0; no quota command.

- Heat B-spline completion shortened-title push initiated: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; push_started_utc=2026-09-29 09:56:01.381 UTC; cap=6480s (1.80h); private Kaggle T4x2; internet disabled; quota not queried.
- Heat B-spline completion shortened-title push returned: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; push_started_utc=2026-09-29 09:56:01.381 UTC; push_returned_utc=2026-09-29 09:56:05.734 UTC; cap=6480s (1.80h); private Kaggle T4x2; CLI exit=0.

- Heat B-spline completion status observed at 2026-09-29 09:56:19.285 UTC: [REDACTED]/af-pikans-heat-b-spline-completion-20260929 has status "KernelWorkerStatus.QUEUED"; cap=6480s; slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; CLI exit=0.

- DFG score-repair retry6 status observed at 2026-09-29 09:56:32.061 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-6-20260929 has status "KernelWorkerStatus.COMPLETE"; cap=373.499s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; CLI exit=0.

- DFG score-repair retry6 output download started: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; download_started_utc=2026-09-29 09:56:46.190 UTC; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry6-20260929; cap=373.499s.
- DFG score-repair retry6 output download returned at 2026-09-29 09:56:52.218 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-6-20260929; destination=var\agent_runs\baseline-competitive-2026-09-27\kaggle-outputs\dfg-field-score-repair-retry6-20260929; cap=373.499s; CLI exit=0.

- 2026-09-29 09:56:52 UTC DFG score-repair retry6 artifact inspection: manifest actual_session_seconds=21.685/373.499; failed before loading/scoring because the mounted path did not contain the assumed /dfg/DFG-L3/ sequence. New filter will match DFG-L3 as a path component independent of ancestors and report/stop if L3 mesh geometry, velocity DOF coordinates, and P2 coefficient counts disagree. Cumulative DFG actual=1448.186s; remaining of 1800s allowance=351.814s.

- 2026-09-29 09:59:59 UTC DFG score-repair retry7 static reread: level filter matches DFG-L3 component independently of parent nesting; L3 geometry/velocity counts checked before mapping; full paths recorded; cap=351.814s (remaining DFG budget), private T4x2; source/manifest SHA256=1cf9ffb1de5716f8e0b335e6e5863403586f7c59a22cef0f36e223f92bdeedca; no local execution/tests/validators or quota query.

- 2026-09-29 10:00:12.466 UTC fresh kernels list --mine before DFG score-repair retry7 push: target slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-7-20260929 present=False; CLI exit=0; quota not queried.

- DFG score-repair retry7 push initiated: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-7-20260929; push_started_utc=2026-09-29 10:00:25.809 UTC; cap=351.814s; private Kaggle T4x2; quota not queried.
- DFG score-repair retry7 push returned: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-7-20260929; push_started_utc=2026-09-29 10:00:25.809 UTC; push_returned_utc=2026-09-29 10:00:29.610 UTC; cap=351.814s; private Kaggle T4x2; CLI exit=0.

- DFG score-repair retry7 status observed at 2026-09-29 10:00:41.511 UTC: [REDACTED]/af-pikans-dfg-field-score-repair-retry-7-20260929 has status "KernelWorkerStatus.RUNNING"; cap=351.814s; slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-7-20260929; CLI exit=0.
- DFG score-repair retry7 status observed at 2026-09-29 10:02:48 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-7-20260929; status=COMPLETE; cap=351.814s; CLI exit=0.
- Heat B-spline completion status observed at 2026-09-29 10:02:48 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=QUEUED; cap=6480s; CLI exit=0.
- DFG score-repair retry7 output download started at 2026-09-29 10:03 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-7-20260929; destination=var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-field-score-repair-retry7-20260929; cap=351.814s.
- DFG score-repair retry7 output download returned at 2026-09-29 10:03 UTC: slug=rodrimandub/af-pikans-dfg-field-score-repair-retry-7-20260929; destination=var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-field-score-repair-retry7-20260929; cap=351.814s; CLI exit=0.
- 2026-09-29 10:04 UTC DFG retry7 artifacts inspected: actual_session_seconds=31.555/351.814; status=complete; 15 checkpoints scored, retraining=false, T4x2. Reconstruction matched saved FEM traces (velocity relative L2=2.07716e-12; pressure=1.64406e-11); 221x42 grid, fixed outlet gauge (2.2,0.2). Current 301-305 RBF mean±sample SD: u=0.08617371±0.01219251, v=0.08548699±0.00975838, p=0.27467991±0.04246030 (n=5). 501-505 RBF: u=0.06965182±0.00390355, v=0.06909439±0.00578714, p=0.24616600±0.02262671; tanh-MLP: u=0.04688481±0.06925658, v=0.05203632±0.04914086, p=0.12271549±0.21521997 (n=5 each). Per-seed field differences recorded in JSON/CSV. FEATFLOW QoI checks pass, but finest-two Cl relative change=0.00664849 exceeds 0.002, so DFG verdict remains screening_only. Cumulative DFG session use=1479.741s; remaining item-2 cap=320.259s (no further DFG launch planned). scientific_accepted=false; reference_accepted=false.
- Heat B-spline completion status observed at 2026-09-29 10:05:37 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING (changed from QUEUED); cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:09:07 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:14:26 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:19:25 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:20:28 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:21:31 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:22:33 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:23:36 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:24:38 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:25:41 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:26:43 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:27:46 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:28:48 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:29:51 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:30:53 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:31:56 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:32:58 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:34:00 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:35:05 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:36:07 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:37:11 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- 2026-09-29 10:37:19 UTC runtime reconciliation: the root output manifest for DFG score-repair retry6 records 21.684s while its score_repair.json records 21.685s; other score-repair snapshots also differ by 0.001s. Per-session and cumulative totals use the root output manifest actual_session_seconds consistently: DFG total=1479.740s (sum of all 14 DFG output manifests).
- Heat B-spline completion status poll at 2026-09-29 10:38:13 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:39:16 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:40:19 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:41:21 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- 2026-09-29 10:41 UTC: Optional DeepXDE alpha=0.1 seeds 901-905 package statically prepared under followup-20260929/heat/deepxde-paired-901-905; 1000s kernel cap and 950s in-code watchdog; private T4x2, exact DeepXDE 1.12.2 reference recipe, 20000 Adam and L-BFGS-B, float32 default, 512x256 scoring. Not pushed; only launch if B-spline actual leaves at least 1000s of the 6480s Heat allowance. No quota query or local execution.
- Heat B-spline completion status poll at 2026-09-29 10:42:24 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:43:26 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:44:29 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:45:32 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:46:34 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:47:37 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:48:39 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:49:42 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:50:44 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:51:49 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:52:52 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:53:55 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:54:58 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:56:01 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:57:03 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:58:06 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 10:59:08 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:00:10 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:01:13 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:02:15 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:03:18 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:04:20 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:05:22 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:06:25 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:07:27 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:08:30 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:09:32 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:10:35 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:11:37 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:12:40 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:13:42 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:14:45 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:15:47 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:16:50 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:17:53 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:18:56 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:19:59 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:21:01 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:22:04 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:23:06 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:24:09 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:25:12 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:26:14 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:27:17 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:28:19 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:29:22 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:30:25 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:31:27 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:32:29 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:33:32 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:34:34 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:35:37 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:36:39 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:37:41 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:38:44 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=RUNNING; cap=6480s; CLI exit=0.
- Heat B-spline completion status poll at 2026-09-29 11:39:47 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; status=COMPLETE; cap=6480s; CLI exit=0.
- Heat B-spline completion output download started at 2026-09-29 11:40 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; destination=var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-b-spline-completion-20260929; cap=6480s.
- Heat B-spline completion output download returned at 2026-09-29 11:42:30 UTC: slug=rodrimandub/af-pikans-heat-b-spline-completion-20260929; destination=var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/heat-b-spline-completion-20260929; cap=6480s; CLI exit=0.
- 2026-09-29 11:44 UTC Heat B-spline artifacts inspected: actual_session_seconds=5670.565553738/6480; artifact status=partial; private T4x2 with both Tesla T4 visible; no quota read. All seeds 901-905 used verified original pool hashes, reached 20000 float64 Adam updates, and have 512x256 confirmation scores (n=5). L-BFGS settings were preserved; seeds 901/902/903 stopped after 23/49/49 updates; seeds 904/905 hit their 1930s per-seed caps after 1800/1650 L-BFGS updates. B-spline field relative L2=1.41599818e-6 ± 4.93291664e-7; existing RBF=2.06080322e-6 ± 2.30660614e-7. Paired RBF-minus-B-spline by seeds 901-905: -6.52224880e-8, 9.11119349e-7, 4.09409690e-7, 7.36703923e-7, 1.23201475e-6; mean=6.44805046e-7 ± 4.95916834e-7 (n=5). Remaining Heat item cap=809.434446262s, below the prepared optional DeepXDE cap=1000s; optional reference skipped. DFG actual=1479.740s; total follow-up actual=7150.305553738s (1.986195987h); quota estimate from researcher start 14.42h is 12.433804013h; not verified by CLI. scientific_accepted=false; reference_accepted=false.
