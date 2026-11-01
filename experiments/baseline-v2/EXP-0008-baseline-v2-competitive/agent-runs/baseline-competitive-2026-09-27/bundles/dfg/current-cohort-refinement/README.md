# DFG Re=20 current-cohort RBF L-BFGS continuation

This new Kaggle bundle continues the existing RBF-PIKAN checkpoints for seeds 301–305. It does not repeat Adam training and has no MLP arm.

The source mapping is fixed and checked at runtime:

| Seeds | Kaggle kernel source |
| --- | --- |
| 301–302 | `rodrimandub/af-pikans-dfg-re20-baseline-20260927-r3` |
| 303–305 | `rodrimandub/af-pikans-dfg-re20-baseline-20260927-r4` |

The loader searches beneath `/kaggle/input` for exactly one adjacent `rodrimandub/<kernel-slug>` route for each source. It requires exactly one checkpoint, pool archive, `run.json`, `provenance.json`, and `manifest_snapshot.json` in each expected `seed-N` folder. It checks `run.json` and `provenance.json` seed ids, DFG problem and RBF architecture, and verifies the saved-pool SHA-256 against `run.json`. It fails closed on missing, duplicated, misplaced, or ambiguous source files.

The r4 `src.research_pipeline.sciml.dfg` and `dfg_campaign` modules are embedded in the notebook and installed under a temporary runtime package. Their hashes are recorded in `SOURCE_MODULES.sha256` and checked again in Kaggle. This uses the r4 model definition, including its `output_scales` state buffer.

Each seed receives up to 540 seconds of deterministic full-pool L-BFGS using its saved training pools. Two seeds are optimized concurrently on the two Tesla T4 devices in waves 301/302, 303/304, and 305. The objective includes the official `steady_ns_residual` momentum and continuity residuals, the parabolic inlet, no-slip lower and upper walls and cylinder, and the natural outlet traction. FEATFLOW reference errors are calculated only during scoring and never enter the training objective.

The notebook scores each available checkpoint before and after refinement. It reports FEATFLOW drag, lift, and pressure difference using 512- and 1,024-point cylinder quadrature, plus held-out normalized residual summaries. Held-out points are reconstructed with the official r4 scrambled-Sobol generator from the held-out seed stream and remain separate from optimization pools.

The metadata requests Kaggle's known `NvidiaTeslaT4` GPU shape. The runtime requires exactly two visible Tesla T4 GPUs and disables CPU fallback. It caps planned work at 3,500 seconds, sets a 6,800-second watchdog, and records a 7,200-second session cap. With five seeds at 540 GPU-seconds each, the planned optimizer work is 2,700 GPU-seconds (45 GPU-minutes); the waves use at most 1,620 wall-clock seconds before scoring overhead. This leaves more than 12 hours from the stated fresh weekly allowance of 16 hours 23 minutes.

The output report records source artifact hashes, device assignment, optimizer steps and elapsed time, before/after scores, `field_relative_l2: null`, and `scientific_accepted: false`. There is no packaged converged FEM field, so a field-relative error is not claimed.

Files:

- `dfg-rbf-current-cohort-refinement.ipynb`: Kaggle notebook.
- `kernel-metadata.json`: private, internet-disabled, T4 GPU metadata with two-T4 enforcement in the runtime.
- `notebook_runtime.py`: readable notebook code template used by the builder.
- `dfg_rbf_lbfgs_worker.py`: one-seed L-BFGS worker embedded by the builder.
- `build_current_cohort_bundle.py`: rebuilds the notebook, metadata, source-module hashes, and bundle checksums.
- `SOURCE_MODULES.sha256` and `SHA256SUMS.txt`: source-module and bundle integrity records.

The notebook, training, and local validation were not run as part of bundle creation.
