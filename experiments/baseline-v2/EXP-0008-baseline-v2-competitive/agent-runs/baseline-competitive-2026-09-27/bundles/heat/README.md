# Heat matched-baseline package

**DRAFT_UNVERIFIED; offline review only.** No Kaggle upload, preflight, local training, or remote run has occurred. Heat has a hard 2-hour T4x2 quota/session cap inside the approved global 8-hour cap. It logs both Kaggle wall/quota time and aggregate per-device GPU seconds.

The package runs H-RBF, H-BS, and H-FNN on paired seeds `901–905`, with a finite-output-only smoke on seed `900`. All three arms now use the same Adam-only recipe and 500 device-seconds per seed; there is no FNN-only L-BFGS stage. H-FNN is a Torch-only port of the published DeepXDE tutorial architecture under this matched comparison recipe; it is not DeepXDE library execution or a literal paper reproduction. H-RBF and H-BS use a static model snapshot inside a new full-batch Torch loop. All three arms share float64, one persisted pool per seed, the exact hard trace, residual scaling 1.1, and the same CUDA evaluator.

Pools use preinstalled `torch.quasirandom.SobolEngine` (no SciPy or package install): 4,096 interior, 1,024 initial, and 512 points per spatial boundary. PyTorch Sobol pools differ from historical SciPy/PINA coordinates. Tier2 training caps are 500 GPU-seconds per arm/seed; a 1,000,000-step safety cap is intended to keep the 500-second wall budget, rather than a fixed update count, as the stopping rule. Checkpoints are saved at 120, 300, and 500 seconds; the final checkpoint is the one used by both development and confirmation scoring. Smoke caps remain 120 seconds per arm. All scores, meshes, residuals, and diagnostics run on CUDA. CPU handles artifact I/O only.

Development metrics are computed on `257×129` after each Tier2 run, with a 60-second allowance per two-run wave (480 seconds total). After training, a separate CUDA scorer evaluates the confirmation `512×256` grid with a 120-second allowance per two-checkpoint wave (960 seconds total). Smoke finite checks have a 10-second allowance per wave. Each of the 18 subprocess waves also reserves 20 seconds for process and artifact overhead. Before each smoke or Tier2 wave, the notebook reserves the remaining Tier2 waves, all eight confirmation waves, the 40-second artifact flush, and the 60-second close margin; confirmation scoring starts only when all remaining waves and the flush fit. The notebook watchdog starts in its first cell at 6,900 seconds; retain its 60-second close margin, reserve 300 seconds before that cell for Kaggle startup, and configure the CLI hard timeout to 7,200 seconds. The bounded plan is 300 startup + 300 setup/pools + 4,000 Tier2 training + 480 development scoring + 240 smoke training + 20 smoke checks + 960 confirmation scoring + 360 process-wave reserve + 40 artifact flush + 60 close margin = 6,760 seconds, leaving 440 seconds below the CLI hard cap. If any finite allowance is exceeded, the notebook records an incomplete campaign.

The device cap is 14,400 seconds (4 aggregate GPU-hours): 7,500 Tier2 training + 360 smoke training + 900 development scoring + 30 smoke checks + 1,800 confirmation scoring = 10,590, leaving a 3,810-device-second reserve. The session hard cap is 7,200 seconds (2 Kaggle quota-hours) on Kaggle `NvidiaTeslaT4` (T4×2), with at most two runs concurrent. Kaggle's official metadata docs identify `NvidiaTeslaT4` as T4×2: [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).

The ledger reports `complete` only when the three smoke checks are successful and finite, all 15 Tier2 runs have finite development metrics plus both 500-second and final checkpoints, and all 15 confirmation metrics are finite. Counting ledger rows alone never closes the campaign.

Important files:

- `heat-tier2-t4x2.ipynb`: self-contained notebook, not uploaded.
- `heat-tier2-review-package.zip` and `.sha256`: assembled review artifact and digest.
- `heat_comparison_runner.py`: Torch-only training and fixed Sobol-pool runner.
- `heat_common_eval.py`: CUDA development scorer shared by all arms.
- `heat_final_eval.py`: CUDA confirmation scorer, invoked only after training ends.
- `payload/af_sciml_model.py`: static H-RBF/H-BS model snapshot.
- `build_kaggle_bundle.py`: offline assembler; no Kaggle calls, preflight, or training.
- `adapters/heat_deepxde.py`: earlier standalone adapter draft; not included in the no-install H-FNN payload.

Historical cohorts remain contextual because pools, batching, traces, and scoring differ. The 10% internal margin remains a proposed decision rule requiring separate signature before interpreting confirmation scores; it is not universal SOTA.
