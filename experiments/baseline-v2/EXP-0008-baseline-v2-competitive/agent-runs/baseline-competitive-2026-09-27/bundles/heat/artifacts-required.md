# Artefact contract for the Heat T4x2 package

The run tree records each Tier2 training run's `cost.json`, `run_metadata.json`, `history.csv`, pool hash, `checkpoint-final.pt`, and every reached `120/300/600` checkpoint. Development scores on `257×129` are computed on CUDA after each run. Smoke seed 900 writes only a finite-output check; smoke never computes a quality score.

After all training has stopped, a separate scorer reconstructs each final checkpoint and computes the confirmation metrics on a newly generated `512×256` grid on CUDA. Training code neither imports nor receives the confirmation grid. Confirmation GPU-seconds, development GPU-seconds, smoke GPU-seconds, and training GPU-seconds are all summed into the device-second ledger. Kaggle quota-hours are recorded separately from aggregate per-device GPU-hours.

The campaign root includes one fixed pool NPZ per seed, shared across all arms; code and package SHA-256; all run logs; the campaign ledger; and confirmation metrics/checkpoint links. CPU is used only for JSON/CSV/NPZ serialization and process I/O. If a checkpoint or cost artifact is missing, the run remains recorded as incomplete; no seed is replaced.

The confirmation grid is created only within `heat_final_eval.py`, after the trainer exits. No score is used to select a method or tune the next run. The package includes no `evals/` files, approval record, or automatic SOTA claim.
