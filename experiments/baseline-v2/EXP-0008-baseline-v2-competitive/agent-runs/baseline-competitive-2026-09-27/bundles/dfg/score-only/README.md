# DFG Re=20 score-only package

Files: dfg-re20-score-only.ipynb, kernel-metadata.json, and SHA256SUMS.txt.

The notebook embeds the original DFG campaign runner with SHA-256 4c02809600beb9b543d1bceffe2037ed7d05f5ff654d010b22984227b2cb35a9. It consumes source kernel outputs and saved pools/checkpoints, scores on CUDA, and never calls training. It requires the ten endpoints to record score_budget_exhausted and requires the embedded runner hash to equal the campaign source hash.

Scores include FEATFLOW QoIs, component-wise held-out PDE residuals, near-cylinder and wake residuals using the official region masks, cylinder no-slip diagnostics, per-seed paired summaries, phase timings, CUDA memory telemetry, and artifact hashes. The FEM field score remains null and scientific acceptance remains false.

Metadata selects Kaggle NvidiaTeslaT4, which supplies two T4 GPUs. Internet is disabled. The notebook has a 1,680-second watchdog and an 1,800-second planned session cap. This package was prepared offline and was not run, tested, preflighted, uploaded, or submitted.
