# Burgers GPU-only front diagnostics

Uses the ten validation.npz fields from the completed paired Burgers kernel; it loads no model and runs no training. On two Kaggle Tesla T4 devices it measures final-time zero-crossing position, reference-defined 10–90% shock width, front-window field error and residual. The output records every per-seed endpoint. Width thresholds are derived from reference plateau medians and then applied identically to prediction and reference. Scientific acceptance remains false.
