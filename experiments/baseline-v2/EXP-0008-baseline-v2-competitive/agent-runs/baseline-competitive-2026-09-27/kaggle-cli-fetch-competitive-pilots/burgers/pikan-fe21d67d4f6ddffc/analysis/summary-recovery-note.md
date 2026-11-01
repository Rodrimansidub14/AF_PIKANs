# Burgers post-training summary recovery

Kaggle completed training and wrote all ten final endpoint scores. The notebook’s last display cell failed because analysis/paired-primary-summary.json ends in a literal two-character \n after a complete JSON object. The source file is preserved unchanged. paired-primary-summary-recovered.json strips only that suffix; paired-results-recovered.json aggregates the ten per-endpoint score.json files.

The GPU logs report five RBF/Vanilla pairs and the score records identify Tesla T4. This recovery does not rerun training or alter scores. scientific_accepted remains false.
