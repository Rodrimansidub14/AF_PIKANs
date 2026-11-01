# Harness loop scaffold

This folder contains a small experiment harness for the adaptive PIKAN thesis.
It is meant to wrap existing notebooks or future training scripts without
changing the scientific methodology in the protocol.

## Files

- `experiment_manifest.yaml`: protocol-aligned experiment contract.
- `run_experiment.py`: records a durable run summary and logs to MLflow if installed.

## Dry run

```powershell
python src/harness_loop/run_experiment.py
```

The dry run writes a local memory entry under `tmp/harness_runs`.

## With a training command

```powershell
python src/harness_loop/run_experiment.py --command python path/to/train.py --config path/to/config.yaml
```

The training script should eventually write real metrics and artifacts. The
runner currently captures command status, stdout/stderr tails, elapsed time, a
manifest snapshot, and MLflow parameters/tags.

The manifest is an instance of the research ontology in `context/`. Its
`experiment_id`, `problem_id`, `model_id`, and `focalization_id` provide stable
references for agent tasks and notebook metadata.

## Recommended integration path

1. Parameterize one existing baseline notebook as a script or notebook execution.
2. Fill `experiment_manifest.yaml` with the actual model/training values.
3. Run at least two seeds per hypothesis.
4. Inspect MLflow and `tmp/harness_runs`.
5. Accept/reject the iteration using the gates in the manifest.
