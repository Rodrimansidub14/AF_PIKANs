# Research/production pipeline

## Remote infrastructure

The portable Kaggle/Colab execution layer is documented in
[`remote/README.md`](remote/README.md). Start with its offline infrastructure
smoke test. It does not train a PIKAN or produce scientific evidence.

`gate` now checks finite numeric values against explicit `threshold` values
(`minimize`: <=, `maximize`: >=). Presence alone is not a scientific success.
`accepted` is always false until a separate human decision; `technical_pass`
reports only single-run numerical checks. Missing thresholds are inconclusive.
Cross-configuration and baseline-cost checks are not inferred from RNG seeds.

This folder is the notebook replacement layer. Notebooks can still exist as
scratchpads, but reproducible thesis evidence should come from this pipeline.
The ontology and agent permissions live in `context/`; this folder validates
the concrete experiment manifest and evaluates its gates.

## Target workflow

```text
manifest -> harness run -> metrics/artifacts -> gate -> accepted/rejected evidence -> writing
```

The researcher keeps the hard thinking: physics, math, interpretation, and
writing. The pipeline absorbs repetitive execution, bookkeeping, comparison,
and provenance.

## Commands

Validate a manifest:

```powershell
python src/research_pipeline/pipeline.py validate --manifest src/harness_loop/experiment_manifest.yaml
```

Run a script through the harness:

```powershell
python src/research_pipeline/pipeline.py run-local --manifest src/harness_loop/experiment_manifest.yaml --command python src/research_pipeline/examples/train_stub.py
```

Evaluate gates from a run summary:

```powershell
python src/research_pipeline/pipeline.py gate --manifest src/harness_loop/experiment_manifest.yaml --summary tmp/harness_runs/H001/<timestamp>/run_summary.json
```

Create a bounded evolution proposal:

```powershell
python src/research_pipeline/pipeline.py evolve-proposal --manifest src/harness_loop/experiment_manifest.yaml --summary tmp/harness_runs/H001/<timestamp>/run_summary.json
```

## AlphaEvolve-inspired layer

The `evolve-proposal` command is intentionally conservative. It does not edit
model code. It reads failed metrics and proposes bounded changes inside the
manifest's `focalization.bounded_edit_surface`.

That is the safe version of the AlphaEvolve idea for this thesis:

- candidate generation is allowed;
- automated evaluators/gates decide whether a candidate survives;
- accepted code/config changes require human review;
- all claims must trace back to run records and artifacts.
