import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.research_pipeline.pipeline import validate_manifest
from src.research_pipeline.remote.notebook import render_notebook
from src.sciml.campaign import assert_exploratory_launch_ready


MANIFEST_PATH = Path("experiments/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/manifest.json")
TRACKED_NOTEBOOK = Path("notebooks/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/allen-cahn-rbf-sobol-exploratory.ipynb")


def test_allen_cahn_manifest_fixes_exploratory_budget_and_keeps_field_error_locked():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert validate_manifest(manifest) == []
    assert manifest["iteration"]["protocol_phase"] == "exploratory"
    assert manifest["problem"]["sequence"] == ["allen_cahn"]
    assert manifest["training"]["seeds"] == [101, 102, 103]
    assert manifest["training"]["interior_points"] == 8192
    assert manifest["training"]["heldout_points"] == 16384
    assert manifest["training"]["seconds_per_problem"] == 1200
    assert manifest["budget"]["approved"] is True
    assert manifest["research_status"]["scientific_accepted"] is False
    assert manifest["problem"]["allen_cahn"]["field_error_status"] == "locked_pending_reference"
    assert manifest["acceptance_gates"]["metrics"]["allen_cahn/normalized_residual_mean"]["threshold"] is None


def test_allen_cahn_launch_gate_requires_explicit_run_authorization_and_budget():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert_exploratory_launch_ready(manifest)
    blocked = copy.deepcopy(manifest)
    blocked["budget"]["approved"] = False
    try:
        assert_exploratory_launch_ready(blocked)
    except RuntimeError as exc:
        assert "budget" in str(exc).lower()
    else:
        raise AssertionError("unapproved Allen–Cahn budget must fail closed")


def test_remote_campaign_entrypoint_imports_when_executed_as_a_file():
    result = subprocess.run(
        [sys.executable, "src/sciml/campaign.py", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout


def test_allen_cahn_notebook_reports_physics_only_artifacts_and_compiles():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    decisions = MANIFEST_PATH.with_name("parameter-decisions.md")
    manifest["parameter_decisions_markdown"] = decisions.read_text(encoding="utf-8")
    spec = {"bundle_id": "a" * 64, "evidence_kind": "research",
            "runtime_dependencies": [], "setup_timeout_sec": 900,
            "max_parallel_seeds": 2}
    notebook = render_notebook(spec, {}, manifest)
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    run_cell = "".join(cells["parallel-run"]["source"])
    sampling_cell = "".join(cells["sampling-signals"]["source"])
    field_cell = "".join(cells["field-maps"]["source"])
    assert "assert_exploratory_launch_ready" in run_cell
    assert "heldout_points" in sampling_cell
    assert "sampling_signals('allen_cahn', data)" in sampling_cell
    assert "parameter-table" in cells
    assert "field_reference_used" not in field_cell
    assert "error de campo permanece bloqueado" in "".join(cells["fields-note"]["source"])


def test_tracked_allen_cahn_notebook_is_an_exploratory_development_notebook():
    notebook = json.loads(TRACKED_NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["metadata"]["research"]["role"] == "exploration"
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    assert {"protocol", "parameter-table", "sampling-signals", "parallel-run",
            "quantitative-signals", "visualizations", "field-maps", "closure"} <= set(cells)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    overview = "".join(cells["overview"]["source"]).lower()
    sampling = "".join(cells["sampling-signals"]["source"])
    assert "afirma sota" in overview
    assert "heldout" in sampling
    assert "rad" not in sampling.lower()
