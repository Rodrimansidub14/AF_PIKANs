import json
import tempfile
from pathlib import Path

import numpy as np

from src.research_pipeline.sciml.allen_cahn_evaluation import (
    allen_cahn_energy,
    evaluate_allen_cahn,
)
from src.research_pipeline.sciml.model import build_model
from src.research_pipeline.sciml.sampling import pools


def test_allen_cahn_energy_uses_declared_double_well_functional():
    x = np.linspace(-1.0, 1.0, 17)
    u = np.zeros_like(x)
    ux = np.zeros_like(x)
    assert allen_cahn_energy(x, u, ux, diffusion=0.001, reaction=5.0) == 2.5


def test_allen_cahn_heldout_evaluation_recomputes_signals_and_keeps_field_error_locked():
    data = pools("allen_cahn", seed=101, n=64, boundary=16, heldout=128)
    model = build_model("allen_cahn", {"id": "rbf_fixed"}).double()
    config = {
        "diffusion": 0.001,
        "reaction": 5.0,
        "residual_scale": 6.001,
        "evaluation_times": [0.25, 0.5, 0.75, 1.0],
        "diagnostic_grid": {"nx": 17, "nt": 9},
        "energy_points": 33,
        "evaluation_chunk_size": 32,
    }
    with tempfile.TemporaryDirectory(dir="var/agent_runs") as tmp:
        metrics = evaluate_allen_cahn(model, config, Path(tmp), data)
        report = json.loads((Path(tmp) / "validation.json").read_text())
        assert report["field_error_status"] == "locked_pending_reference"
        assert "field_relative_l2" not in metrics
        assert metrics["heldout_point_count"] == 128
        assert metrics["initial_trace_max_abs_error"] < 1e-12
        assert metrics["boundary_trace_max_abs_error"] < 1e-12
        assert np.isfinite(metrics["heldout_residual_rmse_abs"])
        assert (Path(tmp) / "validation.npz").is_file()
        assert (Path(tmp) / "validation.html").is_file()
        with np.load(Path(tmp) / "validation.npz") as saved:
            assert set(saved.files) >= {"heldout", "heldout_residual", "energy_times", "energy"}
        assert len(report["energy"]) == 5
        assert "field_relative_l2" not in report["metrics"]
