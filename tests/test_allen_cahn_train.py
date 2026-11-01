import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.research_pipeline.sciml.model import build_model
from src.research_pipeline.sciml.sampling import pools
from src.research_pipeline.sciml.train import allen_cahn_residual, make_problem, train


class Sink:
    def log(self, *args, **kwargs):
        pass


def test_allen_cahn_residual_matches_declared_diffusion_reaction_sign():
    u = torch.tensor([[0.25], [-0.5]], dtype=torch.float64)
    ut = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    uxx = torch.tensor([[3.0], [-4.0]], dtype=torch.float64)
    measured = allen_cahn_residual(u, ut, uxx, diffusion=0.001, reaction=5.0)
    expected = ut - 0.001 * uxx - 5.0 * (u - u.pow(3))
    torch.testing.assert_close(measured, expected)


def test_allen_cahn_problem_has_only_physics_training_condition():
    data = pools("allen_cahn", seed=101, n=64, heldout=128)
    problem = make_problem("allen_cahn", data, {"diffusion": 0.001})
    assert set(problem.conditions) == {"physics"}


def test_allen_cahn_cpu_smoke_saves_checkpoint_and_finite_signals():
    cfg = {
        "seed": 101,
        "cpu_threads": 2,
        "interior_points": 64,
        "heldout_points": 128,
        "boundary_points": 16,
        "learning_rate": 0.001,
        "batch_size": 16,
        "steps_per_problem": 8,
        "seconds_per_problem": 120,
        "accelerator": "cpu",
        "diagnostic_points": 16,
        "sensitivity_points": 2,
        "diagnostic_every": 4,
        "metric_log_every": 2,
        "sampling": "scrambled_sobol_fixed",
    }
    problem_config = {"diffusion": 0.001, "reaction": 5.0, "residual_scale": 6.001}
    with tempfile.TemporaryDirectory(dir="var/agent_runs") as tmp:
        out = Path(tmp) / "allen-cahn-smoke"
        checkpoint, summary = train(
            "allen_cahn", cfg, out, Sink(), {"id": "rbf_fixed"}, problem_config
        )
        assert checkpoint.is_file()
        assert summary["steps"] == 8
        assert set(summary["examples_seen"]) == {"physics"}
        with np.load(out / "pools.npz") as stored:
            assert stored["interior"].shape == (64, 2)
            assert stored["heldout"].shape == (128, 2)
        signals = json.loads((out / "diagnostic_signals.json").read_text())
        assert np.isfinite(signals["residual/normalized_rms"])

