"""Independent checkpoint evaluator checks for Heat and Burgers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # The base environment need not install the SciML stack.
    raise unittest.SkipTest("PyTorch is required for checkpoint validation tests") from exc

from evals.checkpoint_validation import evaluate, evaluate_model
from src.research_pipeline.sciml.model import RBFPIKAN


class ExactHeat(torch.nn.Module):
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x, t = coords[:, 0:1], coords[:, 1:2]
        return torch.sin(torch.pi * x) * torch.exp(-0.1 * torch.pi**2 * t)


class ExactBurgers(torch.nn.Module):
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x, t = coords[:, 0:1], coords[:, 1:2]
        return -0.5 * torch.tanh(0.5 * (x + 0.25) / (2 * 0.02))


class PerturbedHeat(torch.nn.Module):
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return ExactHeat()(coords) + coords[:, 1:2]


class NonFiniteModel(torch.nn.Module):
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return coords[:, 0:1] * torch.tensor(float("nan"), dtype=coords.dtype)


class TestCheckpointValidation(unittest.TestCase):
    def test_exact_heat_has_negligible_field_residual_and_trace_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = evaluate_model(ExactHeat().double(), "heat", tmp)

            self.assertLess(metrics["field_relative_l2"], 1e-12)
            self.assertLess(metrics["residual_rmse_abs"], 1e-12)
            self.assertAlmostEqual(metrics["field_rmse_normalized"], metrics["field_rmse_abs"])
            self.assertLess(metrics["residual_mean_abs_normalized"], 1e-12)
            self.assertLess(metrics["initial_max_abs_error"], 1e-12)
            self.assertLess(metrics["boundary_max_abs_error"], 1e-12)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
            self.assertTrue((Path(tmp) / "validation.npz").is_file())
            self.assertTrue((Path(tmp) / "validation.json").is_file())
            report = json.loads((Path(tmp) / "validation.json").read_text())
            self.assertFalse(report["final_test_used"])
            self.assertFalse(report["scientific_accepted"])
            with np.load(Path(tmp) / "validation.npz") as artifact:
                self.assertEqual(artifact["prediction"].shape, (129, 65))
                self.assertFalse(bool(artifact["final_test_used"]))
                self.assertFalse(bool(artifact["scientific_accepted"]))
            self.assertTrue((Path(tmp) / "validation.html").is_file())

    def test_exact_burgers_reports_front_and_complement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = evaluate_model(ExactBurgers().double(), "burgers", tmp)

            self.assertLess(metrics["field_relative_l2"], 1e-12)
            self.assertLess(metrics["residual_rmse_abs"], 1e-10)
            self.assertLess(metrics["front_residual_rmse_abs"], 1e-10)
            self.assertLess(metrics["complement_residual_rmse_abs"], 1e-10)
            self.assertAlmostEqual(metrics["field_rmse_normalized"], metrics["field_rmse_abs"] / 0.5)
            self.assertIn("front_field_rmse_abs", metrics)
            self.assertIn("complement_field_rmse_abs", metrics)
            self.assertIn("front_field_rmse_normalized", metrics)
            self.assertIn("complement_field_rmse_normalized", metrics)
            self.assertIn("front_residual_p95_abs", metrics)
            self.assertIn("complement_residual_p95_abs", metrics)

    def test_field_and_physics_metrics_respond_to_a_perturbation(self) -> None:
        with tempfile.TemporaryDirectory() as exact_dir, tempfile.TemporaryDirectory() as perturbed_dir:
            exact_metrics = evaluate_model(ExactHeat().double(), "heat", exact_dir)
            perturbed_metrics = evaluate_model(PerturbedHeat().double(), "heat", perturbed_dir)

            self.assertGreater(perturbed_metrics["field_relative_l2"], exact_metrics["field_relative_l2"] + 0.01)
            self.assertGreater(perturbed_metrics["residual_rmse_abs"], exact_metrics["residual_rmse_abs"] + 0.5)
            self.assertAlmostEqual(
                perturbed_metrics["residual_mean_abs_normalized"],
                perturbed_metrics["residual_mean_abs"] / 1.1,
            )

    def test_non_finite_candidate_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "non-finite"):
                evaluate_model(NonFiniteModel().double(), "heat", tmp)

    def test_checkpoint_entrypoint_loads_rbf_state_and_returns_numeric_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            model = RBFPIKAN("heat").double()
            torch.save({"problem": "heat", "state_dict": model.state_dict(), "step": 7}, checkpoint)

            metrics = evaluate(checkpoint, root / "out")

            self.assertTrue(metrics)
            self.assertTrue(all(isinstance(value, (int, float)) and torch.isfinite(torch.tensor(value)) for value in metrics.values()))
            report = json.loads((root / "out" / "validation.json").read_text())
            self.assertEqual(report["step"], 7)
            self.assertEqual(report["problem"], "heat")


if __name__ == "__main__":
    unittest.main()
