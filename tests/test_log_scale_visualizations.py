import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class LogScaleVisualizationTests(unittest.TestCase):
    def test_training_figures_keep_only_positive_values_on_log_axes(self):
        from src.research_pipeline.remote.reporting import training_figures

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "seed-101" / "heat"
            run.mkdir(parents=True)
            history = [
                {"elapsed_sec": 1.0, "train_loss": 0.25,
                 "gradient_norm": 0.0, "residual/normalized_rms": 0.5},
                {"elapsed_sec": 2.0, "train_loss": 0.0,
                 "gradient_norm": 0.1, "residual/normalized_rms": 0.0},
                {"elapsed_sec": 3.0, "train_loss": 0.0625,
                 "gradient_norm": 0.05, "residual/normalized_rms": 0.125},
            ]
            (run / "history.jsonl").write_text(
                "\n".join(json.dumps(row) for row in history) + "\n",
                encoding="utf-8",
            )

            figures = training_figures(root)

        for figure in figures.values():
            self.assertEqual(figure.layout.yaxis.type, "log")
            for trace in figure.data:
                self.assertTrue(all(value > 0 for value in trace.y))

    def test_final_error_field_is_log_color_scaled_and_signed_fields_stay_linear(self):
        from src.research_pipeline.remote.reporting import field_figures

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "seed-101" / "heat"
            run.mkdir(parents=True)
            prediction = np.array([[-0.5, 0.0], [0.5, 0.25]])
            reference = np.array([[-0.45, 0.0], [0.4, 0.2]])
            error = np.abs(prediction - reference)
            np.savez_compressed(
                run / "validation.npz",
                x=np.array([0.0, 1.0]),
                t=np.array([0.0, 1.0]),
                prediction=prediction,
                reference=reference,
                absolute_error=error,
            )

            figure = field_figures(root)["validation-fields-heat-seed-101"]

        np.testing.assert_array_equal(figure.data[0].z, prediction)
        np.testing.assert_array_equal(figure.data[1].z, reference)
        plotted_error = np.asarray(figure.data[2].z, dtype=float)
        self.assertEqual(plotted_error.shape, error.shape)
        np.testing.assert_allclose(plotted_error[np.isfinite(plotted_error)],
                                   np.log10(error[error > 0]))
        self.assertEqual(np.count_nonzero(np.isnan(plotted_error)), 1)
        self.assertIn("log", figure.layout.coloraxis2.colorbar.title.text.lower())
        self.assertLess(figure.layout.coloraxis2.cmin, figure.layout.coloraxis2.cmax)

    def test_wandb_receives_interactive_log_scale_training_and_validation_figures(self):
        from src.research_pipeline.remote.reporting import log_wandb_figures

        class FakeRun:
            def __init__(self):
                self.logged = []

            def log(self, payload):
                self.logged.append(payload)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.npz"
            prediction = np.array([[-0.2, 0.0], [0.3, 0.4]])
            reference = np.array([[-0.1, 0.0], [0.25, 0.35]])
            np.savez_compressed(
                validation,
                x=np.array([0.0, 1.0]),
                t=np.array([0.0, 1.0]),
                prediction=prediction,
                reference=reference,
                absolute_error=np.abs(prediction - reference),
            )
            history = [
                {"elapsed_sec": 1.0, "train_loss": 0.2,
                 "gradient_norm": 0.5, "residual/normalized_rms": 0.4},
                {"elapsed_sec": 2.0, "train_loss": 0.05,
                 "gradient_norm": 0.1, "residual/normalized_rms": 0.08},
            ]
            run = FakeRun()

            log_wandb_figures(run, history, validation, "heat")

        self.assertEqual(len(run.logged), 1)
        self.assertIn("plots/training-losses-log", run.logged[0])
        self.assertIn("plots/training-signals-log", run.logged[0])
        self.assertIn("plots/validation-error-log", run.logged[0])
        self.assertEqual(run.logged[0]["plots/training-losses-log"].layout.yaxis.type, "log")
        self.assertEqual(run.logged[0]["plots/training-signals-log"].layout.yaxis.type, "log")
        self.assertIn("log", run.logged[0]["plots/validation-error-log"].layout.coloraxis2.colorbar.title.text.lower())

    def test_allen_cahn_visualizations_do_not_expect_a_reference_field(self):
        from src.research_pipeline.remote.reporting import field_figures, log_wandb_figures

        class FakeRun:
            def __init__(self):
                self.logged = []

            def log(self, payload):
                self.logged.append(payload)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            problem = root / "seed-101" / "allen_cahn"
            problem.mkdir(parents=True)
            validation = problem / "validation.npz"
            prediction = np.array([[-0.5, -0.25], [0.25, 0.5]])
            residual = np.array([[0.01, 0.1], [0.001, 0.02]])
            np.savez_compressed(validation, x=np.array([-1.0, 1.0]),
                                t=np.array([0.0, 1.0]), prediction=prediction,
                                residual=residual, energy_times=np.array([0.0, 1.0]),
                                energy=np.array([1.0, 0.5]),
                                field_reference_used=np.asarray(False))
            figures = field_figures(root)
            run = FakeRun()
            log_wandb_figures(
                run,
                [{"elapsed_sec": 1.0, "train_loss": 1.0,
                  "residual/normalized_rms": 0.5}],
                validation,
                "allen_cahn",
            )

        self.assertIn("validation-fields-allen_cahn-seed-101", figures)
        self.assertIn("plots/validation-physics-only", run.logged[0])
        self.assertIn("plots/energy-trajectory", run.logged[0])
        self.assertEqual(len(run.logged[0]["plots/validation-physics-only"].data), 2)
        self.assertIn("log", run.logged[0]["plots/validation-physics-only"].data[1].colorbar.title.text.lower())

    def test_training_figures_only_render_problems_present_in_the_campaign(self):
        from src.research_pipeline.remote.reporting import training_figures

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "seed-101" / "allen_cahn"
            run.mkdir(parents=True)
            (run / "history.jsonl").write_text(json.dumps({
                "elapsed_sec": 1.0, "train_loss": 0.2,
                "gradient_norm": 0.5, "residual/normalized_rms": 0.3,
            }) + "\n")
            figures = training_figures(Path(temporary))

        for figure in figures.values():
            self.assertTrue(all("Allen" in item.text for item in figure.layout.annotations))


if __name__ == "__main__":
    unittest.main()
