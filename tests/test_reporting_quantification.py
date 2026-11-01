import json
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class ReportingQuantificationTests(unittest.TestCase):
    def test_tables_report_saved_measurements_and_descriptive_seed_counts(self):
        from src.research_pipeline.remote.reporting import (
            aggregate_quantitative_table,
            long_quantitative_table,
            quantitative_table,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_errors = {101: 0.1, 102: 0.3}
            for seed, error in expected_errors.items():
                problem = root / f"seed-{seed}/heat"
                problem.mkdir(parents=True)
                (problem / "sampling_signals.json").write_text(json.dumps({
                    "interior_points": 4096,
                    "sobol_centered_discrepancy": 0.0007,
                    "nearest_neighbor_p95": 0.012,
                }))
                (problem / "diagnostic_signals.json").write_text(json.dumps({
                    "residual/normalized_rms": 0.23,
                    "rbf/occupancy_mean": 0.61,
                }))
                (problem / "training_summary.json").write_text(json.dumps({
                    "steps": 200, "training_and_diagnostics_sec": 12.5,
                    "diagnostic_sec": 1.5, "parameters": 128,
                    "peak_gpu_bytes": 1024,
                }))
                (problem / "validation.json").write_text(json.dumps({
                    "metrics": {"field_rmse_abs": error, "field_relative_l2": error / 10},
                }))

            measured = quantitative_table(root)
            direct = long_quantitative_table(measured)
            errors = direct[(direct["family"] == "validation")
                            & (direct["signal"] == "field_rmse_abs")]
            self.assertEqual(errors["value"].tolist(), [0.1, 0.3])
            self.assertEqual(errors["unit"].tolist(), ["U", "U"])
            sampled_points = direct[(direct["family"] == "sampling")
                                    & (direct["signal"] == "interior_points")]
            self.assertEqual(sampled_points["value"].tolist(), [4096.0, 4096.0])
            self.assertEqual(sampled_points["unit"].tolist(), ["count", "count"])

            summary = aggregate_quantitative_table(measured)
            self.assertEqual(summary.loc["heat", ("validation_field_rmse_abs", "count")], 2)
            self.assertAlmostEqual(
                summary.loc["heat", ("validation_field_rmse_abs", "mean")], 0.2
            )

    def test_allen_cahn_measurements_are_included_without_a_field_error_column(self):
        from src.research_pipeline.remote.reporting import (
            long_quantitative_table,
            quantitative_table,
        )

        with tempfile.TemporaryDirectory() as temporary:
            problem = Path(temporary) / "seed-101/allen_cahn"
            problem.mkdir(parents=True)
            (problem / "sampling_signals.json").write_text(json.dumps({
                "interior_count": 8192, "heldout_count": 16384,
            }))
            (problem / "diagnostic_signals.json").write_text(json.dumps({
                "residual/normalized_rms": 0.25,
            }))
            (problem / "training_summary.json").write_text(json.dumps({
                "steps": 20000, "training_and_diagnostics_sec": 900,
                "diagnostic_sec": 12, "parameters": 3648,
            }))
            (problem / "validation.json").write_text(json.dumps({
                "field_error_status": "locked_pending_reference",
                "metrics": {"normalized_residual_mean": 0.2,
                            "energy_relative_change": -0.1},
            }))

            measured = quantitative_table(Path(temporary))

        self.assertEqual(measured["problem"].tolist(), ["allen_cahn"])
        self.assertIn("validation_normalized_residual_mean", measured.columns)
        self.assertNotIn("validation_field_relative_l2", measured.columns)

        signals = long_quantitative_table(measured)
        energy = signals[signals["signal"] == "energy_relative_change"]
        self.assertEqual(energy["unit"].tolist(), ["nondimensional Allen–Cahn energy"])

    def test_results_notebook_finds_extracted_repository_run_directory(self):
        import numpy as np
        from src.research_pipeline.remote.reporting import write_results_notebook

        class Figure:
            def to_json(self):
                return '{"data":[],"layout":{}}'

            def to_html(self, **_kwargs):
                return "<div>plot</div>"

        express = types.ModuleType("plotly.express")
        express.scatter = lambda **_kwargs: Figure()
        plotly = types.ModuleType("plotly")
        plotly.express = express

        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "runs"
            for problem_name in ("heat", "burgers"):
                problem = runs / f"seed-101/{problem_name}"
                problem.mkdir(parents=True)
                np.savez(problem / "pools.npz", interior=np.array([[0.0, 0.0], [1.0, 1.0]]))
                (problem / "sampling_signals.json").write_text(json.dumps({"interior_points": 2}))
                (problem / "diagnostic_signals.json").write_text(json.dumps({}))
                (problem / "training_summary.json").write_text(json.dumps({"steps": 1}))
                (problem / "validation.json").write_text(json.dumps({"metrics": {}}))
            with patch.dict(sys.modules, {"plotly": plotly, "plotly.express": express}), \
                 patch("src.research_pipeline.remote.reporting.training_figures", return_value={}), \
                 patch("src.research_pipeline.remote.reporting.field_figures", return_value={}):
                notebook_path = write_results_notebook(runs, Path(temporary) / "analysis", "bundle-id")

            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            base_path_code = "".join(notebook["cells"][1]["source"])
            self.assertIn("download-extracted/runs", base_path_code)

    def test_allen_cahn_results_notebook_keeps_field_error_locked_and_plots_both_pools(self):
        import numpy as np
        from src.research_pipeline.remote.reporting import write_results_notebook

        class Figure:
            def to_json(self):
                return '{"data":[],"layout":{}}'

            def to_html(self, **_kwargs):
                return "<div>plot</div>"

        express = types.ModuleType("plotly.express")
        express.scatter = lambda **_kwargs: Figure()
        plotly = types.ModuleType("plotly")
        plotly.express = express

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            problem = runs / "seed-101/allen_cahn"
            problem.mkdir(parents=True)
            np.savez(problem / "pools.npz",
                     interior=np.array([[-.5, .2], [.5, .8]]),
                     heldout=np.array([[-.25, .3], [.25, .7]]))
            (problem / "sampling_signals.json").write_text(json.dumps({
                "interior_count": 8192, "heldout_count": 16384,
            }))
            (problem / "diagnostic_signals.json").write_text(json.dumps({
                "residual/normalized_rms": .25,
            }))
            (problem / "training_summary.json").write_text(json.dumps({"steps": 20000}))
            (problem / "validation.json").write_text(json.dumps({"metrics": {
                "normalized_residual_mean": .2,
                "energy_relative_change": -.1,
            }}))
            with patch.dict(sys.modules, {"plotly": plotly, "plotly.express": express}), \
                 patch("src.research_pipeline.remote.reporting.training_figures", return_value={}), \
                 patch("src.research_pipeline.remote.reporting.field_figures", return_value={}):
                notebook_path = write_results_notebook(runs, root / "analysis", "bundle-id")

            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            cells = notebook["cells"]
            intro = "".join(cells[0]["source"])
            point_plot = next("".join(cell["source"]) for cell in cells
                              if cell["cell_type"] == "code"
                              and "heldout = pools['heldout']" in "".join(cell["source"]))

        self.assertIn("Error de campo y L2 permanecen bloqueados", intro)
        self.assertNotIn("error absoluto usa color logarítmico", intro)
        self.assertIn("heldout", point_plot)
        self.assertIn("independent training and held-out", point_plot)

    def test_dashboard_discovers_download_extracted_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            runs = repository / "var/agent_runs/heat-burgers-002-t4x2/download-extracted/runs"
            problem = runs / "seed-101/heat"
            problem.mkdir(parents=True)
            (problem / "sampling_signals.json").write_text(json.dumps({"interior_points": 8}))
            (problem / "diagnostic_signals.json").write_text(json.dumps({}))
            (problem / "training_summary.json").write_text(json.dumps({"steps": 1}))
            (problem / "validation.json").write_text(json.dumps({"metrics": {"field_rmse_abs": 0.2}}))
            dashboard_path = Path(__file__).resolve().parents[1] / "src/research_pipeline/remote/analysis_dashboard.py"
            module_spec = importlib.util.spec_from_file_location("af_pikan_dashboard_test", dashboard_path)
            module = importlib.util.module_from_spec(module_spec)
            with patch.dict(os.environ, {"AF_PIKAN_RUNS": str(runs)}):
                module_spec.loader.exec_module(module)
            with patch.dict(os.environ, {"AF_PIKAN_RUNS": ""}):
                selected = module._runs_path(repository_root=repository)

            self.assertEqual(selected, runs.resolve())


if __name__ == "__main__":
    unittest.main()
