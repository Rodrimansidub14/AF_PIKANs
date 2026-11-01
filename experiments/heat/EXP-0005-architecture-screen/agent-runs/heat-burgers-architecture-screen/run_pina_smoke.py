"""CPU integration smoke for the three model families and canonical PDE contracts."""

from __future__ import annotations

import json
import math
import sys
import uuid
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from evals.checkpoint_validation import evaluate_model
from src.research_pipeline.sciml.model import build_model
from src.research_pipeline.sciml.train import train
from src.research_pipeline.remote.reporting import log_wandb_figures


class NullRun:
    def log(self, *_args, **_kwargs):
        return None


ROOT = Path(__file__).resolve().parent
OUT = ROOT / ("pina-smoke-" + uuid.uuid4().hex[:8])
MODEL_IDS = ("rbf_fixed", "mlp_tanh", "bspline_cubic")
CONFIG = {
    "seed": 101,
    "steps_per_problem": 4,
    "seconds_per_problem": 120,
    "interior_points": 64,
    "boundary_points": 32,
    "batch_size": 32,
    "learning_rate": 0.001,
    "diagnostic_every": 2,
    "diagnostic_points": 8,
    "sensitivity_points": 2,
    "cpu_threads": 2,
    "accelerator": "cpu",
    "metric_log_every": 2,
}
PROBLEMS = {
    "heat": {
        "alpha": 0.1,
        "residual_scale": 1.1,
        "evaluation_id": "heat",
    },
    "burgers": {
        "definition": "raissi_pinnacle_1d",
        "evaluation_id": "burgers_pinnacle_1d",
        "nu": 0.01 / 3.141592653589793,
        "field_scale": 1.0,
        "residual_scale": 2.0 + 0.01 / 3.141592653589793,
    },
}


def main():
    results = []
    for model_id in MODEL_IDS:
        model_config = {"id": model_id}
        for problem, problem_config in PROBLEMS.items():
            run_dir = OUT / model_id / problem
            cfg = {**CONFIG, "interior_points": 64, "boundary_points": 32}
            checkpoint, summary = train(
                problem,
                cfg,
                run_dir,
                NullRun(),
                model_config=model_config,
                problem_config=problem_config,
            )
            model = build_model(problem, model_config).double()
            payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            validation_dir = run_dir / "validation-smoke"
            measured = evaluate_model(model, problem_config["evaluation_id"], validation_dir)
            history = [
                json.loads(line)
                for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            logged_figures = log_wandb_figures(
                NullRun(), history, validation_dir / "validation.npz", problem,
                output_dir=run_dir / "plots-smoke",
            )
            plot_files = sorted(path.name for path in (run_dir / "plots-smoke").glob("*.html"))
            results.append({
                "model_id": model_id,
                "problem": problem,
                "steps": summary["steps"],
                "parameters": summary["parameters"],
                "validation_metric_count": len(measured),
                "validation_metrics_finite": all(math.isfinite(v) for v in measured.values()),
                "log_scale_plot_count": len(logged_figures),
                "log_scale_html_count": len(plot_files),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke-summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "results": results}, indent=2))


if __name__ == "__main__":
    main()
