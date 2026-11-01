"""Research/production CLI for reproducible PIKAN experiments."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency: PyYAML. Install with `python -m pip install pyyaml`."
    ) from exc


REQUIRED_TOP_LEVEL_KEYS = {
    "project",
    "iteration",
    "hypothesis",
    "problem",
    "model",
    "training",
    "focalization",
    "acceptance_gates",
    "artifacts",
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors = []
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS.difference(manifest))
    if missing:
        errors.append(f"Missing top-level keys: {', '.join(missing)}")

    project = manifest.get("project", {})
    if project.get("ontology") != "pikan-research":
        errors.append("Missing project.ontology=pikan-research")

    iteration = manifest.get("iteration", {})
    for key in (
        "id",
        "experiment_id",
        "protocol_phase",
        "domain_level",
        "problem_id",
        "model_id",
        "focalization_id",
        "hypothesis_id",
    ):
        if not iteration.get(key):
            errors.append(f"Missing iteration.{key}")

    hypothesis = manifest.get("hypothesis", {})
    for key in ("statement", "expected_effect", "falsification_condition"):
        if not hypothesis.get(key):
            errors.append(f"Missing hypothesis.{key}")

    gates = manifest.get("acceptance_gates", {})
    if not gates.get("metrics"):
        errors.append("Missing acceptance_gates.metrics")

    return errors


def run_local(manifest_path: Path, command: list[str]) -> int:
    harness = Path("src/harness_loop/run_experiment.py")
    full_command = [
        sys.executable,
        str(harness),
        "--manifest",
        str(manifest_path),
        "--command",
        *command,
    ]
    return subprocess.run(full_command, check=False).returncode


def extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("result", {}).get("metrics", {})


def evaluate_gates(manifest: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    metrics = extract_metrics(summary)
    gate_metrics = manifest.get("acceptance_gates", {}).get("metrics", {})
    checks = []
    technical_pass = True

    for name, spec in gate_metrics.items():
        required = bool(spec.get("required", False))
        value = metrics.get(name)
        finite = type(value) in (int, float) and math.isfinite(value)
        threshold = spec.get("threshold")
        valid_threshold = type(threshold) in (int, float) and math.isfinite(threshold)
        direction = spec.get("direction")
        passed = not required and value is None
        reason = "optional_missing" if passed else "invalid_or_missing_metric"
        if finite:
            if valid_threshold and direction in {"minimize", "maximize"}:
                passed = value <= threshold if direction == "minimize" else value >= threshold
                reason = "within_threshold" if passed else "outside_threshold"
            else:
                passed, reason = False, "threshold_not_defined"
        if not passed:
            technical_pass = False
        checks.append(
            {
                "metric": name,
                "required": required,
                "direction": spec.get("direction"),
                "value": value,
                "passed": passed,
                "reason": reason,
            }
        )

    status = summary.get("result", {}).get("status")
    if status != "completed":
        technical_pass = False

    return {
        "accepted": False,
        "technical_pass": technical_pass and bool(gate_metrics),
        "requires_human_review": True,
        "scope": "single_run_thresholds_only",
        "run_status": status,
        "checks": checks,
    }


def evolution_proposal(manifest: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    gate_result = evaluate_gates(manifest, summary)
    failed_required = [
        check["metric"]
        for check in gate_result["checks"]
        if check["required"] and not check["passed"]
    ]
    edit_surface = manifest.get("focalization", {}).get("bounded_edit_surface", [])

    recommendations = []
    if "critical_region_residual_mean" in failed_required:
        recommendations.append(
            {
                "edit": "collocation_sampling_policy",
                "reason": "Critical-region residual is missing or failed; prioritize explicit residual-map logging and region-aware sampling.",
            }
        )
    if "stability_score" in failed_required:
        recommendations.append(
            {
                "edit": "loss_weight_schedule",
                "reason": "Stability evidence is missing or failed; try a slower schedule before changing architecture.",
            }
        )
    if "relative_l2_error" in failed_required:
        recommendations.append(
            {
                "edit": "grid_update_policy",
                "reason": "Global error evidence is missing or failed; compare fixed grid versus conservative grid updates.",
            }
        )

    recommendations = [
        item for item in recommendations if item["edit"] in set(edit_surface)
    ]

    return {
        "type": "bounded_alphaevolve_style_proposal",
        "accepted_current_run": gate_result["accepted"],
        "failed_required_metrics": failed_required,
        "allowed_edit_surface": edit_surface,
        "recommendations": recommendations,
        "requires_human_review": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)

    run = subparsers.add_parser("run-local")
    run.add_argument("--manifest", required=True)
    run.add_argument("--command", nargs=argparse.REMAINDER, required=True)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--manifest", required=True)
    gate.add_argument("--summary", required=True)

    evolve = subparsers.add_parser("evolve-proposal")
    evolve.add_argument("--manifest", required=True)
    evolve.add_argument("--summary", required=True)
    evolve.add_argument("--out")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command_name == "validate":
        errors = validate_manifest(load_yaml(Path(args.manifest)))
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 1

    if args.command_name == "run-local":
        return run_local(Path(args.manifest), args.command)

    manifest = load_yaml(Path(args.manifest))
    summary = load_json(Path(args.summary))

    if args.command_name == "gate":
        print(json.dumps(evaluate_gates(manifest, summary), indent=2))
        return 0

    if args.command_name == "evolve-proposal":
        proposal = evolution_proposal(manifest, summary)
        output = json.dumps(proposal, indent=2)
        if args.out:
            Path(args.out).write_text(output, encoding="utf-8")
        print(output)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
