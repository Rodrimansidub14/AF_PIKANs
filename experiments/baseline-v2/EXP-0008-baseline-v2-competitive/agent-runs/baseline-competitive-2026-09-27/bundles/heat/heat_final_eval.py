"""Post-training CUDA confirmation scorer, invoked only after all training ends."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from af_sciml_model import build_model
from heat_comparison_runner import TutorialFNN
from heat_common_eval import score_torch_model


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def model_for(arm: str) -> torch.nn.Module:
    if arm == "rbf":
        return build_model("heat", {"id": "rbf_fixed"})
    if arm == "bspline":
        return build_model("heat", {"id": "bspline_cubic"})
    if arm == "fnn":
        return TutorialFNN()
    raise ValueError(f"Unknown arm {arm!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training CUDA-only Heat score")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("rbf", "bspline", "fnn"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each final-score worker must see exactly one assigned T4.")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if "t4" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError(f"Expected T4, got {torch.cuda.get_device_name(0)}.")

    # This script is called by a separate notebook cell after the trainer exits.
    # The 512x256 grid is constructed inside this scorer and is never read by training.
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = model_for(args.arm)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device="cuda:0", dtype=torch.float64).eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = score_torch_model(
        model,
        args.out,
        nx=512,
        nt=256,
        device="cuda:0",
        grid_id="heat-confirmatory-512x256-v1",
    )
    torch.cuda.synchronize()
    score_seconds = max(0.0, time.perf_counter() - started)
    report = {
        "status": "DRAFT_UNVERIFIED",
        "score_stage": "post-training confirmation only",
        "arm": args.arm,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "pool_sha256": checkpoint.get("pool_sha256"),
        "grid_id": "heat-confirmatory-512x256-v1",
        "grid": {"nx": 512, "nt": 256},
        "device": torch.cuda.get_device_name(0),
        "evaluation_gpu_seconds_estimate": score_seconds,
        "allocated_gpu_seconds_estimate": score_seconds,
        "gpu_total_memory_bytes": int(props.total_memory),
        "metrics": metrics,
        "scoring_used_for_training_or_selection": False,
        "training_code_imported_protected_grid": False,
        "protected_grid_generated_only_in_this_post_training_scorer": True,
    }
    write_json(args.out / "confirmation-cost.json", report)
    write_json(args.out / "confirmation-metrics.json", {f"heat/{k}": v for k, v in metrics.items()})
    print(json.dumps({"arm": args.arm, "seed": args.seed, "evaluation_gpu_seconds_estimate": score_seconds}))


if __name__ == "__main__":
    main()
