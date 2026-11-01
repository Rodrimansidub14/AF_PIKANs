"""Package the existing v5 Heat runner for a B-spline-only Kaggle campaign."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
RUN = Path(__file__).resolve().parent
SLUG = "rodrimandub/af-pikans-heat-b-spline-completion-follow-up-20260929"
SESSION_CAP = 6480
RUN_CAP = 1930
WAVE_OVERHEAD = 15
CONFIRMATION_CAP = 30
SOURCES = ["rodrimandub/af-pikans-heat-tier2-torch-pilot-t4x2"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"Expected one source fragment; found {text.count(old)}: {old[:120]}")
    return text.replace(old, new, 1)


def main() -> None:
    generated = ROOT / "var/agent_runs/baseline-competitive-2026-09-27/s2/heat-stepmatched-s2.py"
    template_path = ROOT / "var/agent_runs/baseline-competitive-2026-09-27/s2/heat_stepmatched_campaign.py"
    generated_tree = ast.parse(generated.read_text(encoding="utf-8"))
    payload_assignment = next(
        node for node in generated_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PAYLOAD_B64" for target in node.targets)
    )
    payload = ast.literal_eval(payload_assignment.value)
    campaign = template_path.read_text(encoding="utf-8")

    replacements = [
        ('KERNEL_SLUG = "rodrimandub/af-pikans-heat-step-matched-s2-20260928"', f'KERNEL_SLUG = "{SLUG}"'),
        ('SESSION_CAP = 4500.0', f'SESSION_CAP = {float(SESSION_CAP):.1f}'),
        ('RUN_CAP = 680.0', f'RUN_CAP = {float(RUN_CAP):.1f}'),
        ('WAVE_OVERHEAD = 20.0', f'WAVE_OVERHEAD = {float(WAVE_OVERHEAD):.1f}'),
        ('CONFIRMATION_WAVE_CAP = 120.0', f'CONFIRMATION_WAVE_CAP = {float(CONFIRMATION_CAP):.1f}'),
        ('DEVICE_CAP = 9000.0', 'DEVICE_CAP = 13000.0'),
        ('ARMS = ["rbf", "bspline"]', 'ARMS = ["bspline"]'),
        ('/kaggle/working/heat-stepmatched-s2', '/kaggle/working/heat-bspline-completion-followup'),
        ('"campaign": "heat-stepmatched-s2-2026-09-28"', '"campaign": "heat-bspline-completion-followup-2026-09-29"'),
        ('"session": "S2",', '"session": "S2-B-spline-completion-followup-2026-09-29",'),
        ('"session_quota_cap_hours": 1.25,', '"session_quota_cap_hours": 1.8,'),
    ]
    for old, new in replacements:
        campaign = replace_once(campaign, old, new)

    old_schedule = '''for index, seed in enumerate(SEEDS):
    remaining_waves = len(SEEDS) - index - 1
    reserve = remaining_waves * TRAINING_WAVE_WALL + 5 * (CONFIRMATION_WAVE_CAP + WAVE_OVERHEAD) + 40.0
    run_training_wave([(seed, "rbf", RUN_CAP), (seed, "bspline", RUN_CAP)], future_reserve_seconds=reserve)
save_ledger("training_stage_finished")'''
    new_schedule = '''TRAINING_BATCHES = [SEEDS[index:index + 2] for index in range(0, len(SEEDS), 2)]
CONFIRMATION_COUNT = len(SEEDS)
CONFIRMATION_WAVES = (CONFIRMATION_COUNT + 1) // 2
CONFIRMATION_BUDGET = CONFIRMATION_WAVES * (CONFIRMATION_WAVE_CAP + WAVE_OVERHEAD) + 40.0
for index, seeds in enumerate(TRAINING_BATCHES):
    remaining_waves = len(TRAINING_BATCHES) - index - 1
    reserve = remaining_waves * TRAINING_WAVE_WALL + CONFIRMATION_BUDGET
    run_training_wave([(seed, "bspline", RUN_CAP) for seed in seeds], future_reserve_seconds=reserve)
save_ledger("training_stage_finished")'''
    campaign = replace_once(campaign, old_schedule, new_schedule)

    old_confirmation = '''confirmations = [(seed, arm) for seed in SEEDS for arm in ARMS]
confirm_waves = (len(confirmations) + 1) // 2
confirmation_budget = confirm_waves * (CONFIRMATION_WAVE_CAP + WAVE_OVERHEAD) + 40.0'''
    new_confirmation = '''confirmations = [(seed, "bspline") for seed in SEEDS]
confirm_waves = (len(confirmations) + 1) // 2
confirmation_budget = confirm_waves * (CONFIRMATION_WAVE_CAP + WAVE_OVERHEAD) + 40.0'''
    campaign = replace_once(campaign, old_confirmation, new_confirmation)

    old_completion = '''all_training_complete = len(training) == 10 and all(
    item.get("adam_updates") == 20_000 and item.get("lbfgs_stop_reason") in accepted_stop_reasons for item in training
)
campaign_status = "complete" if all_training_complete and len(scored) == 10 else "partial"
ledger["completion"] = {
    "training_runs": len(training),
    "runs_reaching_adam_20000": sum(item.get("adam_updates") == 20_000 for item in training),
    "confirmation_scores": len(scored),
    "scientific_accepted": False,
    "reference_accepted": False,
}'''
    new_completion = '''adam_complete = [item for item in training if item.get("adam_updates") == 20_000]
lbfgs_terminal = [item for item in training if item.get("lbfgs_stop_reason") in accepted_stop_reasons]
all_training_complete = len(training) == len(SEEDS) and len(adam_complete) == len(SEEDS) and len(lbfgs_terminal) == len(SEEDS)
campaign_status = "complete" if all_training_complete and len(scored) == len(SEEDS) else "partial"
ledger["completion"] = {
    "training_runs": len(training),
    "requested_seeds": len(SEEDS),
    "runs_reaching_adam_20000": len(adam_complete),
    "runs_with_terminal_lbfgs_stop": len(lbfgs_terminal),
    "runs_missing_adam_target": [item.get("seed") for item in training if item.get("adam_updates") != 20_000],
    "confirmation_scores": len(scored),
    "optional_deepxde_reference": "skipped; follow-up cap allocated to DFG and B-spline completion",
    "scientific_accepted": False,
    "reference_accepted": False,
}'''
    campaign = replace_once(campaign, old_completion, new_completion)

    campaign = campaign.replace('"session_quota_cap_hours": 1.25,', '"session_quota_cap_hours": 1.8,')
    campaign = replace_once(campaign, 'PAYLOAD_B64 = "__S2_PAYLOAD_B64__"', f"PAYLOAD_B64 = {payload!r}")
    (RUN / "heat_bspline_followup.py").write_text(campaign, encoding="utf-8", newline="\n")

    metadata = {
        "id": SLUG,
        "title": "AF PIKANs Heat B-spline Completion Follow-up 20260929",
        "code_file": "heat_bspline_followup.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": SOURCES,
        "model_sources": [],
    }
    (RUN / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "session": "S2-B-spline-completion-followup-2026-09-29",
        "kernel_slug": SLUG,
        "private": True,
        "session_cap_seconds": SESSION_CAP,
        "per_seed_cap_seconds": RUN_CAP,
        "training_schedule": "five B-spline seeds 901-905; two T4 workers in parallel; no Adam step reduction",
        "pool_source_kernel": SOURCES[0],
        "pool_hashes": {
            "901": "318cdd951dc76a19ce0be65dd170beaffdf40741d4e11301abda5a8b2fe18f51",
            "902": "c89cd6ae60703dcb8cdf04448e2d50f8cdc02324446c1807dcfb3a2145b926d6",
            "903": "0e3557a470006da65b348dd066ba73ccad768c6682e3c337c5392c1e8bc6d4e7",
            "904": "f3067f1997a1b17947fc3910fd4f70db0e5fded96cb87ef3f87a12f43ea05825",
            "905": "dc567fc082828a68ac5483975659f155644d9040cf01916147758e1b5e6736b0",
        },
        "optimizer": {
            "adam_updates": 20000,
            "learning_rate": 0.001,
            "lbfgs_backend": "PyTorch L-BFGS using DeepXDE 1.12.2 settings",
            "maxiter": 15000,
            "maxfun": 18750,
            "history_size": 100,
            "tolerance_grad": 1e-8,
            "tolerance_change": 0.0,
            "line_search_fn": "strong_wolfe",
            "chunk_iter": 25,
            "chunk_max_eval": 32,
        },
        "confirmation_grid": "heat-confirmatory-512x256-v1",
        "optional_reference": "DeepXDE alpha=0.1 skipped; no residual cap after planning both approved items",
        "scientific_accepted": False,
        "reference_accepted": False,
        "source_hashes": {
            "campaign_entrypoint_sha256": digest(RUN / "heat_bspline_followup.py"),
            "source_template_sha256": digest(template_path),
            "prior_campaign_entrypoint_sha256": digest(generated),
        },
    }
    (RUN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
