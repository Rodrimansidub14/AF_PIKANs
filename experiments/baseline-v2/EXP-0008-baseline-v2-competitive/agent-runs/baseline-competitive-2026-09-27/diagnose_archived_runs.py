"""Descriptive diagnostics from archived baseline CSVs; no training or validation.

Run from the AF_PIKANs repository root with any Python 3.11+ interpreter.
"""

import csv
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path("var/agent_runs/baseline-close-2026-09-27/delivery")
OUT = Path("var/agent_runs/baseline-competitive-2026-09-27")


def rows(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summary(values):
    return {"n": len(values), "mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None}


burgers = [row for row in rows("burgers-by-seed.csv") if row["arm"] == "sobol"]
dfg = rows("dfg-by-seed.csv")
ac = rows("allen-cahn-by-seed.csv")
heat = rows("heat-by-seed.csv")

result = {
    "source": str(ROOT),
    "purpose": "descriptive diagnosis from archived results; spatial points are not independent replicas",
    "burgers_sobol": {
        "front_to_complement_field_rmse_ratio": summary(
            [float(r["front_field_rmse_abs"]) / float(r["complement_field_rmse_abs"]) for r in burgers]
        ),
        "front_to_complement_residual_rmse_ratio": summary(
            [float(r["front_residual_rmse_abs"]) / float(r["complement_residual_rmse_abs"]) for r in burgers]
        ),
        "initial_max_abs_error": summary([float(r["initial_max_abs_error"]) for r in burgers]),
        "boundary_max_abs_error": summary([float(r["boundary_max_abs_error"]) for r in burgers]),
    },
    "allen_cahn": {
        "field_relative_l2_provisional": summary([float(r["field_relative_l2"]) for r in ac]),
        "normalized_residual_mean": summary([float(r["normalized_residual_mean"]) for r in ac]),
        "field_error_to_residual_mean_ratio": summary(
            [float(r["field_relative_l2"]) / float(r["normalized_residual_mean"]) for r in ac]
        ),
    },
    "dfg": {
        "near_to_global_continuity_rms_ratio": summary(
            [float(r["heldout/near_cylinder/continuity_rms"]) / float(r["heldout/global/continuity_rms"]) for r in dfg]
        ),
        "near_to_global_momentum_x_rms_ratio": summary(
            [float(r["heldout/near_cylinder/momentum_x_rms"]) / float(r["heldout/global/momentum_x_rms"]) for r in dfg]
        ),
        "near_to_global_momentum_y_rms_ratio": summary(
            [float(r["heldout/near_cylinder/momentum_y_rms"]) / float(r["heldout/global/momentum_y_rms"]) for r in dfg]
        ),
        "cylinder_velocity_rms_normalized": summary(
            [float(r["boundary/boundary/cylinder_velocity_rms_normalized"]) for r in dfg]
        ),
    },
    "heat": {
        "cohorts": sorted(set(r["cohort"] for r in heat)),
        "models": sorted(set(r["model_id"] for r in heat)),
        "rbf_combined_after_static_equivalence_audit": summary(
            [float(r["field_relative_l2"]) for r in heat if r["model_id"] == "rbf_fixed"]
        ),
        "rbf_combined_training_and_diagnostics_sec": summary(
            [float(r["training_and_diagnostics_sec"]) for r in heat if r["model_id"] == "rbf_fixed"]
        ),
    },
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "diagnostics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
for problem in ("burgers_sobol", "allen_cahn", "dfg"):
    print(problem)
    for key, value in result[problem].items():
        print(f"  {key}: n={value['n']} mean={value['mean']:.6g} sd={value['sample_sd']:.6g}")
for key in ("rbf_combined_after_static_equivalence_audit", "rbf_combined_training_and_diagnostics_sec"):
    value = result["heat"][key]
    print(f"heat {key}: n={value['n']} mean={value['mean']:.6g} sd={value['sample_sd']:.6g}")
