"""Post-hoc CPU float64 re-score of archived Heat S2 RBF snapshots (no training).

Purpose: attribute the S2 score 2.06e-6 to a training stage. The archived score is for
checkpoint-final.pt (post partial L-BFGS). This scores the Adam-only snapshots of the SAME
runs (checkpoint-adam-final.pt = 20k Adam updates; checkpoint-500s.pt = 500 s snapshot) with the
same analytic field_relative_l2 formula as heat_common_eval.score_torch_model on the
512x256 grid, CPU float64. checkpoint-final.pt is re-scored as a sanity check against the
archived CUDA value. Post hoc, descriptive; no selection, no reference or eval edits.
Run with: py -3.13 rescore_heat_stages_cpu.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT / "kaggle-outputs" / "closure-s2-20260928" / "heat-stepmatched-s2"
sys.path.insert(0, str(S2 / "payload"))
from af_sciml_model import build_model  # noqa: E402
from heat_common_eval import _trap_weights  # noqa: E402

ALPHA = 0.1
NX, NT = 512, 256
torch.set_num_threads(4)


def field_rel_l2(model):
    x = torch.linspace(0.0, 1.0, NX, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, NT, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    coords = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
    with torch.no_grad():
        pred = model(coords).reshape(NX, NT)
    exact = torch.sin(torch.pi * xx) * torch.exp(-ALPHA * torch.pi ** 2 * tt)
    w = _trap_weights(NX, torch.device("cpu"))[:, None] * _trap_weights(NT, torch.device("cpu"))[None, :]
    return float(torch.sqrt(torch.sum(w * (pred - exact) ** 2) / torch.sum(w * exact ** 2)))


out = []
for seed in range(901, 906):
    rdir = S2 / "runs" / "tier2" / f"seed-{seed}" / "rbf"
    archived = json.load(open(S2 / "confirmation" / f"seed-{seed}" / "rbf" / "confirmation-metrics.json"))[
        "heat/field_relative_l2"]
    row = {"seed": seed, "archived_final_cuda": archived}
    for name in ("checkpoint-final", "checkpoint-adam-final", "checkpoint-500s"):
        ck = torch.load(rdir / f"{name}.pt", map_location="cpu")
        model = build_model("heat", {"id": "rbf_fixed"})
        model.load_state_dict(ck["state_dict"])
        model = model.to(dtype=torch.float64).eval()
        row[name] = field_rel_l2(model)
        row[name + "_meta"] = {k: ck.get(k) for k in ("adam_updates", "lbfgs_updates", "stage", "elapsed_seconds")
                               if k in ck}
    row["final_cpu_vs_cuda_rel_diff"] = abs(row["checkpoint-final"] - archived) / archived
    out.append(row)
    print(json.dumps({k: v for k, v in row.items() if not k.endswith("_meta")}))

json.dump(out, open(Path(__file__).parent / "heat-s2-stage-rescore-cpu.json", "w"), indent=2)
