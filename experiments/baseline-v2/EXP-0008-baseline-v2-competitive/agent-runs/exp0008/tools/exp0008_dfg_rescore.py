"""WP-E3: rescore DFG cohorts 301-305 and 501-505 with ONE sigma-n force scorer (CPU, float32).

Also reproduces the archived (tangential) 501-505 numbers with the campaign's own scorer to prove
which scorer produced them. Reads checkpoints only; no training.
"""
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from src.research_pipeline.sciml import dfg  # noqa: E402

KO = REPO / "var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs"
R501 = KO / "dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372"
R301 = KO / "dfg-rbf-current-cohort-refinement/dfg-current-cohort-refinement/run-1790606331368753261"
PRE501 = REPO / "var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch/dfg/dfg-re20-t4x2/dfg-re20-t4x2/campaign"
PRE301 = {301: "attempt-03/pikan-682b323105591987", 302: "attempt-03/pikan-682b323105591987",
          303: "attempt-01/pikan-c349281ede3181b8", 304: "attempt-01/pikan-c349281ede3181b8",
          305: "attempt-01/pikan-c349281ede3181b8"}
PRE301_ROOT = {301: "dfg-re20-gpu-r3", 302: "dfg-re20-gpu-r3", 303: "dfg-re20-gpu-r4", 304: "dfg-re20-gpu-r4", 305: "dfg-re20-gpu-r4"}
FEATFLOW = {"drag_coefficient": 5.57953523384, "lift_coefficient": 0.010618948146, "pressure_difference": 0.11752016697}

spec = importlib.util.spec_from_file_location("run501", R501 / "run_campaign_source.py")
run501 = importlib.util.module_from_spec(spec); spec.loader.exec_module(run501)
torch.set_num_threads(2)


def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["state_dict"] if "state_dict" in ck else ck


def sigma_n_qoi(model, surface_count=512):
    """Same contour and quadrature as dfg.evaluate_dfg_qois, generic over model classes."""
    theta = 2 * math.pi * np.arange(surface_count) / surface_count
    surf = np.column_stack((0.2 + 0.05 * np.cos(theta), 0.2 + 0.05 * np.sin(theta)))
    xy = torch.as_tensor(surf, dtype=torch.float32).requires_grad_(True)
    f = model(xy)
    g = lambda c: torch.autograd.grad(f[:, c].sum(), xy, create_graph=False, retain_graph=True)[0].detach().numpy()
    gu, gv = g(0), g(1)
    p = f[:, 2].detach().numpy()
    pf = float(model(torch.tensor([[0.15, 0.2]]))[0, 2]); pr = float(model(torch.tensor([[0.25, 0.2]]))[0, 2])
    q = dfg.qoi_from_cylinder(theta, p, gu[:, 0], gu[:, 1], gv[:, 0], gv[:, 1], pf, pr)
    # decomposition
    nx, ny = np.cos(theta), np.sin(theta); ds = 2 * math.pi * 0.05 / surface_count
    cs = 2.0 / (0.2 ** 2 * 0.1)
    q["cd_pressure"] = float(cs * np.sum(-p * nx) * ds)
    q["cd_viscous"] = float(cs * np.sum(1e-3 * (gu[:, 0] * nx + gu[:, 1] * ny) * 1.0) * ds)
    return q


def old_tangential(model):
    q = run501.qoi_for_model(model, torch.device("cpu"))
    return {k: q[k] for k in ("drag_coefficient", "lift_coefficient", "pressure_difference")}


def rel(q):
    return {"cd_rel_err": abs(q["drag_coefficient"] - FEATFLOW["drag_coefficient"]) / FEATFLOW["drag_coefficient"],
            "cl_abs_err": abs(q["lift_coefficient"] - FEATFLOW["lift_coefficient"]),
            "dp_rel_err": abs(q["pressure_difference"] - FEATFLOW["pressure_difference"]) / FEATFLOW["pressure_difference"]}


rows = []
# ---- cohort 501-505 (campaign classes) ----
for arm in ("DFG-RBF-PIKAN", "DFG-Tanh-MLP"):
    for s in range(501, 506):
        for stage, path in (("pre_lbfgs", PRE501 / arm / f"seed-{s}" / "checkpoint-final.pt"),
                            ("post_lbfgs", R501 / "training" / arm / f"seed-{s}" / "checkpoint-lbfgs.pt")):
            if not path.exists():
                rows.append({"cohort": "501-505", "arm": arm, "seed": s, "stage": stage, "status": "missing"}); continue
            m = run501.build_model(arm, s).float(); m.load_state_dict(load_sd(path)); m.eval()
            new = sigma_n_qoi(m); old = old_tangential(m)
            rows.append({"cohort": "501-505", "arm": arm, "seed": s, "stage": stage, "model_class": "run501",
                         "sigma_n": {k: new[k] for k in ("drag_coefficient", "lift_coefficient", "pressure_difference", "cd_pressure", "cd_viscous")},
                         "tangential_archived_formula": old, "sigma_n_err": rel(new), "tangential_err": rel(old)})
            print(arm, s, stage, "sigma_n Cd=%.4f dp=%.4f | tangential Cd=%.4f dp=%.4f" % (new["drag_coefficient"], new["pressure_difference"], old["drag_coefficient"], old["pressure_difference"]), flush=True)
# ---- cohort 301-305 (dfg.py classes) ----
for s in range(301, 306):
    for stage, path in (("pre_lbfgs", next((REPO / "var/agent_runs/baseline-close-2026-09-27" / PRE301_ROOT[s] / "download-attempts").glob(f"attempt-*/*/runs/seed-{s}/dfg_rbf_pikan.pt"), None)),
                        ("post_lbfgs", R301 / "training" / f"seed-{s}" / "checkpoint-lbfgs.pt")):
        if path is None or not Path(path).exists():
            rows.append({"cohort": "301-305", "arm": "DFG-RBF-PIKAN", "seed": s, "stage": stage, "status": "missing"}); continue
        m = dfg.DFGRBFPIKAN(seed=s).float(); m.load_state_dict(load_sd(path)); m.eval()
        new = sigma_n_qoi(m)
        rows.append({"cohort": "301-305", "arm": "DFG-RBF-PIKAN", "seed": s, "stage": stage, "model_class": "dfg.py",
                     "sigma_n": {k: new[k] for k in ("drag_coefficient", "lift_coefficient", "pressure_difference", "cd_pressure", "cd_viscous")},
                     "sigma_n_err": rel(new)})
        print("301-305", s, stage, "sigma_n Cd=%.4f dp=%.4f" % (new["drag_coefficient"], new["pressure_difference"]), flush=True)

out = REPO / "var/agent_runs/exp0008/wp-e3"; out.mkdir(parents=True, exist_ok=True)
(out / "dfg_rescore_sigma_n.json").write_text(json.dumps(rows, indent=1))
