"""Run the DFG signal battery over existing checkpoints (301-305 and 501-505, pre/post L-BFGS). CPU."""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from src.research_pipeline.sciml import dfg, dfg_signals as ds  # noqa: E402
torch.set_num_threads(1)
KO = REPO / "var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs"
R501 = KO / "dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372"
R301 = KO / "dfg-rbf-current-cohort-refinement/dfg-current-cohort-refinement/run-1790606331368753261"
PRE501 = REPO / "var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch/dfg/dfg-re20-t4x2/dfg-re20-t4x2/campaign"
spec = importlib.util.spec_from_file_location("run501", R501 / "run_campaign_source.py")
run501 = importlib.util.module_from_spec(spec); spec.loader.exec_module(run501)
out = REPO / "var/agent_runs/exp0008/wp-e3"; (out / "dense").mkdir(parents=True, exist_ok=True)
pts, fluid = ds.dev_grid(220, 41)
sd = lambda p: (lambda c: c["state_dict"] if "state_dict" in c else c)(torch.load(p, map_location="cpu", weights_only=False))
rows = []


def run(tag, model):
    model.eval()
    d = ds.dense_features(model, pts); s = ds.scalar_signals(model, d, fluid)
    np.savez_compressed(out / "dense" / (tag.replace("/", "__") + ".npz"), **d, fluid=fluid)
    rows.append({"tag": tag, **s}); print(tag, round(s["dfg/qoi_drag_coefficient"], 3), round(s["dfg/flux_imbalance_rel"], 3), flush=True)


for s in range(301, 306):
    pre = next((REPO / "var/agent_runs/baseline-close-2026-09-27" / ("dfg-re20-gpu-r3" if s < 303 else "dfg-re20-gpu-r4") / "download-attempts").glob(f"attempt-*/*/runs/seed-{s}/dfg_rbf_pikan.pt"), None)
    for stage, p in (("pre", pre), ("post", R301 / "training" / f"seed-{s}" / "checkpoint-lbfgs.pt")):
        if p is not None and Path(p).exists():
            m = dfg.DFGRBFPIKAN(seed=s).float(); m.load_state_dict(sd(p)); run(f"301-305/rbf/{stage}/seed-{s}", m)
for arm, key in (("DFG-RBF-PIKAN", "rbf"), ("DFG-Tanh-MLP", "mlp")):
    for s in range(501, 506):
        for stage, p in (("pre", PRE501 / arm / f"seed-{s}" / "checkpoint-final.pt"), ("post", R501 / "training" / arm / f"seed-{s}" / "checkpoint-lbfgs.pt")):
            if p.exists():
                m = run501.build_model(arm, s).float(); m.load_state_dict(sd(p)); run(f"501-505/{key}/{stage}/seed-{s}", m)
(out / "dfg_signals.json").write_text(json.dumps(rows, indent=1))
