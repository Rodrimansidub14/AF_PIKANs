"""Humo CPU de la validacion: MLP del paper (semilla 20260728) con Adam reducido contra la historia de perdida publicada, y RBF16 con Adam reducido. Escribe smoke_baseline.json.
Comando completo (NO lanzado): ver budget_piezo.md."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_piezo as rp
torch.set_num_threads(6)
REF = HERE.parents[3] / "var/agent_runs/piezo-ref-20260930/daniel-ref-20261003/results/runs"
n_mlp, n_rbf = int(sys.argv[1]), int(sys.argv[2])
out = {}
for mode, name, f in (("converse", "indirect_mixed", "loss_indirect.npy"), ("direct", "direct_mixed", "loss_direct.npy")):
    prob = rp.make_problem(mode)
    ref = np.load(REF / f"paper20k_{name}" / f)
    m = rp.build(prob, "MLP", 20260728)
    tr = []
    A = rp.adam_piezo(prob, m, 20260728, n_mlp, torch.device("cpu"), torch.float64, None, eval_every=n_mlp, trace=tr)
    rel = [abs(a - b) / b for a, b in zip(tr, ref[:n_mlp])]
    out[f"{mode}_MLP"] = dict(steps=n_mlp, max_rel_diff_loss_vs_paper=max(rel), loss_last=tr[-1], paper_last=float(ref[n_mlp - 1]), score=A["hist"][-1])
    print(mode, "MLP", out[f"{mode}_MLP"], flush=True)
    r = rp.run_one(mode, "B", 20260728, n_rbf, "cpu", "float64", eval_every=n_rbf, quiet=True)
    out[f"{mode}_RBF16"] = dict(steps=n_rbf, score=r["score_final"], loss_end=r["loss_end"], seconds=r["total_seconds"])
    print(mode, "RBF16", out[f"{mode}_RBF16"], flush=True)
    (HERE / "smoke_baseline.json").write_text(json.dumps(out, indent=1))
