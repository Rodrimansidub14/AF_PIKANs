"""EXP-0016, etapa B, comprobacion POST-HOC (no preregistrada; se declara asi): ¿localiza mejor el error otra senal propia de la red, la estructura |grad u|?

Mismos checkpoints de Adam de la malla (5 por semilla), misma malla de la referencia y mismas metricas que `analyze_stage_b.py`
(Spearman y solape del 5 % superior con el error |u - u_ref|; azar del solape = 0.05). La senal |grad u| sale de la red en el checkpoint, sin referencia.
Se compara con el residuo |r| y con el mapa interno de asentamiento, ya calculados por la etapa B (stage_b_analysis.json). No decide nada del spec.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[2]))
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
DIR = HERE / "results/stage-a/stageA"
spec = importlib.util.spec_from_file_location("exp0016_sweep", str(HERE / "sweep.py"))
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)
pc = sw.load_campaign(BL / "burgers-close-20261003/pikan_campaign.py")
torch.set_num_threads(4)
PROBLEMS = {
    "burgers": dict(seeds=[301, 302, 303, 304, 305], ref=BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz", width=48, sub=(2, 4)),
    "ac": dict(seeds=[601, 602, 603], ref=BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat", width=16, sub=(1, 1)),
}
GRID_F = ["0.25", "0.4", "0.6", "0.8", "1.0"]


def top_overlap(a, b, frac=0.05):
    n = max(1, int(round(frac * a.size)))
    return len(set(np.argsort(a)[-n:].tolist()) & set(np.argsort(b)[-n:].tolist())) / n


prev = json.loads((HERE / "results/stage-a/stage_b_analysis.json").read_text())
out = {}
for problem, cfg in PROBLEMS.items():
    prob = pc.make_problem(problem, str(cfg["ref"]))
    st, sx = cfg["sub"]
    xx, tt = np.meshgrid(prob.x[::sx], prob.t[::st], indexing="xy")
    u_ref = prob.u[::st, ::sx]
    pts = torch.as_tensor(np.stack((xx.reshape(-1), tt.reshape(-1)), 1), dtype=torch.float32)
    rows = []
    for seed in cfg["seeds"]:
        res = json.loads((DIR / f"{problem}-seed{seed}.json").read_text())
        for f in GRID_F:
            k = res["grid"][f]
            ck = torch.load(DIR / f"{problem}-seed{seed}-adam-k{k}.pt", map_location="cpu", weights_only=False)["state_dict"]
            model = prob.build("rbf", cfg["width"], 11, seed).to(dtype=torch.float32)
            model.load_state_dict(ck)
            xt = pts.clone().requires_grad_(True)
            u = prob.transform(xt, model(xt))
            g = pc.grad(u, xt).detach()
            s = g.norm(dim=1).numpy()
            err = np.abs(u.detach().numpy().reshape(u_ref.shape) - u_ref).reshape(-1)
            rows.append(dict(seed=seed, f=f, sp_struct=float(spearmanr(s, err)[0]), ov_struct=top_overlap(s, err)))
    pb = {(r["seed"], r["f"]): r for r in prev[problem]["rows"]}
    for r in rows:
        r.update(sp_res=pb[(r["seed"], r["f"])]["sp_res"], ov_res=pb[(r["seed"], r["f"])]["ov_res"],
                 sp_map=pb[(r["seed"], r["f"])]["sp_map"], ov_map=pb[(r["seed"], r["f"])]["ov_map"])
    med = lambda key: float(np.median([r[key] for r in rows]))  # noqa: E731
    out[problem] = dict(rows=rows, **{key: med(key) for key in ("sp_struct", "ov_struct", "sp_res", "ov_res", "sp_map", "ov_map")})
    better = sum(np.median([r["sp_struct"] for r in rows if r["seed"] == s]) > np.median([r["sp_res"] for r in rows if r["seed"] == s])
                 and np.median([r["ov_struct"] for r in rows if r["seed"] == s]) > np.median([r["ov_res"] for r in rows if r["seed"] == s])
                 for s in cfg["seeds"])
    out[problem]["seeds_struct_better_than_residual"] = int(better)
    print(f"{problem}: mediana Spearman | estructura {out[problem]['sp_struct']:+.3f} | residuo {out[problem]['sp_res']:+.3f} | mapa {out[problem]['sp_map']:+.3f}")
    print(f"{problem}: mediana solape 5 % | estructura {out[problem]['ov_struct']:.3f} | residuo {out[problem]['ov_res']:.3f} | mapa {out[problem]['ov_map']:.3f}"
          f"   (estructura > residuo en {better}/{len(cfg['seeds'])} semillas)", flush=True)
(HERE / "results/stage-a/posthoc_structure.json").write_text(json.dumps(out, indent=1))
