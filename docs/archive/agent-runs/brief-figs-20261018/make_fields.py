"""Brief 2026-10-18: evalúa un checkpoint de control por problema y guarda campos (pred y referencia) para graficar.
Lee referencias solo para las figuras (diagnóstico); no toca evals/ ni ningún mecanismo."""
import csv, sys, json
from pathlib import Path
import numpy as np, torch
D = Path(__file__).resolve().parent
EXP13 = D.parents[2] / "experiments/diagnostics/EXP-0013-run-diagnosis"
sys.path.insert(0, str(EXP13))
from grids import get_grid
from adapters import load_bundle, _to_t
from labels import reference, model_components

SEL = {  # problema: (campaña, etapa, {brazo: ancho}, semilla)
 "heat_1d": ("heat-lbfgs-redo", "lbfgs_final", {"rbf": ""}, "901"),
 "burgers_1d": ("burgers-big-20261003", "lbfgs_final", {"rbf": "48", "mlp": ""}, "301"),
 "allen_cahn_1d": ("ac-rerun-20260930", "lbfgs_final", {"rbf": "16", "mlp": ""}, "601"),
 "ns2d_cg": ("ns2d-cg-20260930", "adam_final", {"rbf": "56", "mlp": ""}, "701"),
 "dfg_2d": ("cohort-B-lbfgs", "lbfgs_final", {"rbf": "", "mlp": ""}, "501"),
 "piezo_2d": ("piezo-daniel-recipe", "adam_final", {"rbf": "16", "mlp": ""}, "20260728"),
}
rows = list(csv.DictReader(open(EXP13 / "results/inventory.csv", encoding="utf-8")))
p = sys.argv[1]
camp, stage, arms, seed = SEL[p]
g = get_grid(p); pts = g["points"]
ref = reference(p, pts)["comps"]
out = {"points": pts, "shape": np.array(g["shape"] if g["shape"] else [0, 0])}
for k, v in ref.items(): out[f"ref_{k}"] = v
meta = {}
for arm, w in arms.items():
    r = [x for x in rows if x["problem"] == p and x["campaign"] == camp and x["stage"] == stage and x["arm"] == arm and x["width"] == w and x["seed"] == seed][0]
    b = load_bundle(r)
    with torch.no_grad():
        f = np.concatenate([b.fields(_to_t(pts[k:k + 8192], requires_grad=False)).numpy() for k in range(0, len(pts), 8192)])
        anchor = float(b.fields(torch.tensor([[2.2, 0.2]]))[0, 2]) if p == "dfg_2d" else None
    mc = model_components(p, f, anchor)
    if p == "dfg_2d":  # misma convención de presión que labels: se fija el gauge con la referencia
        pass
    for k, v in mc.items():
        out[f"{arm}_{k}"] = v
        meta[f"{arm}_{k}"] = float(np.linalg.norm(v - ref[k]) / np.linalg.norm(ref[k]))
    meta[f"{arm}_ckpt"] = r["path"]
np.savez_compressed(D / f"fields_{p}.npz", **out)
json.dump({"problem": p, "campaign": camp, "stage": stage, "seed": seed, "l2": meta}, open(D / f"fields_{p}.json", "w"), indent=1)
print(p, {k: round(v, 5) for k, v in meta.items() if not k.endswith("ckpt")})
