"""py3.13 step: evaluate Daniel's committed paper weights against his own FEM grid (fem-grid-*.npz). CPU, no training."""
import os, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
for k in ("DATA", "MODELS", "OUTPUTS"): os.environ[f"PINN_PIEZO_{k}_DIR"] = str(HERE / "_scratch" / k.lower())
sys.path.insert(0, str(HERE / "daniel-ref" / "src"))
import numpy as np, torch
from pinn_piezo.indirect import model as mi
from pinn_piezo.direct import model as md
out = {}
for case, mod in (("indirect", mi), ("direct", md)):
    z0 = np.load(HERE / f"fem-grid-{case}.npz"); z1 = np.load(HERE / f"fem-grid-{case}-polingminus.npz")
    net = mod.build_default_model(device="cpu")
    sd = torch.load(HERE / "daniel-ref" / "models" / case / f"model_PINN_{case}_paper_3.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd)
    net = net.double().eval()
    xy = torch.tensor(z0["xy"], dtype=torch.float64)
    with torch.no_grad(): p = net(xy).numpy()
    out[case] = {}
    for tag, z in (("poling+1", z0), ("poling-1", z1)):
        row = {}
        for i, name in enumerate(("u", "v", "phi")):
            ref, pr = z[name], p[:, i]
            row[name] = {"rel_L2": float(np.linalg.norm(pr - ref) / np.linalg.norm(ref)), "ref_absmax": float(np.abs(ref).max()), "pred_absmax": float(np.abs(pr).max())}
        out[case][tag] = row
        print(case, tag, {k: round(v["rel_L2"], 4) for k, v in row.items()}, {k: (v["ref_absmax"], v["pred_absmax"]) for k, v in row.items()})
json.dump(out, open(HERE / "paper-weights-vs-fem.json", "w"), indent=2)
