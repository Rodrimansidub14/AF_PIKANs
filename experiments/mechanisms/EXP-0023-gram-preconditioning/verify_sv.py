"""Enmienda 1, verificacion previa a la 7b: rango de valores singulares de P normalizado (J, E, I, F) en el estado inicial y en el de 30 pasos de Adam (float64, N = 512)."""
import json, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pre_trainer as pt
torch.set_num_threads(2)
st = HERE / "results/shorttest-gpu/exp0023_shorttest"
out = {}
for problem in ("ac", "burgers"):
    cfg = pt.PROBLEMS[problem]
    saved = torch.load(st / f"state-{problem}.pt", map_location="cpu")
    for when in ("init", "adam30"):
        for arm in ("J", "E", "I", "F"):
            prob = pt.pc.make_problem(cfg["name"], str(cfg["ref"]))
            m = prob.build("rbf", cfg["width"], 11, 1).to(dtype=torch.float64)
            if when == "adam30":
                m.load_state_dict(saved)
            P = prob.pools(1, torch.device("cpu"), torch.float64)
            pre = pt.gp.GramPreconditioner(arm, n_points=512)
            pre.attach(m)
            if arm != "F":
                pre.refresh(m, prob, P["residual"])
            rng = []
            for l in m.layers:
                sv = torch.linalg.svdvals(l.pre_P.double())
                rng.append((float(sv.min()), float(sv.max())))
            out[f"{problem}-{when}-{arm}"] = rng
            print(problem, when, arm, [(f"{a:.3g}", f"{b:.3g}") for a, b in rng], flush=True)
(HERE / "results/verify_sv.json").write_text(json.dumps(out, indent=1))
