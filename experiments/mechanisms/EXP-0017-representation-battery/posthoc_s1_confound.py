"""EXP-0017, comprobacion POST-HOC (no preregistrada): ¿la ventaja de S1 sobre el residuo es un artefacto de la restriccion dura?

En Burgers y Allen-Cahn la salida es u = g(x, t) + (1 - x^2) t raw(x, t) con la condicion inicial y de frontera impuestas por construccion: el error es EXACTAMENTE
cero donde el factor w(x, t) = (1 - x^2) t se anula (t = 0, |x| = 1). S1 = ‖du/dtheta‖ = w ‖d raw / dtheta‖ hereda ese factor. Se separa en:
  F      = el factor w solo (geometria del dominio, sin red),
  S1raw  = ‖d raw / d theta‖ = S1 / w  (lo que aporta la red),
y se compara cada una con el error en los mismos 40 estados y las mismas metricas. No decide nada del spec.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_exp0017", str(HERE / "run_exp0017.py"))
rx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rx)
torch.set_num_threads(4)

out = {}
for problem, cfg in rx.PROBLEMS.items():
    prob = rx.pc.make_problem(problem, str(cfg["ref"]))
    sy, sx = cfg["sub"]
    xx, tt = np.meshgrid(prob.x[::sx], prob.t[::sy], indexing="xy")
    u_ref = prob.u[::sy, ::sx]
    pts = torch.as_tensor(np.stack((xx.reshape(-1), tt.reshape(-1)), 1), dtype=torch.float32)
    idx = np.sort(np.random.default_rng(rx.SUB_SEED).choice(pts.shape[0], size=min(rx.N_SUB, pts.shape[0]), replace=False))
    xs = pts[idx]
    w = ((1 - xs[:, 0] ** 2) * xs[:, 1]).numpy()
    rows = []
    for seed in cfg["seeds"]:
        for f in rx.GRID_F:
            res = json.loads((rx.DIR / f"{problem}-seed{seed}.json").read_text())
            k = res["grid"][f]
            m = rx.build(prob, cfg, seed, rx.DIR / f"{problem}-seed{seed}-adam-k{k}.pt")
            with torch.no_grad():
                err = np.abs(prob.transform(pts, m(pts)).numpy().reshape(u_ref.shape) - u_ref).reshape(-1)[idx]
            s1 = rx.sig.s1_jacobian_norm(m, xs, prob.transform, chunk=256).numpy()
            inside = w > 1e-6
            raw = np.where(inside, s1 / np.where(inside, w, 1.0), 0.0)
            rows.append(dict(seed=seed, f=f,
                             F=dict(sp=rx.st.spearman(w, err), ov=rx.st.top_overlap(w, err)),
                             S1=dict(sp=rx.st.spearman(s1, err), ov=rx.st.top_overlap(s1, err)),
                             S1raw=dict(sp=rx.st.spearman(raw[inside], err[inside]), ov=rx.st.top_overlap(raw[inside], err[inside]))))
    med = lambda name, m_: float(np.median([r[name][m_] for r in rows]))  # noqa: E731
    out[problem] = {n: dict(sp=med(n, "sp"), ov=med(n, "ov")) for n in ("F", "S1", "S1raw")}
    print(problem, "(mediana Spearman / solape 5 %):", {n: (round(v["sp"], 3), round(v["ov"], 3)) for n, v in out[problem].items()}, flush=True)
(HERE / "results/posthoc_s1_confound.json").write_text(json.dumps(out, indent=1))
