"""Comprobación de H2 sin entrenar: ¿cuánto cambia la solución del directo si el voltaje flotante V_f (escalar) se sustituye por el mínimo exacto de la pérdida?
La pérdida es cuadrática en V_f (ec. 75 del paper), así que se ajusta una parábola a 3 evaluaciones y se verifica con un barrido.
Carga los estados finales guardados (artefactos) de B, E, J del directo (semillas 20260728-30), en CPU float64."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piezo"))
import run_piezo as rp  # noqa: E402

RES = HERE.parent / "results"
DIRS = ("piezo-s1a", "piezo-s1b", "piezo-modal-run")
SEEDS = (20260728, 20260729, 20260730)
torch.set_num_threads(2)
dev, dt = torch.device("cpu"), torch.float64
prob = rp.make_problem("direct", 0)


def art(arm, seed):
    for dd in DIRS:
        for f in glob.glob(str(RES / dd / "**" / f"piezo-direct-{arm}-seed{seed}-artifacts.npz"), recursive=True):
            return np.load(f, allow_pickle=True)


def load_model(arm, seed):
    m = rp.build(prob, arm if arm in ("B",) else "B", seed, dt).to(dev)
    z = art(arm, seed)
    if arm != "B":
        pre = rp.gp.GramPreconditioner(arm, every=10 ** 9, n_points=64, chunk=2)
        pre.attach(m)
    sd = {k.split("/state/")[1]: torch.tensor(z[k]) for k in z.keys() if k.startswith("s020000/state/")}
    missing = m.load_state_dict(sd, strict=False)
    assert not missing.missing_keys and not missing.unexpected_keys, missing
    return m


rows = []
out = ["# Comprobación de H2 (voltaje flotante) sobre los estados finales guardados, directo, CPU\n",
       "| brazo | semilla | V̂_f aprendido | V̂_f exacto | phi_top medio aprendido → exacto (FEM −3.103 V) | l2re_phi | l2re_u | l2re_v | pérdida |\n|---|---|---|---|---|---|---|---|---|"]
for arm in ("B", "E", "J"):
    for seed in SEEDS:
        m = load_model(arm, seed)
        P = prob.pools(seed, dev, dt, 99)
        v0 = float(m.floating_voltage.detach())

        def ev(v):
            with torch.no_grad():
                m.floating_voltage.copy_(torch.tensor([v], dtype=dt))
            loss, _ = prob.loss(m, P)
            return float(loss.detach())

        pts = [v0 - 0.1, v0, v0 + 0.1]
        ls = [ev(v) for v in pts]
        a2 = (ls[0] - 2 * ls[1] + ls[2]) / (0.1 ** 2)
        a1 = (ls[2] - ls[0]) / 0.2
        vstar = v0 - a1 / a2 if a2 > 0 else v0
        res = {}
        for tag, v in (("aprendido", v0), ("exacto", vstar)):
            ev(v)
            sc = prob.score(m, dev, dt)
            res[tag] = dict(v=v, loss=ev(v), phi_top=sc["phi_top_mean"], l2phi=sc["l2re_phi"], l2u=sc["l2re_u"], l2v=sc["l2re_v"])
        a, b = res["aprendido"], res["exacto"]
        rows.append(dict(arm=arm, seed=seed, **{f"{k}_{t}": res[t][k] for t in res for k in res[t]}))
        out.append(f"| {arm} | {seed % 100} | {a['v']:.3f} | {b['v']:.3f} | {a['phi_top']:.2f} → {b['phi_top']:.2f} | {a['l2phi']:.3f} → {b['l2phi']:.3f} | {a['l2u']:.3f} → {b['l2u']:.3f} | {a['l2v']:.3f} → {b['l2v']:.3f} | {a['loss']:.4g} → {b['loss']:.4g} |")
        print(out[-1], flush=True)
Path(HERE / "check_Vf.md").write_text("\n".join(out), encoding="utf-8")
json.dump(rows, open(HERE / "check_Vf.json", "w"), indent=1)
