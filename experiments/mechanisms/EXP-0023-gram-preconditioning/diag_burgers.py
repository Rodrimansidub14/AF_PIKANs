"""Diagnostico post-hoc (sin computo nuevo) de la razon por semilla de E y F en Burgers, T1 (semillas 1-5) + V (101-105). EXPLORATORIO, no preregistrado.
Predictores disponibles en los logs: error de B, razon al final de Adam, razon de la perdida de entrenamiento en el paso 2000, condicionamiento de P (E) y densidad de puntos de
colocacion en el frente (del pool de la semilla; el pool es determinista). NO hay checkpoints, asi que no hay Gram de las activaciones entrenadas ni error por region.
Con n = 10 por brazo, los p-valores son descriptivos (sin correccion por multiplicidad)."""
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
rows = [json.load(open(f)) for d in ("t1-block/exp0023_t1block", "v-block/exp0023_vblock") for f in glob.glob(str(HERE / "results" / d / "burgers-*-seed*.json"))]
import pre_trainer as pt
cfg = pt.PROBLEMS["burgers"]
prob = pt.pc.make_problem("burgers", str(cfg["ref"]))
nu = prob.nu
get = lambda arm, s: next(r for r in rows if r["arm"] == arm and r["seed"] == s)
seeds = sorted({r["seed"] for r in rows})
feat = {}
for s in seeds:
    X = prob.pools(s, "cpu", pt.torch.float64)["residual"].numpy()
    x, t = X[:, 0], X[:, 1]
    feat[s] = dict(front_n=int(((abs(x) <= 5 * nu) & (t >= 1 / np.pi)).sum()), near_n=int(((abs(x) <= 0.1) & (t >= 0.3)).sum()),
                   B_final=get("B", s)["primary_final"], B_adam=get("B", s)["primary_adam"], B_loss2000=get("B", s)["hist"][-1]["loss"])
out = {"seeds": seeds, "features": feat, "arms": {}}
for arm in ("E", "F"):
    y = np.array([np.log(get(arm, s)["primary_final"] / feat[s]["B_final"]) for s in seeds])
    cols = {
        "log B_final (denominador)": np.log([feat[s]["B_final"] for s in seeds]),
        "log arm_final": np.log([get(arm, s)["primary_final"] for s in seeds]),
        "log razon fin de Adam": np.log([get(arm, s)["primary_adam"] / feat[s]["B_adam"] for s in seeds]),
        "log razon perdida paso 2000": np.log([get(arm, s)["hist"][-1]["loss"] / feat[s]["B_loss2000"] for s in seeds]),
        "puntos en el frente (|x|<=5nu, t>=1/pi)": [feat[s]["front_n"] for s in seeds],
        "puntos cerca del frente (|x|<=0.1, t>=0.3)": [feat[s]["near_n"] for s in seeds],
    }
    if arm == "E":
        cols["cond(P) capa 1 en paso 0"] = [get(arm, s)["pre_diag"][0]["p_cond"][1] for s in seeds]
        cols["cond(P) capa 1 en L-BFGS"] = [get(arm, s)["pre_diag"][-1]["p_cond"][1] for s in seeds]
        cols["cond(P) capa 0 en paso 0"] = [get(arm, s)["pre_diag"][0]["p_cond"][0] for s in seeds]
    res = {}
    print(f"\n== {arm}: log(razon final) por semilla ==")
    for s, v in zip(seeds, np.exp(y)):
        print(f"  semilla {s:3d}: razon {v:7.2f}  B_final {feat[s]['B_final']:.4f}  frente {feat[s]['front_n']}  cerca {feat[s]['near_n']}")
    for k, v in cols.items():
        rho, p = spearmanr(v, y)
        res[k] = dict(spearman=float(rho), p=float(p))
        print(f"  {k:45s} rho {rho:+.2f}  p {p:.3f}")
    out["arms"][arm] = dict(ratio_final=dict(zip(map(str, seeds), map(float, np.exp(y)))), correlations=res)
(HERE / "results/diag_burgers.json").write_text(json.dumps(out, indent=1))
