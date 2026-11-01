"""Tier 1 (CPU, sin referencia): donde dispara la senal de estancamiento sobre las historias de perdida YA registradas.

Lee SOLO perdida y reloj de los history de los baselines. No lee errores contra la referencia ni puntua nada: no es la etapa B.
Sirve para saber, antes de gastar GPU, si la senal dispara dentro de la ventana permitida o queda inerte (k_t = k_max).
Uso: python preview_trigger.py  (desde cualquier carpeta; AF_ROOT opcional)
"""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
from src.mechanisms.stall_handoff import handoff_step  # noqa: E402

BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"


def burgers(seed):
    d = BL / f"burgers-close-20261003/outputs/burgers/results/burgers/rbf48-seed-{seed}"
    h = json.loads((d / "history.json").read_text())
    return [(e["step"], e["loss"], e["seconds"]) for e in h], 20000


def ac(seed):
    d = BL / f"ns2d-cg-20260930/outputs/ac/ac/results/ac/rbf16-seed-{seed}"
    h = json.loads((d / "history.json").read_text())
    return [(e["step"], e["loss"], e["seconds"]) for e in h], 40000


def heat(seed):
    f = BL / f"kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2/runs/tier2/seed-{seed}/rbf/history.csv"
    rows = [r for r in csv.DictReader(open(f)) if r["stage"] == "adam"]
    return [(int(r["step"]), float(r["loss"]), float(r["elapsed_gpu_seconds_estimate"])) for r in rows], 20000


CASES = [("burgers", burgers, [301, 302, 303, 304, 305]), ("allen_cahn", ac, [601, 602, 603, 604, 605]),
         ("heat", heat, [901, 902, 903, 904, 905])]
out = []
for name, loader, seeds in CASES:
    for s in seeds:
        try:
            h, k_max = loader(s)
        except FileNotFoundError as e:
            print(name, s, "sin historia:", e)
            continue
        steps = [x[0] for x in h]
        loss = [x[1] for x in h]
        sec = dict((x[0], x[2]) for x in h)
        res = {}
        for rho in (0.05, 0.10, 0.20):
            r = handoff_step(steps, loss, k_max, rho_star=rho)
            res[rho] = r
        r = res[0.10]
        t_adam_total = sec.get(k_max)
        t_at = sec.get(r["k_t"])
        row = dict(problem=name, seed=s, k_max=k_max, k_t=r["k_t"], fired=r["fired"], reason=r["reason"],
                   frac=round(r["k_t"] / k_max, 3), r_ref=r["r_ref"],
                   k_t_rho005=res[0.05]["k_t"], k_t_rho020=res[0.20]["k_t"],
                   adam_seconds_saved=(round(t_adam_total - t_at) if t_adam_total and t_at else None))
        out.append(row)
        print(row, flush=True)
(Path(__file__).parent / "preview_trigger.json").write_text(json.dumps(out, indent=1))
