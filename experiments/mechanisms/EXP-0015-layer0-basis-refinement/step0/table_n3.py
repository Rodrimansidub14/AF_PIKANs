"""Tabla n=3 del paso 0: razon brazo/C por semilla, media geometrica y criterio del spec.
Uso: python table_n3.py [results_dir]  (semillas 1-3; ajusta por problema los brazos presentes)."""
import json, math, sys
from pathlib import Path
R = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
PROBS = ["dfg", "burgers", "heat", "ns2d_cg", "allen_cahn", "piezo"]
ARMS = ["O11", "U41", "X11"]
def l2(p, a, s):
    f = R / f"{p}-{a}-seed{s}.json"
    return json.loads(f.read_text())["final"]["u"] if f.exists() else None
print("| Problema | n | C (L2, media geom.) | PIKAN-PDE | PIKAN-PDE/C | " + " | ".join(f"{a}/C por semilla [GM] sost." for a in ARMS) + " |")
print("|---|---|---|---|---|" + "---|" * len(ARMS))
for p in PROBS:
    seeds = [s for s in (1, 2, 3) if l2(p, "C", s) is not None]
    if len(seeds) < 3: continue
    c = [l2(p, "C", s) for s in seeds]
    gm_c = math.exp(sum(map(math.log, c)) / len(c))
    base = json.loads((R / f"{p}-C-seed{seeds[0]}.json").read_text())["baseline_l2"]
    cells = []
    for a in ARMS:
        v = [l2(p, a, s) for s in seeds]
        if any(x is None for x in v): cells.append("-"); continue
        r = [x / y for x, y in zip(v, c)]
        gm = math.exp(sum(map(math.log, r)) / len(r))
        ok = gm <= 0.80 and sum(x <= 0.80 for x in r) >= 2
        cells.append(" / ".join(f"{x:.2f}" for x in r) + f" [{gm:.2f}] " + ("SI" if ok else "no"))
    print(f"| {p} | {len(seeds)} | {gm_c:.2e} | {base:.2e} | {base/gm_c:.2f}x | " + " | ".join(cells) + " |")
