"""EXP-0024 T0: razon del error final de cada diseño frente al control (misma semilla, mismo presupuesto). Uso: python analyze_t0.py"""
import glob, json, numpy as np
from pathlib import Path
D = Path(__file__).resolve().parent / "results/t0"
rows = [json.load(open(f)) for f in glob.glob(str(D / "*-a600-l250-n3000.json"))]
get = lambda p, d, s, k: next(r[k]["u"] for r in rows if r["problem"] == p and r["design"] == d and r["seed"] == s)
out = {}
for p in ("allen_cahn", "heat"):
    for d in ("D1", "D2", "D3", "D4"):
        seeds = [s for s in range(1, 6) if any(r["problem"] == p and r["design"] == d and r["seed"] == s for r in rows)]
        if not seeds:
            continue
        rf = np.array([get(p, d, s, "final") / get(p, "CTRL", s, "final") for s in seeds])
        ra = np.array([get(p, d, s, "adam") / get(p, "CTRL", s, "adam") for s in seeds])
        out[f"{p}/{d}"] = dict(n=len(seeds), median_ratio_final=float(np.median(rf)), improved=int((rf < 1).sum()), ratios_final=[round(float(x), 2) for x in rf],
                               median_ratio_end_of_adam=float(np.median(ra)))
        print(f"{p:10s} {d}  final: mediana {np.median(rf):.2f}  mejora {int((rf<1).sum())}/{len(seeds)}  {np.round(rf,2).tolist()}   fin de Adam: mediana {np.median(ra):.2f}")
(D.parent / "t0_summary.json").write_text(json.dumps(out, indent=1))
