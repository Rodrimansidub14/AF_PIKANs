"""EXP-0023 ronda 1-B, nivel T1: razon del error tras L-BFGS de cada brazo frente a B (misma semilla, misma sesion), por problema.
Criterio firmado (regla de salida del EXP-0024 + Enmienda 1): un brazo pasa T1 si la MEDIANA de la razon <= 0.90 en Burgers Y en Allen-Cahn.
Tambien reporta: fin de Adam, mejora por semilla, guardia de reloj (total del brazo <= 1.2 x el de B, mediana), parametros finitos (g3) y ausencias (no se imputan como 0).
Uso: python analyze_block.py [carpeta_con_los_json] [--v]  ->  escribe t1_block_analysis.json junto a la carpeta
"""
import glob, json, sys
from pathlib import Path
import numpy as np

D = Path(next((x for x in sys.argv[1:] if not x.startswith("--")), None) or Path(__file__).resolve().parent / "results/t1-block")
rows = [json.load(open(f)) for f in glob.glob(str(D / "*-seed*-a*-l*.json"))]
ARMS = tuple(x for x in ("J", "E", "I", "F") if any(r["arm"] == x for r in rows))
V_MODE = "--v" in sys.argv   # etapa V: ademas exige mejora en >= 4 de 5 semillas en ambos problemas
out, passed = {}, {a: True for a in ARMS}
for prob in ("burgers", "ac"):
    get = lambda arm, seed, k: next((r.get(k) for r in rows if r["problem"] == prob and r["arm"] == arm and r["seed"] == seed), None)
    seeds = sorted({r["seed"] for r in rows if r["problem"] == prob})
    for arm in ARMS:
        pairs = [(s, get(arm, s, "primary_final"), get("B", s, "primary_final"), get(arm, s, "primary_adam"), get("B", s, "primary_adam"),
                  get(arm, s, "total_seconds"), get("B", s, "total_seconds"), get(arm, s, "finite_params")) for s in seeds]
        ok = [p for p in pairs if p[1] is not None and p[2] is not None]
        missing = [p[0] for p in pairs if p[1] is None or p[2] is None]
        if not ok:
            out[f"{prob}/{arm}"] = dict(n=0, missing=missing)
            passed[arm] = False
            continue
        rf = np.array([p[1] / p[2] for p in ok])
        ra = np.array([p[3] / p[4] for p in ok if p[3] is not None and p[4] is not None])
        clock = np.array([p[5] / p[6] for p in ok if p[5] and p[6]])
        out[f"{prob}/{arm}"] = dict(n=len(ok), missing=missing, median_ratio_final=float(np.median(rf)), improved_final=int((rf < 1).sum()), ratios_final=[round(float(x), 3) for x in rf],
                                    median_ratio_end_of_adam=float(np.median(ra)) if len(ra) else None, median_clock_ratio=float(np.median(clock)) if len(clock) else None,
                                    all_finite=bool(all(p[7] for p in ok)))
        if missing or len(ok) < 5 or np.median(rf) > 0.90 or (V_MODE and int((rf < 1).sum()) < 4):
            passed[arm] = False
        print(f"{prob:8s} {arm}  final: mediana {np.median(rf):.3f}  mejora {int((rf < 1).sum())}/{len(ok)}  {np.round(rf, 2).tolist()}  reloj x{np.median(clock):.2f}" if len(clock) else f"{prob} {arm} sin reloj")
out["passes_V_median_le_0.90_and_4of5" if V_MODE else "passes_T1_median_le_0.90_both_problems"] = passed
print("pasan T1:", passed)
(D.parent / "t1_block_analysis.json").write_text(json.dumps(out, indent=1))
