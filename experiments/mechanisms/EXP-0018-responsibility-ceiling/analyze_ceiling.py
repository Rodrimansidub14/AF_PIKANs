"""EXP-0018: criterios del spec v2 firmado sobre los JSON del bloque (ramas B, P, H, N, O; N-H no se corrio).

Razon pareada por semilla: r = error(brazo) / error(B), error = L2 relativo global tras L-BFGS desde el mismo estado final de Adam.
H-CEIL-1 (para P o H; en Burgers Y Allen-Cahn): mediana de r <= 0.85 y r < 1 en >= 4 de 5 semillas. Guardias: g1 error en la region critica (razon, media geometrica) <= 1.10;
g2 reloj de L-BFGS (media geometrica de brazo/B) <= 1.5; g3 parametros finitos.
H-CEIL-2 (para el brazo que paso): P frente a N (H frente a N-H, no corrido): mediana de la razon <= 0.90 y mejora en >= 4 de 5, en ambos problemas.
Los p (permutacion con signo por semilla) se reportan; no deciden (con n = 5 el minimo ajustado por Holm es 0.125).
Uso: python analyze_ceiling.py [carpeta con los JSON]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results/ceiling/ceiling"
SEEDS = {"burgers": [301, 302, 303, 304, 305], "ac": [601, 602, 603, 604, 605]}
ARMS = ["P", "H", "N", "O"]


def gm(x):
    return float(math.exp(np.mean(np.log(x))))


def exact_flip_p(d):
    """p unilateral exacto, signo de cada diferencia (log razon) invertido: H1 = la mediana es negativa."""
    d = np.asarray(d)
    n = len(d)
    obs = np.median(d)
    cnt = 0
    for mask in range(2 ** n):
        s = np.array([-1.0 if (mask >> i) & 1 else 1.0 for i in range(n)])
        cnt += np.median(s * d) <= obs + 1e-15
    return cnt / 2 ** n


def main():
    data = {p: [json.loads((DIR / f"{p}-seed{s}.json").read_text()) for s in SEEDS[p]] for p in SEEDS}
    out = {}
    print("Control B frente a la etapa A del EXP-0016 (k = 1.00), error global:")
    for p, rows in data.items():
        a = {s: json.loads((HERE.parent / f"EXP-0016-stall-handoff/results/stage-a/stageA/{p}-seed{s}.json").read_text()) for s in SEEDS[p] if (HERE.parent / f"EXP-0016-stall-handoff/results/stage-a/stageA/{p}-seed{s}.json").exists()}
        for r in rows:
            if r["seed"] in a:
                ref = next(iter(a[r["seed"]]["runs"]["1.0"]["final_score"].values()))
                print(f"  {p} {r['seed']}: B {r['arms']['B']['global_l2']:.6e} vs etapa A {ref:.6e}  (dif rel {abs(r['arms']['B']['global_l2'] - ref) / ref:.1e})")
    for p, rows in data.items():
        print(f"\n=== {p} (n={len(rows)}) ===")
        B = np.array([r["arms"]["B"]["global_l2"] for r in rows])
        Bc = np.array([r["arms"]["B"]["critical_l2"] for r in rows])
        Bt = np.array([r["arms"]["B"]["lbfgs_seconds"] for r in rows])
        print(f" B: error global {np.round(B, 6).tolist()}")
        out[p] = {}
        for arm in ARMS:
            g = np.array([r["arms"][arm]["global_l2"] for r in rows])
            c = np.array([r["arms"][arm]["critical_l2"] for r in rows])
            t = np.array([r["arms"][arm]["lbfgs_seconds"] for r in rows])
            fin = all(r["arms"][arm]["finite"] for r in rows)
            ratio, cr, tr = g / B, c / Bc, t / Bt
            out[p][arm] = dict(ratio=ratio.tolist(), median_ratio=float(np.median(ratio)), gm_ratio=gm(ratio), n_better=int((ratio < 1).sum()),
                               critical_gm=gm(cr), time_gm=gm(tr), finite=fin, p_exact=exact_flip_p(np.log(ratio)),
                               status=[r["arms"][arm]["lbfgs_status"] for r in rows])
            o = out[p][arm]
            print(f" {arm}: r por semilla {np.round(ratio, 2).tolist()} | mediana {o['median_ratio']:.3f}, GM {o['gm_ratio']:.3f}, mejora {o['n_better']}/{len(rows)} | "
                  f"region critica GM {o['critical_gm']:.3f} | reloj GM {o['time_gm']:.2f} | finito {fin} | p exacto {o['p_exact']:.3f}")
        # contra N
        for arm in ("P",):
            g = np.array([r["arms"][arm]["global_l2"] for r in rows])
            n_ = np.array([r["arms"]["N"]["global_l2"] for r in rows])
            rr = g / n_
            out[p][f"{arm}_vs_N"] = dict(ratio=rr.tolist(), median=float(np.median(rr)), n_better=int((rr < 1).sum()), gm=gm(rr))
            print(f" {arm}/N: {np.round(rr, 2).tolist()} | mediana {np.median(rr):.3f}, mejora {int((rr < 1).sum())}/{len(rr)}")
        sh = [r["top20_share_Astar"] for r in rows]
        print(f" reparto del 20 % superior de A* (descriptivo): {np.round(sh, 2).tolist()}")
        out[p]["top20_share_Astar"] = sh

    print("\n--- H-CEIL-1 (para P o H, ambos problemas) ---")
    verdict = {}
    for arm in ("P", "H"):
        ok_main, ok_guard = True, True
        for p in SEEDS:
            o = out[p][arm]
            ok_main &= (o["median_ratio"] <= 0.85) and (o["n_better"] >= 4)
            ok_guard &= (o["critical_gm"] <= 1.10) and (o["time_gm"] <= 1.5) and o["finite"]
        verdict[arm] = dict(main=bool(ok_main), guards=bool(ok_guard), passes=bool(ok_main and ok_guard))
        print(f" {arm}: criterio principal {'SI' if ok_main else 'NO'} | guardias {'SI' if ok_guard else 'NO'} -> {'SE SOSTIENE' if verdict[arm]['passes'] else 'no se sostiene'}")
    h1 = any(v["passes"] for v in verdict.values())
    # H-CEIL-2 solo para P (H necesitaria N-H, no corrido)
    h2 = None
    if verdict["P"]["passes"]:
        h2 = all(out[p]["P_vs_N"]["median"] <= 0.90 and out[p]["P_vs_N"]["n_better"] >= 4 for p in SEEDS)
    print(f"\n H-CEIL-1: {'SE SOSTIENE' if h1 else 'NO se sostiene'}   H-CEIL-2: {'(no aplica: ningun brazo paso H-CEIL-1)' if h2 is None else ('SE SOSTIENE' if h2 else 'NO')}")
    out["verdict"] = dict(arms=verdict, h_ceil_1=bool(h1), h_ceil_2=h2)
    (HERE / "results/ceiling_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
