"""EXP-0028 Fase 1c (DFG 2D-1 Re=20): analisis preregistrado. Reusa SIN cambios el criterio, los umbrales y los contrastes de analyze_phase1.py (spec.md, Enmienda v1.1); solo cambia el problema y la metrica.

Metrica primaria (fijada aqui, antes de ver datos): l2re_flat = error L2 relativo aplanado de (u, v, p) contra la referencia de trabajo FEM DFG-L3 (presion con ancla en (2.2, 0.2)); es la misma cifra de
cabecera que en NS2d-CG. Se reportan ADEMAS, con el mismo criterio y como lectura descriptiva (no cambian la clase primaria): los QoI de la campana DFG (error relativo de Cd, error absoluto de Cl,
error relativo de dp contra FEATFLOW), la media de componentes y cada componente (u, v, p).
Receta: Adam 2000 pasos (minibatch 1024 + 256/frontera) + L-BFGS 500 it. sobre el pool completo; brazos F, J, E, I frente a B (10 semillas 4001-4010). S, R, W no se corren (missing_arms_1c.md).
Cada metrica se analiza en dos puntos: fin de Adam y fin de L-BFGS (Fase 1b), igual que en 1a. Las ausencias no se imputan; una razon indefinida (e <= 0 o no finita) se descarta (a1.val).

Uso: python analyze_phase1c_dfg.py --dir DIR [--dir DIR2] [--out analisis_fase1c_dfg]   -> <out>.json y <out>.md
"""
import argparse
import glob
import json
from pathlib import Path

import analyze_phase1 as a1

PROBLEM = "dfg_2d1_re20"
METRICS = {"l2re_flat": "PRIMARIA: L2RE aplanado (u, v, p)", "Cd_relerr": "QoI: error relativo de Cd", "dp_relerr": "QoI: error relativo de dp", "Cl_abserr": "QoI: error absoluto de Cl",
           "l2re_mean_components": "media de componentes", "l2re_u": "componente u", "l2re_v": "componente v", "l2re_p": "componente p"}
a1.PROBLEMS = (PROBLEM,)
a1.LABEL = {PROBLEM: "DFG 2D-1 Re20 (RBF16, Adam 2000 + L-BFGS 500 it.)"}
a1.ARMS = ("F", "J", "E", "I")


def load(dirs):
    R = {}
    for d in dirs:
        for f in glob.glob(str(Path(d) / "dfg-*-seed*.json")):
            r = json.load(open(f))
            if r.get("problem") == PROBLEM and r.get("arm") in ("B",) + a1.ARMS and r.get("seed", 0) >= 4001 and r.get("score_final") and not r.get("smoke_pool"):
                R[(PROBLEM, r["arm"], r["seed"])] = r
    return R


def with_metric(R, m):
    out = {}
    for k, r in R.items():
        c = dict(r)
        c["primary_adam"], c["primary_final"] = r["score_adam"][m], r["score_final"][m]
        out[k] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = load(a.dir)
    res, md = {}, [f"# EXP-0028 Fase 1c — DFG 2D-1 Re20 ({len(R)} corridas)", ""]
    for m, name in METRICS.items():
        res[m] = a1.analyze(with_metric(R, m))[PROBLEM]
        md += [f"## Metrica: {name} (`{m}`)", "", a1.markdown({PROBLEM: res[m]}).split("\n", 1)[1], ""]
    print("\n".join(md))
    if a.out:
        Path(a.out + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        Path(a.out + ".md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
