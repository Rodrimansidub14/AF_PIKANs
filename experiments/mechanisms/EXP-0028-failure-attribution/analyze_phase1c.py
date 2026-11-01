"""EXP-0028 Fase 1c (NS2d-CG): analisis preregistrado. Reusa SIN cambios el criterio, los umbrales y los contrastes de analyze_phase1.py (spec.md, Enmienda v1.1); solo cambia el problema y la metrica.

Metrica primaria (fijada aqui, antes de ver datos): l2re_flat = error L2 relativo aplanado de todas las salidas (u, v, p), la cifra de cabecera de la campana NS2d-CG. Se reportan ADEMAS, con el mismo criterio
y como lectura descriptiva (no cambian la clase primaria): media de componentes (l2re_mean_components) y cada componente (u, v, p), porque v y p dominan el error de campo y u domina el aplanado.
Brazos: F, J, E, I frente a B (10 semillas 4001-4010). Las palancas S, R, W no se corren en NS (missing_arms_1c.md).
Cada metrica se analiza en dos puntos: fin de Adam y fin de L-BFGS (Fase 1b), igual que en 1a.

Uso: python analyze_phase1c.py --dir DIR [--dir DIR2] [--out analisis_fase1c]   -> <out>.json y <out>.md
"""
import argparse
import glob
import json
from pathlib import Path

import analyze_phase1 as a1

PROBLEM = "ns2d_cg"
METRICS = {"l2re_flat": "PRIMARIA: L2RE aplanado", "l2re_mean_components": "media de componentes", "l2re_u": "componente u", "l2re_v": "componente v", "l2re_p": "componente p"}
a1.PROBLEMS = (PROBLEM,)
a1.LABEL = {PROBLEM: "NS2d-CG (RBF16, Adam 20000 + L-BFGS 500 it.)"}
a1.ARMS = ("F", "J", "E", "I")


def load(dirs):
    R = {}
    for d in dirs:
        for f in glob.glob(str(Path(d) / "ns-*-seed*.json")):
            r = json.load(open(f))
            if r.get("problem") == PROBLEM and r.get("arm") in ("B",) + a1.ARMS and r.get("seed", 0) >= 4001 and r.get("score_final"):
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
    res, md = {}, [f"# EXP-0028 Fase 1c — NS2d-CG ({len(R)} corridas)", ""]
    for m, name in METRICS.items():
        res[m] = a1.analyze(with_metric(R, m))[PROBLEM]
        md += [f"## Metrica: {name} (`{m}`)", "", a1.markdown({PROBLEM: res[m]}).split("\n", 1)[1], ""]
    print("\n".join(md))
    if a.out:
        Path(a.out + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        Path(a.out + ".md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
