"""Recalcula, solo desde los JSON versionados por corrida, las cifras de piezo (converso) que cita la tesis, y las compara con
lo escrito en docs/plans/2026-10-26-estado-para-redaccion.md. Es el equivalente de «reconstruir los certificados»: si una cifra citada
no sale de los datos guardados, este script sale con 1.  Uso:  python tools/reproduce_numbers.py
Métrica robusta = mediana de l2re_u en los pasos >= 18 000 (definida DESPUÉS de ver los datos: post hoc). Salida del llano = primer paso de
evaluación con l2re_u < 0.5."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "experiments/mechanisms/EXP-0028-failure-attribution/results"
DIRS = ("piezo-s1a/exp0028_piezo", "piezo-s1b/exp0028_piezo", "piezo-modal-run/piezo", "piezo-exp0029-final/piezo_exp0029")
SEMILLAS = (28, 29, 30, 31, 32)


def cargar():
    R = {}
    for d in DIRS:
        for f in glob.glob(str(RES / d / "piezo-*-seed*.json")):
            if "artifacts" in f:
                continue
            r = json.load(open(f, encoding="utf-8"))
            if r.get("adam_steps") == 20000 and r["arm"] in ("B", "E", "M"):
                R[(r["problem"], r["arm"], r["seed"] % 100)] = r
    return R


def ultimo(r):
    return r["score_final"]["l2re_u"]


def robusto(r):
    return float(np.median([h["l2re_u"] for h in r["hist"] if h["step"] >= 18000]))


def salida(r, umbral=0.5):
    return next((h["step"] for h in r["hist"] if h["l2re_u"] < umbral), np.nan)


def main():
    R = cargar()
    falta = [(a, s) for a in ("B", "E") for s in SEMILLAS if ("converse", a, s) not in R]
    if falta:
        print("FALTAN corridas B/E del converso:", falta)
        return 1
    ok = True

    def chequear(nombre, valor, esperado, tol):
        nonlocal ok
        bien = abs(valor - esperado) <= tol
        ok &= bien
        print(f"{'OK ' if bien else 'MAL'} {nombre}: {valor:.4g} (documentado {esperado})")

    B = {s: R[("converse", "B", s)] for s in SEMILLAS}
    E = {s: R[("converse", "E", s)] for s in SEMILLAS}
    chequear("media del último paso, B", np.mean([ultimo(B[s]) for s in SEMILLAS]), 0.152, 0.002)
    chequear("media del último paso, E", np.mean([ultimo(E[s]) for s in SEMILLAS]), 0.158, 0.002)
    rb, re_ = [robusto(B[s]) for s in SEMILLAS], [robusto(E[s]) for s in SEMILLAS]
    chequear("error robusto mínimo, E", min(re_), 0.033, 0.002)
    chequear("error robusto máximo, E", max(re_), 0.102, 0.002)
    chequear("error robusto mínimo, B", min(rb), 0.039, 0.002)
    chequear("error robusto máximo, B", max(rb), 0.391, 0.002)
    sb, se = [salida(B[s]) for s in SEMILLAS], [salida(E[s]) for s in SEMILLAS]
    chequear("mediana de salida del llano, B", float(np.nanmedian(sb)), 7000, 0)
    chequear("mediana de salida del llano, E", float(np.nanmedian(se)), 5500, 0)
    antes = sum(e < b for e, b in zip(se, sb))
    chequear("semillas donde E sale antes que B (de 5)", antes, 5, 0)
    print("\npor semilla (B, E): salida", list(zip(sb, se)), "| robusto", [(round(a, 3), round(b, 3)) for a, b in zip(rb, re_)])
    print("TODO COINCIDE" if ok else "HAY CIFRAS QUE NO SALEN DE LOS DATOS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
