"""EXP-0030: carga de las corridas de B, E y M en piezo (solo lectura de EXP-0028/0029). Exploratorio y post hoc."""
import glob
import json
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
RES_IN = EXP.parent / "EXP-0028-failure-attribution" / "results"
FUENTES = ("piezo-s1a/exp0028_piezo", "piezo-s1b/exp0028_piezo", "piezo-modal-run/piezo", "piezo-exp0029-final/piezo_exp0029")
SALIDA = EXP / "results"
SALIDA.mkdir(exist_ok=True)
(EXP / "figures").mkdir(exist_ok=True)


def hw_corto(nombre):
    n = nombre or "?"
    for k, v in (("T4", "T4"), ("H100", "H100"), ("A100-SXM4-80GB", "A100-80"), ("A100-SXM4-40GB", "A100-40")):
        if k in n:
            return v
    return n


def cargar():
    """dict[(problema, brazo, semilla)] -> dict(json=..., npz=ruta, hw=...). Solo corridas de 20 000 pasos de B, E y M."""
    R = {}
    for d in FUENTES:
        for f in sorted(glob.glob(str(RES_IN / d / "piezo-*-seed*.json"))):
            if "artifacts" in f:
                continue
            r = json.load(open(f))
            if r.get("adam_steps") != 20000 or r["arm"] not in ("B", "E", "M"):
                continue
            npz = f.replace(".json", "-artifacts.npz")
            z = np.load(npz, allow_pickle=True)
            meta = json.loads(str(z["meta_json"]))
            R[(r["problem"], r["arm"], r["seed"] % 100)] = dict(json=r, npz=npz, hw=hw_corto(meta.get("cuda")), meta=meta)
    return R


def serie(r, clave="l2re_u"):
    h = r["json"]["hist"]
    return np.array([x["step"] for x in h]), np.array([x[clave] for x in h])


def final(r, clave="l2re_u"):
    return float(r["json"]["score_final"][clave])


def robusto(r, clave="l2re_u", desde=18000):
    s, v = serie(r, clave)
    return float(np.median(v[s >= desde]))


def salida_llano(r, umbral=0.5):
    s, v = serie(r, "l2re_u")
    i = np.where(v < umbral)[0]
    return float(s[i[0]]) if len(i) else float("nan")


def md_tabla(cabeza, filas):
    out = ["| " + " | ".join(cabeza) + " |", "|" + "---|" * len(cabeza)]
    out += ["| " + " | ".join(str(c) for c in f) + " |" for f in filas]
    return "\n".join(out)
