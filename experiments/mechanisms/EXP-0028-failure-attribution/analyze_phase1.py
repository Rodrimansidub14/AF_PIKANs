"""EXP-0028 Fase 1 / 1b / ampliacion v1.1: analisis preregistrado (spec.md, Enmienda v1.1). Umbrales fijados en el spec; este archivo no los cambia.

Por problema (sin agrupar problemas) y por brazo X en {F, J, E, I, R, C, W} frente a B, pareado por semilla, sobre la razon r_s = e_X / e_B (L2 relativo, `primary_*`):
  sirve        : mediana <= 0.90, mejora (r < 1) en >= 70 % de las semillas, y ninguna semilla con r > 2
  daña         : mediana >= 1.10 y empeora (r > 1) en >= 70 % de las semillas
  dependiente  : ni una ni otra y alguna semilla fuera de [0.5, 2]
  indiferente  : el resto
Se reporta mediana, k/n, razon maxima, y la fase: el mismo criterio sobre el error al final de Adam y al final de L-BFGS (Fase 1b). «>= 70 %» = k >= ceil(0.7 n)
(7 de 10, 4 de 5). Las ausencias no se imputan. W es un techo oraculo (usa la referencia): se reporta con la etiqueta, no como mecanismo.

Contrastes pareados (D5 y otros, sobre e_A / e_B' por semilla; «respaldado» = mediana <= 0.90 y A mejor en >= 7 de 10; «contrario» = mediana >= 1.10 y A peor en >= 7 de 10;
con n < 10 el conteo exigido es ceil(0.7 n) y se marca n):  E frente a F (D5), J frente a B, E frente a J, I frente a E.
H-ATTR-A y H-ATTR-B se evaluan por problema con las medianas del log de la razon (reglas del spec).

Uso: python analyze_phase1.py --dir DIR [--dir DIR2] [--out analisis_fase1]   -> escribe <out>.json y <out>.md
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

PROBLEMS = ("ac", "ac2e4", "burgers", "kpp")
LABEL = {"ac": "Allen-Cahn d=1e-3", "ac2e4": "Allen-Cahn d=2e-4", "burgers": "Burgers RBF48", "kpp": "Fisher-KPP"}
ARMS = ("F", "J", "E", "I", "S", "R", "C", "W")
CONTRASTS = (("E", "F", "D5: E frente a F (¿residuo o solo la base?)"), ("J", "B", "J frente a B (¿basta reescalar?)"),
             ("E", "J", "E frente a J (¿importan los terminos fuera de la diagonal?)"), ("I", "E", "I frente a E (¿importa acoplar aristas?)"))
MED_GOOD, MED_BAD, FRAC, TAIL, LOW = 0.90, 1.10, 0.70, 2.0, 0.5


def load(dirs):
    R = {}
    for d in dirs:
        for f in glob.glob(str(Path(d) / "*-seed*.json")):
            if f.endswith(".note.json"):
                continue
            r = json.load(open(f))
            if r.get("problem") in PROBLEMS and r.get("arm") and r.get("seed", 0) >= 4001:
                R[(r["problem"], r["arm"], r["seed"])] = r
    return R


def val(R, problem, arm, seed, key):
    r = R.get((problem, arm, seed))
    v = None if r is None else r.get(key)
    return v if (v is not None and math.isfinite(v) and v > 0) else None


def pair(R, problem, a, b, key):
    seeds = sorted({k[2] for k in R if k[0] == problem and k[1] == a} & {k[2] for k in R if k[0] == problem and k[1] == b})
    out = []
    for s in seeds:
        x, y = val(R, problem, a, s, key), val(R, problem, b, s, key)
        if x is not None and y is not None:
            out.append((s, x / y))
    return out


def need(n):
    return int(math.ceil(FRAC * n - 1e-9))


def classify(q):
    """Criterio uniforme del spec (Enmienda v1.1) sobre la lista de razones por semilla."""
    n = len(q)
    if n == 0:
        return dict(n=0, clase="sin datos")
    q = np.asarray(q)
    med, better, worse, mx = float(np.median(q)), int((q < 1).sum()), int((q > 1).sum()), float(q.max())
    if med <= MED_GOOD and better >= need(n) and mx <= TAIL:
        c = "sirve"
    elif med >= MED_BAD and worse >= need(n):
        c = "daña"
    elif ((q < LOW) | (q > TAIL)).any():
        c = "dependiente de semilla"
    else:
        c = "indiferente"
    return dict(n=n, clase=c, mediana=med, geomean=float(np.exp(np.log(q).mean())), mejora_k=better, empeora_k=worse, razon_max=mx, razon_min=float(q.min()))


def phase(c_adam, c_final):
    ea, ef = c_adam.get("clase"), c_final.get("clase")
    if ea in (None, "sin datos") or ef in (None, "sin datos"):
        return "sin datos"
    hot = ("sirve", "daña")
    if ea in hot and ef == ea:
        return "Adam y L-BFGS (se mantiene)"
    if ea in hot and ef not in hot:
        return f"Adam ({ea}); se pierde o cambia en L-BFGS ({ef})"
    if ea in hot and ef in hot and ef != ea:
        return f"Adam ({ea}); L-BFGS invierte ({ef})"
    if ef in hot:
        return f"aparece en L-BFGS ({ef}); Adam: {ea}"
    return "ninguna"


def contrast(R, problem, a, b, key="primary_final"):
    p = pair(R, problem, a, b, key)
    if not p:
        return dict(n=0)
    q = np.asarray([x for _, x in p])
    n = len(q)
    med, k = float(np.median(q)), int((q < 1).sum())
    kw = int((q > 1).sum())
    return dict(n=n, mediana=med, geomean=float(np.exp(np.log(q).mean())), a_mejor_k=k, a_peor_k=kw, razon_max=float(q.max()),
                respaldado=bool(med <= MED_GOOD and k >= need(n)), contrario=bool(med >= MED_BAD and kw >= need(n)), completo_10=bool(n == 10),
                por_semilla={str(s): round(float(x), 4) for s, x in p})


def med_log(R, problem, a, key="primary_final"):
    p = pair(R, problem, a, "B", key)
    return float(np.median(np.log([x for _, x in p]))) if p else None


def attribution(R, problem):
    """H-ATTR-A y H-ATTR-B del spec, por problema, con medianas de log(e_X/e_B) y la clasificacion de E."""
    eB = classify([x for _, x in pair(R, problem, "E", "B", "primary_final")])
    out = dict(E=eB.get("clase"))
    lE, lF, lJ = (med_log(R, problem, x) for x in ("E", "F", "J"))
    if eB.get("n", 0) and lE is not None:
        if eB["mediana"] <= MED_GOOD and eB["mejora_k"] >= need(eB["n"]):
            out["H-ATTR-A"] = dict(aplica=True, F_logB=lF, E_logB=lE, respaldada=bool(lF is not None and lF <= 0.7 * lE),
                                   lectura="la ventaja es de correccion de la base" if lF is not None and lF <= 0.7 * lE else "la ventaja no se explica solo por la base")
        else:
            out["H-ATTR-A"] = dict(aplica=False, motivo="E no mejora a B en este problema (mediana <= 0.90 y >= 7/10)")
        if eB["mediana"] > MED_BAD:
            jw = lJ is not None and math.exp(lJ) > MED_BAD
            fw = lF is not None and math.exp(lF) > MED_BAD
            lect = ("el dano esta en la parte adaptada al residuo" if jw and not fw else
                    "el dano esta en la reparametrizacion misma" if fw else "J no empeora: el dano no es de escala; E vs J lo localiza")
            out["H-ATTR-B"] = dict(aplica=True, J_empeora=jw, F_empeora=fw, lectura=lect)
        else:
            out["H-ATTR-B"] = dict(aplica=False, motivo="E no empeora (mediana <= 1.10)")
    return out


def timing(R, problem):
    out = {}
    for arm in ("B",) + ARMS:
        t = [r.get("total_seconds") for (p, a, _), r in R.items() if p == problem and a == arm and r.get("total_seconds")]
        if t:
            out[arm] = dict(n=len(t), mediana_s=float(np.median(t)))
    return out


def analyze(R):
    res = {}
    for pr in PROBLEMS:
        blk = dict(label=LABEL[pr], n_corridas=sum(1 for k in R if k[0] == pr), brazos={}, contrastes={}, timing=timing(R, pr))
        for arm in ARMS:
            ca = classify([x for _, x in pair(R, pr, arm, "B", "primary_adam")])
            cf = classify([x for _, x in pair(R, pr, arm, "B", "primary_final")])
            blk["brazos"][arm] = dict(adam=ca, final=cf, fase=phase(ca, cf), etiqueta=("oracle_ceiling" if arm == "W" else None))
        for a, b, name in CONTRASTS:
            blk["contrastes"][f"{a}_vs_{b}"] = dict(nombre=name, final=contrast(R, pr, a, b, "primary_final"), adam=contrast(R, pr, a, b, "primary_adam"))
        blk["atribucion"] = attribution(R, pr)
        res[pr] = blk
    return res


def fmt(c):
    return "-" if not c.get("n") else f"{c['mediana']:.3f} ({c['mejora_k']}/{c['n']}, max {c['razon_max']:.2f})"


def markdown(res):
    L = ["# EXP-0028 Fase 1 — analisis (criterio uniforme, por problema)", ""]
    for pr, b in res.items():
        L += [f"## {b['label']} ({b['n_corridas']} corridas)", "", "| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |", "|---|---|---|---|---|---|"]
        for arm, v in b["brazos"].items():
            tag = " (oraculo)" if arm == "W" else ""
            L.append(f"| {arm}{tag} | {fmt(v['final'])} | {v['final']['clase']} | {fmt(v['adam'])} | {v['adam']['clase']} | {v['fase']} |")
        L += ["", "| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |", "|---|---|---|---|---|"]
        for k, v in b["contrastes"].items():
            f, ad = v["final"], v["adam"]
            L.append(f"| {v['nombre']} | " + ("-" if not f.get("n") else f"{f['mediana']:.3f} ({f['a_mejor_k']}/{f['n']})") + f" | {f.get('respaldado', '-')} | {f.get('contrario', '-')} | "
                     + ("-" if not ad.get("n") else f"{ad['mediana']:.3f}") + " |")
        L += ["", "Atribucion: " + json.dumps(b["atribucion"], ensure_ascii=False), "", "Reloj (mediana s por corrida): " + json.dumps(b["timing"]), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--out", default=None, help="prefijo de salida; escribe .json y .md")
    a = ap.parse_args()
    res = analyze(load(a.dir))
    md = markdown(res)
    print(md)
    if a.out:
        Path(a.out + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        Path(a.out + ".md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
