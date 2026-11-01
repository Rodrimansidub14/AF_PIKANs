"""EXP-0028 Fase 1d (Enmienda v1.2): analisis de la confirmacion con receta completa. Criterio UNIFORME de la Enmienda v1.1 (el mismo de analyze_phase1.py, importado sin cambios):
por problema y brazo X frente a B, pareado por semilla, razon r_s = e_X / e_B (L2 relativo), sin agrupar problemas;
  sirve        : mediana <= 0.90, mejora en >= 70 % de las semillas (n = 5 -> 4 de 5) y ninguna semilla con r > 2
  daña         : mediana >= 1.10 y empeora en >= 70 % (4 de 5)
  dependiente  : ni una ni otra y alguna semilla fuera de [0.5, 2]
  indiferente  : el resto
Se aplica al error tras L-BFGS (`primary_final`, primario) y al final de Adam (`primary_adam`, fase del efecto). Umbrales fijados en el spec; este archivo no los cambia.

Comparacion con la etiqueta de la fase 1 (lee results/phase1_scored.json, misma clase por brazo y problema; la fase 1 uso n = 10 para F/E y n = 5 para C):
  sostenido        : la fase 1 era sirve/daña y 1d da la MISMA clase
  se pierde        : la fase 1 era sirve/daña y 1d da dependiente/indiferente (se anota si la mediana conserva la direccion: «debilitado», o queda en la banda [0.90, 1.10]: «neutro»)
  cambia de signo  : sirve <-> daña, o la mediana de 1d cae al lado opuesto (<= 0.90 frente a >= 1.10) del de la fase 1 aunque la clase no sea sirve/daña
  aparece          : la fase 1 NO era sirve/daña y 1d si lo es (el efecto solo existe a receta completa)
  sin efecto       : ni la fase 1 ni 1d dan sirve/daña (se anota si la direccion de la mediana cambio)
Las ausencias no se imputan; con n < 5 el veredicto se marca PROVISIONAL, con n < 4 «sin veredicto». Las corridas de humo (`smoke: true`) se descartan.
Tambien: contraste D5 (E frente a F), H-ATTR-A/B a receta completa, B absoluto frente a la linea base congelada (informativo, sin umbral) y estados de L-BFGS.

Uso: python analyze_phase1d.py --dir DIR [--dir DIR2] [--phase1 results/phase1_scored.json] [--out analisis_fase1d]   -> escribe <out>.json y <out>.md
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

import analyze_phase1 as a1

HERE = Path(__file__).resolve().parent
PROBLEMS = ("ac", "burgers", "kpp")
LABEL = {"ac": "Allen-Cahn d=1e-3", "burgers": "Burgers RBF48", "kpp": "Fisher-KPP"}
PLANNED = {"ac": ("E", "F", "C"), "burgers": ("F", "E"), "kpp": ("E", "F")}
HOT = ("sirve", "daña")
# B congelado (baseline v1/v2), solo para el control de sanidad de la receta: (fin de Adam, tras L-BFGS)
FROZEN_B = {"ac": (2.838e-2, 2.686e-3), "burgers": (5.415e-3, 7.165e-5), "kpp": None}
RECIPE_TXT = {"ac": "Adam 40 000 f32 + L-BFGS f32 (tope 300 s / 15 000 it.)", "burgers": "16 384 pts, Adam 24 000 f32 + L-BFGS f64 (tope 1 800 s)",
              "kpp": "receta de Allen-Cahn (heredada; sin linea base propia)"}


def load_1d(dirs):
    R = {}
    for d in dirs:
        for f in glob.glob(str(Path(d) / "*-seed*.json")):
            if f.endswith(".note.json"):
                continue
            r = json.load(open(f))
            if r.get("smoke") or r.get("phase") != "1d" or r.get("problem") not in PROBLEMS or r.get("seed", 0) < 4101:
                continue
            R[(r["problem"], r["arm"], r["seed"])] = r
    return R


def side(med):
    return "mejora" if med <= a1.MED_GOOD else "empeora" if med >= a1.MED_BAD else "neutro"


def verdict(c1, cd):
    """c1: clase de la fase 1 (dict de analyze_phase1.classify); cd: clase de 1d. Devuelve (veredicto, detalle)."""
    if not cd.get("n"):
        return "sin datos", ""
    n = cd["n"]
    if n < 4:
        return "sin veredicto", f"n = {n} < 4"
    l1, ld = c1.get("clase"), cd["clase"]
    s1 = side(c1["mediana"]) if c1.get("n") else None
    sd = side(cd["mediana"])
    if l1 in HOT:
        if ld == l1:
            v, det = "sostenido", ""
        elif ld in HOT:
            v, det = "cambia de signo", f"{l1} -> {ld}"
        elif {s1, sd} == {"mejora", "empeora"}:
            v, det = "cambia de signo", f"{l1} -> {ld}, la mediana pasa de {s1} a {sd}"
        else:
            v, det = "se pierde", ("debilitado: la mediana conserva la direccion" if sd == s1 else "neutro: la mediana cae en la banda [0.90, 1.10]") + f" ({ld})"
    else:
        if ld in HOT:
            v, det = "aparece", f"fase 1: {l1}"
        elif s1 and {s1, sd} == {"mejora", "empeora"}:
            v, det = "cambia de signo", f"{l1} -> {ld}, la mediana pasa de {s1} a {sd} (ninguna fue sirve/daña)"
        else:
            v, det = "sin efecto", f"fase 1: {l1}; 1d: {ld}" + ("" if s1 == sd else f"; direccion de la mediana: {s1} -> {sd}")
    if n < 5:
        v += " (PROVISIONAL)"
    return v, det


def fmt(c):
    return "-" if not c.get("n") else f"{c['mediana']:.3f} ({c['mejora_k']}/{c['n']}, max {c['razon_max']:.2f})"


def lbfgs_states(R, problem, arm):
    st = {}
    for (p, a, _), r in R.items():
        if p == problem and a == arm:
            st[r.get("lbfgs_status", "?")] = st.get(r.get("lbfgs_status", "?"), 0) + 1
    return st


def b_sanity(R, problem):
    ea = [r["primary_adam"] for (p, a, _), r in R.items() if p == problem and a == "B" and r.get("primary_adam")]
    ef = [r["primary_final"] for (p, a, _), r in R.items() if p == problem and a == "B" and r.get("primary_final")]
    fz = FROZEN_B[problem]
    out = dict(n=len(ef), mediana_adam=float(np.median(ea)) if ea else None, mediana_final=float(np.median(ef)) if ef else None, congelado=fz)
    if fz and ef:
        out["razon_final_vs_congelado"] = out["mediana_final"] / fz[1]
        out["razon_adam_vs_congelado"] = out["mediana_adam"] / fz[0] if ea else None
    return out


def analyze(R, phase1):
    res = {}
    for pr in PROBLEMS:
        arms = sorted({a for (p, a, _) in R if p == pr and a != "B"}, key=lambda a: (a not in PLANNED[pr], a))
        blk = dict(label=LABEL[pr], receta=RECIPE_TXT[pr], n_corridas=sum(1 for k in R if k[0] == pr), brazos={}, contrastes={}, B=b_sanity(R, pr))
        for arm in arms:
            cd_f = a1.classify([x for _, x in a1.pair(R, pr, arm, "B", "primary_final")])
            cd_a = a1.classify([x for _, x in a1.pair(R, pr, arm, "B", "primary_adam")])
            p1 = phase1.get(pr, {}).get("brazos", {}).get(arm, {})
            c1_f, c1_a = p1.get("final", {"n": 0, "clase": "sin datos"}), p1.get("adam", {"n": 0, "clase": "sin datos"})
            vf, df = verdict(c1_f, cd_f)
            va, da = verdict(c1_a, cd_a)
            blk["brazos"][arm] = dict(planeado=arm in PLANNED[pr], fase1_final=c1_f, fase1_adam=c1_a, completa_final=cd_f, completa_adam=cd_a,
                                      veredicto_final=vf, detalle_final=df, veredicto_adam=va, detalle_adam=da, lbfgs=lbfgs_states(R, pr, arm),
                                      fase_efecto_1d=a1.phase(cd_a, cd_f))
        if any(a == "E" for a in arms) and any(a == "F" for a in arms):
            blk["contrastes"]["E_vs_F"] = dict(nombre="D5: E frente a F (¿residuo o solo la base?)", final=a1.contrast(R, pr, "E", "F", "primary_final"),
                                              adam=a1.contrast(R, pr, "E", "F", "primary_adam"), fase1=phase1.get(pr, {}).get("contrastes", {}).get("E_vs_F", {}).get("final"))
        blk["atribucion"] = a1.attribution(R, pr)
        blk["atribucion_fase1"] = phase1.get(pr, {}).get("atribucion")
        res[pr] = blk
    return res


def markdown(res):
    L = ["# EXP-0028 Fase 1d — confirmacion con receta completa (criterio uniforme, n = 5, por problema)", "",
         "Etiqueta de la fase 1 = `results/phase1_scored.md`; veredicto = sostenido / se pierde / cambia de signo (aparece / sin efecto cuando la fase 1 no era sirve/daña). Fase 1: n = 10 (F, E) o 5 (C) a receta corta; 1d: n = 5 a receta completa.", ""]
    for pr, b in res.items():
        B = b["B"]
        L += [f"## {b['label']} ({b['n_corridas']} corridas) — receta: {b['receta']}", ""]
        if B["n"]:
            z = B["congelado"]
            L.append(f"B a receta completa: mediana fin de Adam {B['mediana_adam']:.3e}, tras L-BFGS {B['mediana_final']:.3e} (n = {B['n']})" +
                     (f"; linea base congelada {z[0]:.3e} / {z[1]:.3e} -> razon final {B['razon_final_vs_congelado']:.2f}x (informativo)." if z else "; sin linea base congelada para este problema."))
            L.append("")
        L += ["| brazo | fase 1 (tras L-BFGS) | 1d (tras L-BFGS) | **veredicto** | fase 1 (Adam) | 1d (Adam) | veredicto Adam | L-BFGS 1d | fase del efecto 1d |", "|---|---|---|---|---|---|---|---|---|"]
        for arm, v in b["brazos"].items():
            tag = "" if v["planeado"] else " (no planeado)"
            vf = v["veredicto_final"] + (f" — {v['detalle_final']}" if v["detalle_final"] else "")
            L.append(f"| {arm}{tag} | {fmt(v['fase1_final'])} {v['fase1_final']['clase']} | {fmt(v['completa_final'])} {v['completa_final']['clase']} | **{vf}** | "
                     f"{fmt(v['fase1_adam'])} {v['fase1_adam']['clase']} | {fmt(v['completa_adam'])} {v['completa_adam']['clase']} | {v['veredicto_adam']} | {json.dumps(v['lbfgs'])} | {v['fase_efecto_1d']} |")
        for k, c in b["contrastes"].items():
            f = c["final"]
            f1 = c.get("fase1") or {}
            if f.get("n"):
                txt = f"{c['nombre']}: 1d mediana {f['mediana']:.3f} (E mejor {f['a_mejor_k']}/{f['n']}; respaldado={f['respaldado']}, contrario={f['contrario']})"
                txt += f"; fase 1: {f1['mediana']:.3f} ({f1['a_mejor_k']}/{f1['n']})." if f1.get("n") else "."
            else:
                txt = f"{c['nombre']}: sin datos."
            L += ["", txt]
        L += ["", "Atribucion 1d: " + json.dumps(b["atribucion"], ensure_ascii=False), "Atribucion fase 1: " + json.dumps(b["atribucion_fase1"], ensure_ascii=False), ""]
    L += ["## Resumen para predictions.md (lo confirmado pesa mas que lo corto)", "", "| problema | brazo | fase 1 | 1d | veredicto |", "|---|---|---|---|---|"]
    for pr, b in res.items():
        for arm, v in b["brazos"].items():
            L.append(f"| {b['label']} | {arm} | {v['fase1_final']['clase']} | {v['completa_final']['clase']} | {v['veredicto_final']} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--phase1", default=str(HERE / "results/phase1_scored.json"))
    ap.add_argument("--out", default=None, help="prefijo de salida; escribe .json y .md")
    a = ap.parse_args()
    phase1 = json.load(open(a.phase1))
    res = analyze(load_1d(a.dir), phase1)
    md = markdown(res)
    print(md)
    if a.out:
        Path(a.out + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        Path(a.out + ".md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
