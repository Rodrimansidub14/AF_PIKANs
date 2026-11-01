"""EXP-0028 Fase 1e (Enmienda v1.4): analisis de la confirmacion con receta completa de J e I (y de B, F, E, J, I en NS2d-CG y DFG). Criterio UNIFORME de la Enmienda v1.1 (analyze_phase1.py, importado
sin cambios): por problema y brazo X frente a B, pareado por semilla, razon r_s = e_X / e_B, sin agrupar problemas; sirve (mediana <= 0.90, mejora >= 70 %, ninguna semilla > 2x) / daña (mediana >= 1.10 y
empeora >= 70 %) / dependiente de semilla / indiferente. Umbrales fijados en el spec; este archivo no los cambia.

POR SEPARADO se reportan el FIN DE ADAM (`primary_adam`) y el FIN DE L-BFGS (`primary_final`): cada uno con su clase de 1e y su veredicto contra la etiqueta CORRESPONDIENTE (Adam con Adam, L-BFGS con L-BFGS).
Comparaciones (veredicto = el de analyze_phase1d.verdict: sostenido / se pierde / cambia de signo / aparece / sin efecto; con n < 5 PROVISIONAL, con n < 4 «sin veredicto»):
  * contra la etiqueta de la FASE 1 CORTA: ac/kpp/burgers = results/phase1_scored.json (n = 10, Adam 4 000/2 000 + L-BFGS 500 it.); ns = results/phase1c_scored.json y dfg = results/phase1c_dfg_scored.json
    (n = 10, Adam 2 000 + L-BFGS 500 it. en ambos; ojo: la etiqueta de texto del JSON de NS dice 20 000, las corridas guardadas son de 2 000, ver adam_steps);
  * contra la etiqueta de 1d (receta completa, results/phase1d_*scored.json) POR BRAZO cuando 1d tiene ese brazo (en ac/kpp/burgers: F, E, C; J e I NO estan en 1d y por eso el veredicto 1d es «sin etiqueta 1d»;
    NS y DFG no tienen 1d). Las filas de brazos que solo existen en 1d (F, E, C) se muestran como referencia (1e: «-»).
Pareo: J e I de ac/kpp/burgers se parean con el B de 1d (--dir1d). Si ademas hay B de 1e (contingencia --include-b) se usa el de 1d y se informa cuanto difiere (reproducibilidad del B).
Se descartan las corridas de humo (`smoke`/`smoke_pool`), semillas fuera de 4101-4105 y, en NS/DFG, las que no tengan la receta completa (adam_steps 20 000; lbfgs_iters 500 NS, 4 000 DFG).

Uso: python analyze_phase1e.py --dir DIR1E [--dir DIR1E2 ...] [--dir1d DIR1D ...] [--phase1d results/phase1d_ac_scored.json ...] [--metric l2re_flat] [--out results/phase1e_scored]  -> .json y .md
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

import analyze_phase1 as a1
import analyze_phase1d as d1

HERE = Path(__file__).resolve().parent
SEEDS_1E = range(4101, 4106)
HOT = ("sirve", "daña")
PROBLEMS = {   # nombre en el JSON -> ficha
    "ns2d_cg": dict(key="ns", label="NS2d-CG (RBF16)", full="Adam 20 000 f32 + L-BFGS 500 it.", short="Adam 2 000 + L-BFGS 500 it. (1c, t1)", adam=20000, lbfgs=500,
                    planned=("F", "E", "J", "I"), prior="phase1c", own_b=True),
    "dfg_2d1_re20": dict(key="dfg", label="DFG 2D-1 Re20 (RBF16)", full="Adam 20 000 (minibatch) f32 + L-BFGS 4 000 it.", short="Adam 2 000 + L-BFGS 500 it. (1c)", adam=20000, lbfgs=4000,
                         planned=("F", "E", "J", "I"), prior="phase1c_dfg", own_b=True),
    "ac": dict(key="ac", label="Allen-Cahn d=1e-3", full=d1.RECIPE_TXT["ac"], short="Adam 4 000 + L-BFGS 500 it. (fase 1)", planned=("J", "I"), prior="phase1", own_b=False),
    "kpp": dict(key="kpp", label="Fisher-KPP", full=d1.RECIPE_TXT["kpp"], short="Adam 4 000 + L-BFGS 500 it. (fase 1)", planned=("J", "I"), prior="phase1", own_b=False),
    "burgers": dict(key="burgers", label="Burgers RBF48", full=d1.RECIPE_TXT["burgers"], short="Adam 2 000 + L-BFGS 500 it. (fase 1)", planned=("J", "I"), prior="phase1", own_b=False),
}
ARM_ORDER = ("F", "J", "E", "I", "C", "S", "R", "W")
NONE = {"n": 0, "clase": "sin datos"}


def _accept(r, spec_only_1d):
    if r.get("smoke") or r.get("smoke_pool") or r.get("seed") not in SEEDS_1E:
        return False
    sp = PROBLEMS.get(r.get("problem"))
    if sp is None:
        return False
    if sp["key"] in ("ns", "dfg"):
        return not spec_only_1d and r.get("adam_steps") == sp["adam"] and r.get("lbfgs_iters") == sp["lbfgs"] and bool(r.get("score_final"))
    return r.get("phase") == "1d" and "primary_final" in r


def load(dirs, metric="l2re_flat", only_1d=False):
    R = {}
    for d in dirs:
        for f in glob.glob(str(Path(d) / "*-seed*.json")):
            if f.endswith("note.json"):
                continue
            r = json.load(open(f))
            if not r.get("arm") or not _accept(r, only_1d):
                continue
            if PROBLEMS[r["problem"]]["key"] in ("ns", "dfg"):
                r = dict(r, primary_adam=r["score_adam"][metric], primary_final=r["score_final"][metric])
            R[(r["problem"], r["arm"], r["seed"])] = r
    return R


def prior_label(prior, problem, arm, stage):
    """Etiqueta de la fase 1 corta (stage 'adam' o 'final'). `prior[problema]` ya es el bloque del problema (para NS/DFG, el de la metrica elegida)."""
    return ((prior.get(problem) or {}).get("brazos", {}).get(arm, {})).get(stage, NONE)


def d1_label(d1_blocks, problem, arm, stage):
    key = "completa_adam" if stage == "adam" else "completa_final"
    for blk in d1_blocks:
        b = blk.get(problem, {})
        if b.get("n_corridas", 0) and arm in b.get("brazos", {}) and b["brazos"][arm][key].get("n"):
            return b["brazos"][arm][key]
    return NONE


def lbfgs_info(R, problem, arm):
    rs = [r for (p, a, _), r in R.items() if p == problem and a == arm]
    st = {}
    for r in rs:
        st[r.get("lbfgs_status", "?")] = st.get(r.get("lbfgs_status", "?"), 0) + 1
    its = [r["lbfgs_iterations"] for r in rs if r.get("lbfgs_iterations") is not None]
    return dict(estados=st, iteraciones_mediana=float(np.median(its)) if its else None, n=len(rs))


def b_info(R1e, R1d, Rm, problem):
    sp = PROBLEMS[problem]
    bf = [r for (p, a, _), r in Rm.items() if p == problem and a == "B"]
    out = dict(n=len(bf), fuente=("1e" if sp["own_b"] else "1d"), seeds=sorted(r["seed"] for r in bf))
    for k, tag in (("primary_adam", "mediana_adam"), ("primary_final", "mediana_final")):
        v = [r[k] for r in bf if r.get(k)]
        out[tag] = float(np.median(v)) if v else None
    if not sp["own_b"]:
        out["frozen"] = d1.FROZEN_B.get(problem)
        both = [(R1e[k], R1d[k]) for k in R1e if k[1] == "B" and k[0] == problem and k in R1d]
        if both:
            out["B_1e_vs_1d"] = dict(n=len(both), max_abs_log_ratio_final=float(max(abs(np.log(a["primary_final"] / b["primary_final"])) for a, b in both)),
                                     max_abs_log_ratio_adam=float(max(abs(np.log(a["primary_adam"] / b["primary_adam"])) for a, b in both)),
                                     identico=bool(all(a["primary_final"] == b["primary_final"] and a["primary_adam"] == b["primary_adam"] for a, b in both)))
    return out


def analyze(R1e, R1d, prior, d1_blocks, metric):
    Rm = dict(R1e)
    Rm.update(R1d)                                       # el B de 1d manda (y F, E, C de 1d para contrastes)
    res = {}
    for pr, sp in PROBLEMS.items():
        n1e = sum(1 for k in R1e if k[0] == pr)
        n1d = sum(1 for k in R1d if k[0] == pr)
        if not n1e and not n1d:
            continue
        arms = {a for (p, a, _) in Rm if p == pr and a != "B"} | set(sp["planned"])
        arms |= {a for blk in d1_blocks for a in blk.get(pr, {}).get("brazos", {})}
        arms = sorted(arms, key=lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99)
        blk = dict(label=sp["label"], receta_completa=sp["full"], receta_corta=sp["short"], n_corridas_1e=n1e, n_corridas_1d_usadas=n1d, B=b_info(R1e, R1d, Rm, pr), brazos={}, contrastes={})
        for arm in arms:
            has = any(k[0] == pr and k[1] == arm for k in R1e)
            row = dict(en_1e=has, lbfgs=lbfgs_info(Rm, pr, arm) if has else None)
            for stage, key in (("adam", "primary_adam"), ("final", "primary_final")):
                c1e = a1.classify([x for _, x in a1.pair(Rm, pr, arm, "B", key)]) if has else NONE
                c_p1 = prior_label(prior, pr, arm, stage)
                c_1d = d1_label(d1_blocks, pr, arm, stage)
                if has:
                    v_p1, det_p1 = d1.verdict(c_p1, c1e)
                    v_1d, det_1d = d1.verdict(c_1d, c1e) if c_1d.get("n") else ("sin etiqueta 1d", "1d no tiene este brazo" if sp["key"] in ("ac", "kpp", "burgers") else "este problema no tiene 1d")
                else:
                    v_p1 = v_1d = "-"
                    det_p1 = det_1d = ""
                row[stage] = dict(fase1=c_p1, d1=c_1d, e1=c1e, veredicto_vs_fase1=v_p1, detalle_vs_fase1=det_p1, veredicto_vs_1d=v_1d, detalle_vs_1d=det_1d)
            if has:
                row["fase_efecto_1e"] = a1.phase(row["adam"]["e1"], row["final"]["e1"])
            blk["brazos"][arm] = row
        for a, b, name in a1.CONTRASTS:
            ca, cf = a1.contrast(Rm, pr, a, b, "primary_adam"), a1.contrast(Rm, pr, a, b, "primary_final")
            if cf.get("n") or ca.get("n"):
                blk["contrastes"][f"{a}_vs_{b}"] = dict(nombre=name, adam=ca, final=cf)
        blk["atribucion"] = a1.attribution(Rm, pr)
        res[pr] = blk
    return res


def fmt(c):
    return d1.fmt(c) + (f" {c['clase']}" if c.get("n") else "")


def markdown(res, metric):
    L = ["# EXP-0028 Fase 1e — confirmacion con receta completa (criterio uniforme, por problema; fin de Adam y fin de L-BFGS por separado)", "",
         f"Metrica de NS/DFG: `{metric}`. Veredicto = sostenido / se pierde / cambia de signo / aparece / sin efecto frente a la etiqueta de la fase 1 corta y, si el brazo existe en 1d, frente a la de 1d. "
         "n < 5 -> PROVISIONAL; n < 4 -> sin veredicto. J e I de ac/kpp/burgers se parean con el B de 1d.", ""]
    for pr, b in res.items():
        B = b["B"]
        L += [f"## {b['label']} — 1e: {b['n_corridas_1e']} corridas (+ {b['n_corridas_1d_usadas']} de 1d para el pareo)", "", f"Receta completa: {b['receta_completa'].rstrip('.')}. Fase 1 corta: {b['receta_corta']}.", ""]
        if B["n"]:
            t = f"B (fuente {B['fuente']}, n = {B['n']}): mediana fin de Adam {B['mediana_adam']:.3e}, tras L-BFGS {B['mediana_final']:.3e}."
            if B.get("frozen"):
                t += f" Linea base congelada {B['frozen'][0]:.3e} / {B['frozen'][1]:.3e} (razon final {B['mediana_final'] / B['frozen'][1]:.2f}x, informativo)."
            if B.get("B_1e_vs_1d"):
                v = B["B_1e_vs_1d"]
                t += f" B de 1e frente a B de 1d: {'IDENTICO' if v['identico'] else 'distinto'} (max |log razon| final {v['max_abs_log_ratio_final']:.3g}, Adam {v['max_abs_log_ratio_adam']:.3g}; n = {v['n']})."
            L += [t, ""]
        for stage, name in (("final", "FIN DE L-BFGS (primario)"), ("adam", "FIN DE ADAM")):
            L += [f"### {name}", "", "| brazo | fase 1 corta | 1d | **1e** | veredicto vs fase 1 | veredicto vs 1d |", "|---|---|---|---|---|---|"]
            for arm, v in b["brazos"].items():
                s = v[stage]
                vp = s["veredicto_vs_fase1"] + (f" — {s['detalle_vs_fase1']}" if s["detalle_vs_fase1"] else "")
                vd = s["veredicto_vs_1d"] + (f" — {s['detalle_vs_1d']}" if s["detalle_vs_1d"] and s["veredicto_vs_1d"] != "-" else "")
                L.append(f"| {arm}{'' if v['en_1e'] else ' (solo referencia)'} | {fmt(s['fase1'])} | {fmt(s['d1'])} | **{fmt(s['e1']) if v['en_1e'] else '-'}** | **{vp}** | {vd} |")
            L.append("")
        L += ["Fase del efecto y L-BFGS (1e):", ""]
        for arm, v in b["brazos"].items():
            if v["en_1e"]:
                L.append(f"- {arm}: {v['fase_efecto_1e']}; L-BFGS {json.dumps(v['lbfgs']['estados'], ensure_ascii=False)}, iteraciones (mediana) {v['lbfgs']['iteraciones_mediana']}")
        L.append("")
        for k, c in b["contrastes"].items():
            f, a = c["final"], c["adam"]
            L.append(f"- {c['nombre']}: L-BFGS mediana {f['mediana']:.3f} (A mejor {f['a_mejor_k']}/{f['n']}; respaldado={f['respaldado']}, contrario={f['contrario']})" if f.get("n") else f"- {c['nombre']}: sin datos")
            if a.get("n"):
                L[-1] += f"; Adam mediana {a['mediana']:.3f} ({a['a_mejor_k']}/{a['n']})"
        L += ["", "Atribucion 1e: " + json.dumps(b["atribucion"], ensure_ascii=False), ""]
    L += ["## Resumen para predictions.md (lo confirmado a receta completa pesa mas que lo corto)", "",
          "| problema | brazo | fase 1 (Adam / L-BFGS) | 1d (Adam / L-BFGS) | 1e (Adam / L-BFGS) | veredicto Adam | veredicto L-BFGS | vs 1d (L-BFGS) |", "|---|---|---|---|---|---|---|---|"]
    for pr, b in res.items():
        for arm, v in b["brazos"].items():
            if v["en_1e"]:
                a, f = v["adam"], v["final"]
                L.append(f"| {b['label']} | {arm} | {a['fase1'].get('clase')} / {f['fase1'].get('clase')} | {a['d1'].get('clase')} / {f['d1'].get('clase')} | {a['e1'].get('clase')} / {f['e1'].get('clase')} | "
                         f"{a['veredicto_vs_fase1']} | {f['veredicto_vs_fase1']} | {f['veredicto_vs_1d']} |")
    return "\n".join(L)


def load_priors(phase1, phase1c, phase1c_dfg, metric):
    """ac/kpp/burgers -> phase1_scored[problema]; ns/dfg -> phase1c*_scored[metrica]."""
    prior = {}
    if Path(phase1).exists():
        j = json.load(open(phase1))
        prior.update({k: j[k] for k in ("ac", "kpp", "burgers") if k in j})
    for name, path in (("ns2d_cg", phase1c), ("dfg_2d1_re20", phase1c_dfg)):
        if Path(path).exists():
            prior[name] = json.load(open(path)).get(metric)
    return prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, help="salidas de 1e (ns-*, dfg-*, ac/kpp/burgers-*)")
    ap.add_argument("--dir1d", action="append", default=[], help="salidas de 1d (B pareado y F/E/C de referencia)")
    ap.add_argument("--phase1", default=str(HERE / "results/phase1_scored.json"))
    ap.add_argument("--phase1c", default=str(HERE / "results/phase1c_scored.json"))
    ap.add_argument("--phase1c-dfg", default=str(HERE / "results/phase1c_dfg_scored.json"))
    ap.add_argument("--phase1d", action="append", default=[], help="phase1d_*scored.json (etiquetas de 1d por brazo)")
    ap.add_argument("--metric", default="l2re_flat", help="metrica de NS/DFG (score_*), debe existir en phase1c*_scored.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = analyze(load(a.dir, a.metric), load(a.dir1d, a.metric, only_1d=True), load_priors(a.phase1, a.phase1c, a.phase1c_dfg, a.metric), [json.load(open(p)) for p in a.phase1d], a.metric)
    md = markdown(res, a.metric)
    print(md)
    if a.out:
        Path(a.out + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        Path(a.out + ".md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
