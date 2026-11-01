"""Analisis posterior de la rebanada con artefactos (EXP-0028, Enmienda v1.6). Lee `<problema>-<brazo>-seed<semilla>-artifacts.npz` de uno o varios directorios y escribe `slice_summary.md` + `slice_curves.csv`:
  1. curva EFECTO-vs-PRESUPUESTO: error en la malla de evaluacion a cada instantanea de Adam (y tras 100/250/500 it. de L-BFGS) y razon e_X/e_B pareada por semilla (mediana, k/n) -> lee en una sola corrida
     cuanto del efecto de la receta completa ya esta presente en M/2, M, ...;
  2. por brazo y paso completo: espectro de P (cond mediana/max), traza de B por arista (primer y ultimo recalculo), bases RBF "muertas" (ocupacion < 1 %) por capa, mapa de asentamiento (cambio relativo de
     coeficientes por arista entre instantaneas: mediana y p90 por capa);
  3. tamano por corrida.
Uso: python analyze_slice.py --dir OUT1 [--dir OUT2 ...] --out DIR"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PRIMARY = {"ac": "l2re_mat", "kpp": "l2re_mat", "burgers": "l2re_coleHopf", "ns": "l2re_flat", "dfg": "l2re_flat"}
NAME = re.compile(r"(?P<p>[a-z0-9]+)-(?P<a>[A-Z]+)-seed(?P<s>\d+)-artifacts\.npz")


def load(dirs):
    runs = {}
    for d in dirs:
        for f in sorted(Path(d).rglob("*-artifacts.npz")):
            m = NAME.fullmatch(f.name)
            if not m:
                continue
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z["meta_json"]))
            runs[(m["p"], m["a"], int(m["s"]))] = dict(z=z, meta=meta, mb=f.stat().st_size / 2 ** 20, path=f)
    return runs


def curve(run, problem):
    key = PRIMARY[problem]
    return {h["key"]: (h["step"], h.get(key)) for h in run["meta"]["score_hist"] if h.get(key) is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = load(a.dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md, rows = ["# EXP-0028 rebanada con artefactos: resumen", ""], []
    problems = sorted({p for p, _, _ in runs})
    for p in problems:
        seeds = sorted({s for q, _, s in runs if q == p})
        arms = sorted({r for q, r, _ in runs if q == p}, key=lambda x: "BFJEI".index(x) if x in "BFJEI" else 9)
        md += [f"## {p}  (metrica {PRIMARY[p]}; semillas {seeds})", ""]
        keys = sorted({k for (q, r, s), v in runs.items() if q == p for k in curve(v, p)})
        md += ["### error y razon e_X/e_B por instantanea (mediana pareada; entre parentesis k/n con e_X < e_B)", "", "| instantanea | paso | B (mediana) | " + " | ".join(f"{x}/B" for x in arms if x != "B") + " |",
               "|---|---|---|" + "---|" * (len(arms) - 1)]
        for k in keys:
            e = {(r, s): curve(runs[(p, r, s)], p).get(k) for r in arms for s in seeds if (p, r, s) in runs}
            eb = [e[("B", s)][1] for s in seeds if e.get(("B", s))]
            step = next((v[0] for v in e.values() if v), "")
            cells = []
            for r in arms:
                if r == "B":
                    continue
                rat = [e[(r, s)][1] / e[("B", s)][1] for s in seeds if e.get((r, s)) and e.get(("B", s))]
                cells.append(f"{np.median(rat):.3f} ({sum(x < 1 for x in rat)}/{len(rat)})" if rat else "-")
                for s in seeds:
                    if e.get((r, s)) and e.get(("B", s)):
                        rows.append(dict(problem=p, key=k, step=step, arm=r, seed=s, e_X=e[(r, s)][1], e_B=e[("B", s)][1], ratio=e[(r, s)][1] / e[("B", s)][1]))
            md.append(f"| {k} | {step} | {np.median(eb):.4g} | " + " | ".join(cells) + " |" if eb else f"| {k} | {step} | - | " + " | ".join("-" for _ in cells) + " |")
        md += ["", "### espectro de P, bases muertas y asentamiento (ultima instantanea de Adam con campos)", "", "| brazo | semilla | MB | cond(P) mediana/max por capa | traza B por arista (1.er -> ult. recalculo, mediana capa 1) | bases con ocupacion < 1 % por capa | asentamiento mediano/p90 por capa (ultimo intervalo) |", "|---|---|---|---|---|---|---|"]
        for r in arms:
            for s in seeds:
                v = runs.get((p, r, s))
                if not v:
                    continue
                z = v["z"]
                names = list(z.files)
                full = sorted({n.split("/")[0] for n in names if n.endswith("/fields/values") and n.startswith("a")})
                last = full[-1] if full else None
                cond = "; ".join(f"{np.median(z[n]):.3g}/{np.max(z[n]):.3g}" for n in sorted(x for x in names if last and x.startswith(f"{last}/P_cond/"))) or "-"
                refs = sorted({n.split("/")[0] for n in names if re.fullmatch(r"refresh\d+/step", n)})
                if refs and f"{refs[0]}/layer1/trace" in names:
                    tr = f"{np.median(z[refs[0] + '/layer1/trace']):.3g} -> {np.median(z[refs[-1] + '/layer1/trace']):.3g}"
                else:
                    tr = "-"
                dead = "; ".join(f"{int((z[n] < 0.01).sum())}/{z[n].size}" for n in sorted(x for x in names if last and x.startswith(f"{last}/occupancy/"))) or "-"
                st = sorted({n.split("/")[1] for n in names if n.startswith("settle/")})
                sett = "; ".join(f"{np.median(z[n]):.3g}/{np.percentile(z[n], 90):.3g}" for n in sorted(x for x in names if st and x.startswith(f"settle/{st[-1]}/"))) or "-"
                md.append(f"| {r} | {s} | {v['mb']:.1f} | {cond} | {tr} | {dead} | {sett} |")
        md.append("")
    sizes = [v["mb"] for v in runs.values()]
    md += ["## Tamano", "", f"{len(sizes)} corridas; total {sum(sizes):.1f} MB; max {max(sizes):.1f} MB por corrida." if sizes else "sin corridas"]
    (out / "slice_summary.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / "slice_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "key", "step", "arm", "seed", "e_X", "e_B", "ratio"])
        w.writeheader()
        w.writerows(rows)
    print("escrito", out / "slice_summary.md", len(rows), "filas")


if __name__ == "__main__":
    main()
