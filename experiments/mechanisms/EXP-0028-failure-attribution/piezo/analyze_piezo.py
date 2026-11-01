"""Análisis de piezo contra predictions.md (firmada 2026-10-25). Lee JSON por corrida de las carpetas dadas; razón e_X/e_B pareada por semilla en l2re_u al paso 20 000.
Reglas: bandas mejora < 0.90, indiferente 0.90-1.10, empeora > 1.10 (mediana); H-PRED-1 (v1.3): (i) mejor y peor brazo entre F,E,J; (ii) F y E a < 0.3 décadas de lo previsto; supera tasa base."""
import argparse, glob, json, math, statistics as st
from pathlib import Path

PRED = {"F": 0.6, "E": 0.5, "J": None}          # razones previstas (J: empeora leve, sin magnitud firmada precisa: +0.08 décadas)
PRED_DIR = {"F": "mejora", "E": "mejora", "J": "empeora leve (>1.10)"}
ap = argparse.ArgumentParser(); ap.add_argument("--dir", action="append", required=True); ap.add_argument("--out", default="")
a = ap.parse_args()
R = {}
for d in a.dir:
    for f in glob.glob(str(Path(d) / "**" / "piezo-*-seed*.json"), recursive=True):
        j = json.load(open(f))
        if j.get("adam_status") != "adam_complete": continue
        R[(j["problem"], j["arm"], j["seed"])] = j
def band(r): return "mejora" if r < 0.90 else ("indiferente" if r <= 1.10 else "empeora")
out = []
res = {}
for prob in ("converse", "direct"):
    seeds = sorted({s for (p, a_, s) in R if p == prob and a_ == "B"})
    out.append(f"\n## {prob} (semillas con B: {seeds})\n")
    out.append("| brazo | n | razón l2re_u por semilla | mediana | k/n mejora | máx | banda | previsto |\n|---|---|---|---|---|---|---|---|")
    for arm in ("F", "E", "J"):
        ss = [s for s in seeds if (prob, arm, s) in R]
        rt = [R[(prob, arm, s)]["primary_final"] / R[(prob, "B", s)]["primary_final"] for s in ss]
        if not rt: continue
        med = st.median(rt); res[(prob, arm)] = med
        out.append(f"| {arm} | {len(rt)} | {', '.join(f'{x:.3f}' for x in rt)} | **{med:.3f}** | {sum(x < 1 for x in rt)}/{len(rt)} | {max(rt):.3f} | {band(med)} | {PRED_DIR[arm]}" + (f" (≈{PRED[arm]})" if PRED[arm] else "") + " |")
    bs = [R[(prob, "B", s)]["primary_final"] for s in seeds]
    out.append(f"\nB l2re_u por semilla: {', '.join(f'{x:.3f}' for x in bs)} (mediana {st.median(bs):.3f}); referencia del paper (MLP): u {'0.107' if prob == 'converse' else '0.117'}.")
    # criterio
    arms = [x for x in ("F", "E", "J") if (prob, x) in res]
    if len(arms) == 3:
        best = min(arms, key=lambda x: res[(prob, x)]); worst = max(arms, key=lambda x: res[(prob, x)])
        okF = abs(math.log10(res[(prob, "F")]) - math.log10(PRED["F"])) < 0.3
        okE = abs(math.log10(res[(prob, "E")]) - math.log10(PRED["E"])) < 0.3
        dirs = {x: band(res[(prob, x)]) for x in arms}
        pred_dirs = {"F": "mejora", "E": "mejora", "J": "empeora"}
        hits = sum(dirs[x] == pred_dirs[x] for x in arms)
        naive = {"F": "mejora", "E": "mejora", "J": "indiferente"}; hits_n = sum(dirs[x] == naive[x] for x in arms)
        out.append(f"\n**Criterio (n = {len(seeds)}, provisional):** mejor observado {best} (previsto E); peor observado {worst} (previsto J) → (i) {'cumple' if (best=='E' and worst=='J') else 'NO cumple'}; "
                   f"(ii) F a {abs(math.log10(res[(prob,'F')])-math.log10(PRED['F'])):.2f} décadas ({'ok' if okF else 'NO'}), E a {abs(math.log10(res[(prob,'E')])-math.log10(PRED['E'])):.2f} ({'ok' if okE else 'NO'}); "
                   f"direcciones acertadas {hits}/3 frente a la tasa base ingenua {hits_n}/3 ({'supera' if hits > hits_n else 'no supera'}). **H-PRED-1 en {prob}: {'respaldada' if (best=='E' and worst=='J' and okF and okE and hits>hits_n) else 'no respaldada'} (provisional)**.")
text = "# Piezo vs predictions.md (n = 3 por problema, provisional)\n" + "\n".join(out)
print(text)
if a.out: Path(a.out).write_text(text, encoding="utf-8")
