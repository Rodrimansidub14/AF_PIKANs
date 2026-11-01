"""EXP-0029, lectura de T1 (arranque en caliente desde B paso 4000, 1000 pasos, E frente a S). Uso: python analyze_T1.py <carpeta con t1/*.json> [salida.md]
Regla (estimacion, no prueba; proxy de horizonte corto): S 'va bien' si, en el converso, mediana de l2re_phi(S)/l2re_phi(E) <= 0.95 y ninguna semilla > 1.10, y l2re_u(S)/l2re_u(E) mediana <= 1.10 y ninguna > 1.25; en el directo,
l2re_u(S)/l2re_u(E) mediana <= 1.00 sin semilla > 1.25; y los saltos de S (>5x el minimo de los 200 pasos previos) no superan a los de E + 1. Se reporta ademas el costo relativo del paso."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = {}
for f in sorted(glob.glob(str(root / "t1" / "piezo-*-ws4000.json"))):
    r = json.load(open(f))
    rows[(r["problem"], r["arm"], r["seed"])] = r


def spikes(tr, w=200, k=5.0):
    n, last = 0, -10 ** 9
    for i, v in enumerate(tr):
        if i >= 50 and v > k * min(tr[max(0, i - w):i]) and i - last >= w:
            n, last = n + 1, i
    return n


out = ["# T1: E frente a S, arranque en caliente desde B (paso 4000), 1000 pasos\n",
       "| problema | semilla | l2re_u E → S | l2re_phi E → S | cociente u | cociente phi | pérdida final E → S | saltos E / S | s/paso E → S | eventos P4 |", "|---|---|---|---|---|---|---|---|---|---|"]
ratios = {"converse": {"u": [], "phi": []}, "direct": {"u": [], "phi": []}}
dsp = []
for pr in ("converse", "direct"):
    for sd in sorted({k[2] for k in rows if k[0] == pr}):
        e, s = rows.get((pr, "E", sd)), rows.get((pr, "S", sd))
        if not (e and s):
            continue
        he, hs = e["hist"][-1], s["hist"][-1]
        ru, rp = hs["l2re_u"] / he["l2re_u"], hs["l2re_phi"] / he["l2re_phi"]
        ratios[pr]["u"].append(ru), ratios[pr]["phi"].append(rp)
        se, ss = spikes(e["loss_trace"]), spikes(s["loss_trace"])
        dsp.append(ss - se)
        out.append(f"| {pr} | {sd % 100} | {he['l2re_u']:.3f} → {hs['l2re_u']:.3f} | {he['l2re_phi']:.3f} → {hs['l2re_phi']:.3f} | {ru:.2f} | {rp:.2f} | {he['loss']:.3g} → {hs['loss']:.3g} | {se} / {ss} | "
                   f"{e['adam_seconds'] / e['adam_steps']:.3f} → {s['adam_seconds'] / s['adam_steps']:.3f} | {len(s['pre_events'])} |")
med = lambda v: float(np.median(v)) if v else float("nan")
mx = lambda v: max(v) if v else float("nan")
c, d = ratios["converse"], ratios["direct"]
ok_phi = med(c["phi"]) <= 0.95 and mx(c["phi"]) <= 1.10
ok_uc = med(c["u"]) <= 1.10 and mx(c["u"]) <= 1.25
ok_ud = med(d["u"]) <= 1.00 and mx(d["u"]) <= 1.25
ok_sp = (max(dsp) <= 1) if dsp else False
out += ["", f"Converso: mediana phi {med(c['phi']):.2f} (máx {mx(c['phi']):.2f}) → {'cumple' if ok_phi else 'NO cumple'}; mediana u {med(c['u']):.2f} (máx {mx(c['u']):.2f}) → {'cumple' if ok_uc else 'NO cumple'}.",
        f"Directo: mediana u {med(d['u']):.2f} (máx {mx(d['u']):.2f}) → {'cumple' if ok_ud else 'NO cumple'}. Saltos S−E máx {max(dsp) if dsp else 'n/d'} → {'cumple' if ok_sp else 'NO cumple'}.",
        f"Veredicto del proxy: {'S va bien' if (ok_phi and ok_uc and ok_ud and ok_sp) else 'S NO pasa todos los criterios'} (n = {len(c['phi'])} semillas converso, {len(d['u'])} directo; horizonte corto: da signo, no tamaño)."]
txt = "\n".join(out)
Path(sys.argv[2] if len(sys.argv) > 2 else root / "T1_resultados.md").write_text(txt, encoding="utf-8")
print(txt)
