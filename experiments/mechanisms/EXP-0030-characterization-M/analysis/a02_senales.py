"""EXP-0030, bloque C (señales registradas en cada recálculo) y relación por arista dentro de una corrida (solo descripción). Exploratorio, post hoc."""
import json

import numpy as np

from common import SALIDA, cargar, md_tabla

R = cargar()
out = ["# EXP-0030, bloque C: señales por recálculo (exploratorio, post hoc)\n",
       "Cuotas = traza del bloque GN por grupo físico sumada sobre todas las aristas y capas, entre la traza total. `cond p90` = percentil 90 de la condición del bloque normalizado (B/m + 1e-3 I) sobre aristas vivas, mediana de las capas. Solo M registra cuotas (`gtrace_*`); E y B antiguos no.\n"]
PASOS = (0, 500, 3000, 8000, 18000)
rows, rowsb, rowsp = [], [], []
for (pr, a, s), r in sorted(R.items()):
    if a != "M":
        continue
    z = np.load(r["npz"], allow_pickle=True)
    n = sorted({int(k[7:9]) for k in z.keys() if k.startswith("refresh") and k.endswith("/step")})
    steps = {i: int(z[f"refresh{i:02d}/step"]) for i in n}
    for objetivo in PASOS:
        i = min(steps, key=lambda j: abs(steps[j] - objetivo))
        L = sorted({int(k.split("/layer")[1][0]) for k in z.keys() if k.startswith(f"refresh{i:02d}/layer")})
        g = {nm: sum(float(z[f"refresh{i:02d}/layer{l}/gtrace_{nm}"].astype(float).sum()) for l in L) for nm in ("mec", "elec", "equil", "borde")}
        tot = sum(g.values())
        cond = [np.percentile(z[f"refresh{i:02d}/layer{l}/cond"].astype(float), 90) for l in L]
        rows.append([pr[:4], s, steps[i]] + [f"{g[nm] / tot:.3f}" for nm in ("mec", "elec", "equil", "borde")] + [f"{np.median(cond):.3g}"])
    pd_ = r["json"]["pre_diag"]
    bs = [d["b_mix"] for d in pd_]
    vals, cnt = np.unique(bs, return_counts=True)
    ev = r["json"]["pre_events"]
    rowsb.append([pr[:4], s, " ".join(f"{v}:{c}" for v, c in zip(vals, cnt)), len(bs), len(ev), (f"{ev[0]['step']} / {int(np.median([e['step'] for e in ev]))}" if ev else "-"),
                  f"{r['json']['pre_seconds'] / max(1, r['json']['pre_recomputes']):.1f}"])
out.append("## Cuotas por grupo y condición en recálculos escogidos (M)\n")
out.append(md_tabla(["problema", "semilla", "paso del recálculo", "mecánica", "eléctrica", "equilibrio", "borde", "cond p90"], rows))
out.append("\n## b elegido por P2, eventos de P4 en sombra y costo del recálculo (M)\n")
out.append("`b:conteo` = veces que se eligió cada b en los recálculos. Eventos de P4 = pérdida de un paso > 5× el mínimo de los 200 previos (solo observa). Primer / mediano = pasos de esos eventos.\n")
out.append(md_tabla(["problema", "semilla", "b:conteo", "recálculos", "eventos P4", "primer / mediano paso", "s por recálculo"], rowsb))

# ---- condición y traza por recálculo en E (sin cuotas) y M, para comparar trayectorias
out.append("\n## Condición del bloque (p90, mediana de capas) a lo largo del entrenamiento: E y M, converso\n")
rows = []
for (pr, a, s), r in sorted(R.items()):
    if pr != "converse" or a not in ("E", "M"):
        continue
    z = np.load(r["npz"], allow_pickle=True)
    n = sorted({int(k[7:9]) for k in z.keys() if k.startswith("refresh") and k.endswith("/step")})
    pts = []
    for objetivo in (0, 1000, 5000, 10000, 19000):
        i = min(n, key=lambda j: abs(int(z[f"refresh{j:02d}/step"]) - objetivo))
        L = sorted({int(k.split("/layer")[1][0]) for k in z.keys() if k.startswith(f"refresh{i:02d}/layer") and k.endswith("/cond")})
        pts.append(f"{np.median([np.percentile(z[f'refresh{i:02d}/layer{l}/cond'].astype(float), 90) for l in L]):.3g}")
    rows.append([a, s, r["hw"]] + pts)
out.append(md_tabla(["brazo", "semilla", "hardware", "paso 0", "paso 1000", "paso 5000", "paso 10000", "paso 19000"], rows))

# ---- por arista dentro de una corrida (descriptivo)
out.append("\n## Por arista, dentro de una corrida (descriptivo; n = aristas, no semillas)\n")
rows = []
for key in (("converse", "M", 28), ("direct", "M", 29)):
    r = R[key]
    z = np.load(r["npz"], allow_pickle=True)
    n = max(int(k[7:9]) for k in z.keys() if k.startswith("refresh") and k.endswith("/step"))
    for l in (0, 1):
        tr = z[f"refresh{n:02d}/layer{l}/trace"].astype(float).ravel()
        cd = z[f"refresh{n:02d}/layer{l}/cond"].astype(float).ravel()
        el = z[f"refresh{n:02d}/layer{l}/gtrace_elec"].astype(float).ravel()
        vivo = tr > 1e-12 * tr.max()

        def rho(x, y):
            from scipy.stats import spearmanr
            return spearmanr(x[vivo], y[vivo])[0]
        rows.append([key[0][:4], key[2], f"capa {l}", int(vivo.sum()), f"{rho(tr, cd):+.2f}", f"{rho(el / np.maximum(tr, 1e-300), cd):+.2f}"])
out.append(md_tabla(["problema", "semilla (M)", "capa", "aristas vivas", "Spearman traza-cond", "Spearman cuota eléctrica-cond"], rows))
(SALIDA / "C_senales.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
