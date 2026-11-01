"""Análisis adicionales de piezo sin cómputo nuevo (hipótesis sobre lo que el mecanismo no resuelve). Lee artefactos ya guardados."""
import glob, json
import numpy as np
RES = "results"; DIRS = ("piezo-s1a", "piezo-s1b", "piezo-modal-run"); SEEDS = (20260728, 20260729, 20260730)
def art(prob, arm, seed):
    for dd in DIRS:
        for f in glob.glob(f"{RES}/{dd}/**/piezo-{prob}-{arm}-seed{seed}-artifacts.npz", recursive=True):
            return np.load(f, allow_pickle=True)
def js(prob, arm, seed):
    for dd in DIRS:
        for f in glob.glob(f"{RES}/{dd}/**/piezo-{prob}-{arm}-seed{seed}.json", recursive=True):
            d = json.load(open(f))
            if d["adam_steps"] == 20000: return d
print("## A. Traza del bloque GN de la capa de salida por canal (E; mediana entre semillas; refresco 0 y 19). Canales: 0 u, 1 v, 2 phi, 3 sigma_xx, 4 sigma_yy, 5 tau_xy, 6 D_x, 7 D_y")
names = ["u", "v", "phi", "sxx", "syy", "txy", "Dx", "Dy"]
for prob in ("converse", "direct"):
    for r in ("00", "19"):
        tr = []
        for s in SEEDS:
            z = art(prob, "E", s); t = z[f"refresh{r}/layer2/trace"].astype(float)       # (O*I,) con O=8 salidas
            I = t.shape[0] // 8; tr.append(t.reshape(8, I).sum(1))
        m = np.median(np.array(tr), axis=0); m = m / m.sum()
        print(f"{prob} refresco {r}: " + "  ".join(f"{n}={100*x:.1f}%" for n, x in zip(names, m)))
print("\n## B. Salida de la solución nula: primer paso con l2re_u < 0.5 (hist cada 500 pasos) y error final")
for prob in ("converse", "direct"):
    for arm in ("B", "E", "J", "F"):
        row = []
        for s in SEEDS:
            d = js(prob, arm, s); h = d["hist"]
            esc = next((x["step"] for x in h if x["l2re_u"] < 0.5), None)
            row.append(f"{esc}/{d['score_final']['l2re_u']:.3f}")
        print(f"{prob} {arm}: escape/final por semilla: " + "  ".join(row))
print("\n## C. Consistencia constitutiva (discrepancia relativa entre tensiones predichas y reconstruidas desde du, dv, dphi; 5 componentes sxx, syy, txy, Dx, Dy; mediana entre semillas) al paso 20 000")
for prob in ("converse", "direct"):
    for arm in ("B", "F", "E", "J"):
        v = np.median(np.array([art(prob, arm, s)["s020000/consistency_grid/rel_l2"] for s in SEEDS]), axis=0)
        print(f"{prob} {arm}: " + "  ".join(f"{x:.2f}" for x in v))
