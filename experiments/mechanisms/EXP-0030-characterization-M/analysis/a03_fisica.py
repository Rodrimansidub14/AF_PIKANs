"""EXP-0030, bloque D (física): términos de pérdida del pool de entrenamiento, residuales por ecuación sobre la malla reservada, errores por campo, flujo, interfaz y chequeos de admisibilidad.
Incluye la prueba de cordura (recalcular terms_end desde un estado guardado). Exploratorio, post hoc. No se fijan umbrales."""
import sys
from pathlib import Path

import numpy as np
import torch

from common import EXP, SALIDA, cargar, md_tabla

sys.path.insert(0, str(EXP.parent / "EXP-0028-failure-attribution" / "piezo"))
import run_piezo as rp  # noqa: E402

R = cargar()
out = ["# EXP-0030, bloque D: física (exploratorio, post hoc; sin umbrales)\n"]


def razon_mediana(valores_num, valores_den):
    v = [a / b for a, b in zip(valores_num, valores_den) if b > 0]
    return (float(np.median(v)), float(min(v)), float(max(v))) if v else (None, None, None)


def semillas(pr, a):
    return sorted(s for (p, b, s) in R if p == pr and b == a)


# ---------------------------------------------------------------- D1: términos de pérdida (pool de entrenamiento)
def terminos_finales(r):
    z = np.load(r["npz"], allow_pickle=True)
    st = z["hist/step"]
    sel = st > 19000
    return {k.split("hist/term/")[1]: float(np.median(z[k][sel])) for k in z.keys() if k.startswith("hist/term/")}


out.append("## D1. Términos de pérdida sobre el *pool de entrenamiento* (con peso; mediana de los pasos > 19 000)\n")
out.append("Razón por término: mediana entre semillas (rango). **No es el residual sobre la malla reservada** (ver D2). Hardware de M distinto al de E: la razón M/E está confundida con el hardware.\n")
for pr in ("converse", "direct"):
    T = {(a, s): terminos_finales(R[(pr, a, s)]) for a in ("B", "E", "M") for s in semillas(pr, a)}
    nombres = list(T[("E", semillas(pr, "E")[0])].keys())
    rows = []
    for nm in nombres:
        fila = [nm]
        for num, den in (("E", "B"), ("M", "E")):
            ss = [s for s in semillas(pr, num) if s in semillas(pr, den)]
            m, lo, hi = razon_mediana([T[(num, s)][nm] for s in ss], [T[(den, s)][nm] for s in ss])
            fila.append("0 en todas las corridas" if m is None else f"{m:.2f} ({lo:.2f}-{hi:.2f})")
        rows.append(fila)
    out.append(f"### {pr}\n")
    out.append(md_tabla(["término", "E/B", "M/E"], rows))
    out.append("")

# ---------------------------------------------------------------- D2: residual por ecuación en la malla reservada (guardado) + prueba de cordura
out.append("## D2. Residual por ecuación sobre la malla reservada (RMS del residual guardado en el estado final, `resid_grid`)\n")
for pr in ("converse", "direct"):
    G = {}
    for a in ("B", "E", "M"):
        for s in semillas(pr, a):
            z = np.load(R[(pr, a, s)]["npz"], allow_pickle=True)
            G[(a, s)] = {k.split("resid_grid/")[1]: float(np.sqrt(np.mean(z[k].astype(float) ** 2))) for k in z.keys() if k.startswith("s020000/resid_grid/")}
    nombres = list(G[("E", semillas(pr, "E")[0])].keys())
    rows = []
    for nm in nombres:
        fila = [nm]
        for num, den in (("E", "B"), ("M", "E")):
            ss = [s for s in semillas(pr, num) if s in semillas(pr, den)]
            m, lo, hi = razon_mediana([G[(num, s)][nm] for s in ss], [G[(den, s)][nm] for s in ss])
            fila.append("0 en todas las corridas" if m is None else f"{m:.2f} ({lo:.2f}-{hi:.2f})")
        rows.append(fila)
    out.append(f"### {pr}\n")
    out.append(md_tabla(["ecuación o borde", "E/B", "M/E"], rows))
    out.append("")

out.append("### Prueba de cordura: recalcular `terms_end` desde el estado guardado (CPU, float64)\n")
torch.set_num_threads(4)
dev, dt = torch.device("cpu"), torch.float64
rows = []
for key in (("converse", "M", 28), ("direct", "E", 29)):
    pr, a, s = key
    r = R[key]
    prob = rp.make_problem(pr, 0)
    m = rp.build(prob, "B", 20260700 + s, dt).to(dev)
    if a != "B":
        rp.gp.GramPreconditioner(a, every=10 ** 9, n_points=64, chunk=2).attach(m)
    z = np.load(r["npz"], allow_pickle=True)
    sd = {k.split("/state/")[1]: torch.tensor(z[k]) for k in z.keys() if k.startswith("s020000/state/")}
    ld = m.load_state_dict(sd, strict=False)
    ok = not ld.missing_keys and not ld.unexpected_keys
    P = prob.pools(20260700 + s, dev, dt, 99)
    loss, terms = prob.loss(m, P)
    te = r["json"]["terms_end"]
    dif = max(abs(float(terms[k].detach()) - te[k]) / max(abs(te[k]), 1e-9) for k in te)       # términos nulos (bordes duros) se comparan en absoluto
    rows.append([pr[:4], a, s, "sí" if ok else "NO", f"{float(loss.detach()):.6g}", f"{r['json']['loss_end']:.6g}", f"{dif:.2e}"])
out.append(md_tabla(["problema", "brazo", "semilla", "estado carga sin claves de más o de menos", "pérdida recalculada", "pérdida guardada (loss_end)", "máx. diferencia relativa por término"], rows))

# ---------------------------------------------------------------- D3: error por campo, flujo, interfaz
out.append("\n## D3. Error por campo, flujo e interfaz (valores del JSON; mediana entre semillas)\n")
rows = []
for pr in ("converse", "direct"):
    for a in ("B", "E", "M"):
        ss = semillas(pr, a)
        sf = [R[(pr, a, s)]["json"]["score_final"] for s in ss]
        fx = [R[(pr, a, s)]["json"]["score_flux"] for s in ss]
        rows.append([pr[:4], a, len(ss), f"{np.median([x['l2re_u'] for x in sf]):.3f}", f"{np.median([x['l2re_v'] for x in sf]):.3f}", f"{np.median([x['l2re_phi'] for x in sf]):.3f}",
                     " / ".join(f"{np.median([f[k]['rel_l2'] for f in fx]):.2f}" for k in ("sigma_xx", "sigma_yy", "tau_xy", "D_x", "D_y")),
                     f"{np.median([R[(pr, a, s)]['json']['interface_gap'] for s in ss]):.2g}"])
out.append(md_tabla(["problema", "brazo", "n", "l2re u", "l2re v", "l2re φ", "flujo rel. L2: σxx / σyy / τ / Dx / Dy", "interface_gap"], rows))

# ---------------------------------------------------------------- D4: admisibilidad
out.append("\n## D4. Chequeos de admisibilidad (valores, sin umbral)\n")
out.append("Consistencia constitutiva = error relativo L2 entre las tensiones y D que predice la red y las reconstruidas desde los gradientes de u, v y φ, sobre la malla reservada (5 magnitudes, en el orden que guarda `consistency_grid/rel_l2`). Carga neta: término `bc_top_zero_net_charge` del pool (directo). Trazas de borde: media de los términos `bc_*` del pool. Simetrías de geometría y polarización: **no se calcularon**; la lista no se cerró (pendiente de la spec).\n")
rows = []
for pr in ("converse", "direct"):
    for a in ("B", "E", "M"):
        cs, bcs, nq = [], [], []
        for s in semillas(pr, a):
            z = np.load(R[(pr, a, s)]["npz"], allow_pickle=True)
            cs.append(z["s020000/consistency_grid/rel_l2"].astype(float))
            t = terminos_finales(R[(pr, a, s)])
            bcs.append(np.mean([v for k, v in t.items() if k.startswith("bc_") and k != "bc_top_zero_net_charge"]))
            nq.append(t.get("bc_top_zero_net_charge", float("nan")))
        rows.append([pr[:4], a, len(cs), " / ".join(f"{x:.2f}" for x in np.median(cs, axis=0)), f"{np.median(bcs):.2e}", (f"{np.nanmedian(nq):.2e}" if pr == "direct" else "n/a")])
out.append(md_tabla(["problema", "brazo", "n", "consistencia constitutiva (5 magnitudes)", "media de términos de borde", "carga neta cero"], rows))
(SALIDA / "D_fisica.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
