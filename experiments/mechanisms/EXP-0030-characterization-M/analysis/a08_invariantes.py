"""EXP-0030, Q5 (admisibilidad): invariantes integrales de equilibrio de la viga bimorfa, sobre los campos finales (s020000) de B, E y M y sobre el FEM de referencia.
No existe una lista de simetrias enunciada en ALINEACION_PAPER.md; en su lugar se verifican consecuencias exactas del equilibrio y de las condiciones de borde en cada seccion x (integrales en y, trapecio):
  N(x) = int sigma_xx dy   (sin fuerza axial externa: N = 0),  V(x) = int tau_xy dy,  M(x) = int sigma_xx (y - H/2) dy.
Converso (sin carga mecanica externa, extremo libre): N = V = M = 0 en toda seccion. Directo (fuerza transversal F en el extremo): N = 0, V(x) constante igual a la traccion aplicada (t_y H) y M lineal en x.
Ademas: tracciones en las caras libres y/o 0 y y = H (sigma_yy y tau_xy relativos a max|sigma_xx| del FEM). Se reporta, sin umbral de aprobacion, la violacion relativa a la escala max|sigma_xx|·H del FEM."""
import json
import numpy as np

import common

H = 0.001
ref = {}
for p in ("converse", "direct"):
    fl = np.load(common.RES_IN.parent / "piezo" / "reference" / f"fem_flux_{p}.npz")
    ref[p] = {k: fl[k] for k in fl.keys()}


def secciones(xy, vals):
    xs, ys = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    idx = np.lexsort((xy[:, 0], xy[:, 1]))                    # y lento, x rapido
    g = vals[idx].reshape(len(ys), len(xs), -1)
    return xs, ys, g


def invariantes(xy, sxx, syy, tau):
    xs, ys, g = secciones(xy, np.stack([sxx, syy, tau], axis=1))
    N = np.trapz(g[:, :, 0], ys, axis=0)
    V = np.trapz(g[:, :, 2], ys, axis=0)
    M = np.trapz(g[:, :, 0] * (ys[:, None] - H / 2), ys, axis=0)
    return xs, N, V, M, g[0], g[-1]


rows = []
for p in ("converse", "direct"):
    r = ref[p]
    smax = float(np.abs(r["sigma_xx"][r["inside"]]).max())
    xs, Nf, Vf, Mf, bot, top = invariantes(r["xy"], r["sigma_xx"], r["sigma_yy"], r["tau_xy"])
    sc_N, sc_M = smax * H, smax * H * H
    rows.append(dict(problema=p, brazo="FEM", semilla=0, N=float(np.abs(Nf).max() / sc_N), V=float(np.abs(Vf).max() / sc_N), M_rms=float(np.sqrt((Mf ** 2).mean()) / sc_M),
                     Vmedia=float(Vf.mean()), traccion=float(max(np.abs(bot[:, 1:]).max(), np.abs(top[:, 1:]).max()) / smax)))
    R = common.cargar()
    for (pr, arm, sd), info in sorted(R.items()):
        if pr != p:
            continue
        z = np.load(info["npz"], allow_pickle=True)
        xy, vals = z["s020000/fields/xy"], z["s020000/fields/values"].astype(float)
        xs, N, V, M, bot, top = invariantes(xy, vals[:, 3], vals[:, 4], vals[:, 5])
        rows.append(dict(problema=p, brazo=arm, semilla=sd, N=float(np.abs(N).max() / sc_N), V=float(np.abs(V).max() / sc_N), M_rms=float(np.sqrt((M ** 2).mean()) / sc_M),
                         Vmedia=float(V.mean()), traccion=float(max(np.abs(bot[:, 1:]).max(), np.abs(top[:, 1:]).max()) / smax),
                         dV_fem=float(np.sqrt(((V - Vf) ** 2).mean()) / (np.abs(Vf).max() + 1e-300)), dM_fem=float(np.sqrt(((M - Mf) ** 2).mean() / ((Mf ** 2).mean() + 1e-300)))))
json.dump(rows, open(common.SALIDA / "I_invariantes.json", "w"), indent=1)
md = ["# I. Invariantes integrales de equilibrio (post hoc, exploratorio)\n",
      "Seccion a seccion en x, campos finales (s020000), malla 201 x 21. Violacion relativa a la escala del FEM: `N` = max|∫σxx dy| / (max|σxx|·H); `V` igual con τxy; `M` = rms de M(x) / (max|σxx|·H²); `tracciones` = max(|σyy|, |τxy|) en y = 0 y y = H sobre max|σxx|. Converso: N = V = M = 0 exactos; directo: N = 0, V constante, M lineal. Medianas entre semillas; FEM = discretización de la referencia (cota de lo alcanzable).\n",
      "| problema | brazo | n | N | V | M (rms) | tracciones | V medio | error V vs FEM | error M vs FEM |", "|---|---|---|---|---|---|---|---|---|---|"]
for p in ("converse", "direct"):
    for arm in ("FEM", "B", "E", "M"):
        s = [r for r in rows if r["problema"] == p and r["brazo"] == arm]
        if not s:
            continue
        m = lambda k: float(np.median([r[k] for r in s if k in r])) if any(k in r for r in s) else float("nan")
        md.append(f"| {p} | {arm} | {len(s)} | {m('N'):.3g} | {m('V'):.3g} | {m('M_rms'):.3g} | {m('traccion'):.3g} | {m('Vmedia'):.3g} | {m('dV_fem'):.3g} | {m('dM_fem'):.3g} |")
(common.SALIDA / "I_invariantes.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
