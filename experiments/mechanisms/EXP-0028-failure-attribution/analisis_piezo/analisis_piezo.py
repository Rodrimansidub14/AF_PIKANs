"""Análisis de propiedades de piezo SIN cómputo nuevo (solo datos ya guardados). Salidas en analisis_piezo/.
 1. error por campo y brazo (pareado con B); 2. descomposición de la pérdida por grupos de términos; 3. estabilidad (saltos de pérdida);
 4. curvatura: espectro del Gauss-Newton por arista (de E) y rigidez que F produce con su P fija sobre esa curvatura."""
import csv
import glob
import json
import re
import statistics as st
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
RES = OUT.parent / "results"
SEEDS = (20260728, 20260729, 20260730)
ARMS = ("B", "F", "E", "J")
DIRS = ("piezo-s1a", "piezo-s1b", "piezo-modal-run")      # NO piezo-modal-smoke (300 pasos) ni stage0
J = {}
for dd in DIRS:
    for f in glob.glob(str(RES / dd / "**" / "piezo-*-seed*.json"), recursive=True):
        d = json.load(open(f))
        if d.get("adam_status") == "adam_complete" and d["adam_steps"] == 20000 and d["arm"] in ARMS and d["seed"] in SEEDS:
            J[(d["problem"], d["arm"], d["seed"])] = d
assert len(J) == 24, len(J)


def art(prob, arm, seed):
    fs = [f for dd in DIRS for f in glob.glob(str(RES / dd / "**" / f"piezo-{prob}-{arm}-seed{seed}-artifacts.npz"), recursive=True)]
    assert len(fs) == 1, (prob, arm, seed, fs)
    return np.load(fs[0], allow_pickle=True)


def med_ratio(prob, arm, getter):
    r = []
    for s in SEEDS:
        a, b = J.get((prob, arm, s)), J.get((prob, "B", s))
        if a and b:
            x, y = getter(a), getter(b)
            if x is not None and y:
                r.append(x / y)
    return (st.median(r), r) if r else (None, [])


lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    print(s)



# ---------------------------------------------------------------- 0. errores absolutos
P("## 0. Error absoluto l2re_u al paso 20 000 por semilla (la razón con B engaña cuando B varía mucho)\n")
P("| problema | brazo | semillas 28 / 29 / 30 | media | CV (desv/media) | referencia del paper (MLP) |\n|---|---|---|---|---|---|")
for prob in ("converse", "direct"):
    for arm in ARMS:
        v = [J[(prob, arm, s)]["score_final"]["l2re_u"] for s in SEEDS]
        P(f"| {prob} | {arm} | {' / '.join(f'{x:.3f}' for x in v)} | {np.mean(v):.3f} | {np.std(v, ddof=1) / np.mean(v):.2f} | {'0.107' if prob == 'converse' else '0.117'} |")
P("")

# ---------------------------------------------------------------- 1. campos
P("## 1. Error por campo (razón brazo/B, mediana de 3 semillas; por semilla entre paréntesis)\n")
fields = [("l2re_u", lambda d: d["score_final"]["l2re_u"]), ("l2re_v", lambda d: d["score_final"]["l2re_v"]), ("l2re_phi", lambda d: d["score_final"]["l2re_phi"]),
          ("sigma_xx nrms %", lambda d: d["score_flux"]["sigma_xx"]["nrms_pct"] if d.get("score_flux") else None),
          ("D_y nrms %", lambda d: d["score_flux"]["D_y"]["nrms_pct"] if d.get("score_flux") else None),
          ("pérdida final", lambda d: d["loss_end"])]
for prob in ("converse", "direct"):
    P(f"**{prob}**\n\n| campo | F | E | J |\n|---|---|---|---|")
    for name, g in fields:
        cells = []
        for arm in ("F", "E", "J"):
            m, r = med_ratio(prob, arm, g)
            cells.append("–" if m is None else f"**{m:.2f}** ({', '.join(f'{x:.2f}' for x in r)})")
        P(f"| {name} | " + " | ".join(cells) + " |")
    P("")

# ---------------------------------------------------------------- 2. términos
P("## 2. Dónde está la pérdida (parte de la pérdida de B al final de Adam, y razón brazo/B por grupo)\n")
groups = {"constitutivas σ": ("constitutive_sigma_xx", "constitutive_sigma_yy", "constitutive_tau_xy"), "constitutivas D": ("constitutive_D_x", "constitutive_D_y"),
          "equilibrio y Gauss": ("equilibrium_x", "equilibrium_y", "gauss"),
          "bordes mecánicos": ("bc_top_sigma_yy", "bc_top_tau_xy", "bc_bottom_sigma_yy", "bc_bottom_tau_xy", "bc_right_traction_x", "bc_right_traction_y"),
          "bordes eléctricos": ("bc_right_Dn", "bc_left_Dn", "bc_top_equipotential", "bc_top_zero_net_charge")}


def gsum(d, keys):
    return sum(d["terms_end"].get(k, 0.0) for k in keys)


for prob in ("converse", "direct"):
    P(f"**{prob}**\n\n| grupo | parte de la pérdida de B | F/B | E/B | J/B |\n|---|---|---|---|---|")
    for gname, keys in groups.items():
        shares = [gsum(J[(prob, "B", s)], keys) / J[(prob, "B", s)]["loss_end"] for s in SEEDS if (prob, "B", s) in J]
        cells = []
        for arm in ("F", "E", "J"):
            m, r = med_ratio(prob, arm, lambda d, keys=keys: gsum(d, keys))
            cells.append("–" if m is None else f"{m:.2f}")
        P(f"| {gname} | {100 * st.median(shares):.0f} % | " + " | ".join(cells) + " |")
    P("")

# ---------------------------------------------------------------- 3. estabilidad
P("## 3. Estabilidad: saltos de pérdida (> 5× el mínimo previo, tras el paso 2 000), una cifra por semilla\n")
P("| problema | B | F | E | J |\n|---|---|---|---|---|")
for prob in ("converse", "direct"):
    cells = []
    for arm in ARMS:
        ev = []
        for s in SEEDS:
            z = art(prob, arm, s)
            if z is None:
                continue
            hs, hl = z["hist/step"], z["hist/loss"]
            m = hl[hs >= 2000]
            ev.append(int((m > 5 * np.minimum.accumulate(m)).sum()))
        cells.append(", ".join(map(str, ev)))
    P(f"| {prob} | " + " | ".join(cells) + " |")
P("")

# ---------------------------------------------------------------- 4. curvatura
P("## 4. Curvatura por arista y lo que F hace sobre ella\n")
zF = art("direct", "F", SEEDS[0])
PF = {l: zF[f"s000000/state/layers.{l}.pre_P"].astype(float) for l in range(3)}      # P fija de F: igual en todas las aristas
EPS = 1e-3


def eig_desc(M):
    w = np.linalg.eigvalsh(M)
    return w[..., ::-1]


rows = []
P("Método: E guarda, en cada recálculo, la traza del bloque Gauss-Newton por arista y P = (B/m + 1e-3·I)^(-1/2), con m = traza/11. Con eso, B_e = m·(P⁻² − 1e-3·I). "
  "Con esa B_e se calcula la rigidez que produce F (λmax de P_F·B_e·P_F), frente a la de B (λmax de B_e) y a la de E (λmax de P_E·B_e·P_E). q = λmax(precondicionada)/λmax(B_e): "
  "q > 1 significa que el precondicionador *agrava* la dirección más rígida. Estados a los pasos 2 000, 10 000 y 20 000 (P de E del último recálculo anterior).\n")
P("| problema | paso | dispersión de trazas (log10 p95/p5) | λmax(B_e), log10 [p5, mediana, p95] | aristas con q_F > 1 / > 10 / > 100 | q_F mediana [p5, p95] | q_E mediana [p5, p95] |\n|---|---|---|---|---|---|---|")
for prob in ("converse", "direct"):
    for step in (2000, 10000, 20000):
        tr_all, lmax_B, qF, qE = [], [], [], []
        for s in SEEDS:
            z = art(prob, "E", s)
            if z is None:
                continue
            ks = sorted(int(re.match(r"refresh(\d+)/step", k).group(1)) for k in z.keys() if re.match(r"refresh\d+/step", k))
            steps = {i: int(z[f"refresh{i:02d}/step"]) for i in ks}
            i_last = max(i for i in ks if steps[i] <= step)
            for l in range(3):
                Pm = z[f"s{step:06d}/state/layers.{l}.pre_P"].astype(float)
                tr = z[f"refresh{i_last:02d}/layer{l}/trace"].astype(float)
                if tr.shape[0] != Pm.shape[0]:
                    continue
                w, V = np.linalg.eigh(Pm)
                A = (V * (w ** -2.0)[:, None, :]) @ np.transpose(V, (0, 2, 1)) - EPS * np.eye(Pm.shape[-1])
                Be = (tr / Pm.shape[-1])[:, None, None] * A
                lB = eig_desc(Be)[:, 0]
                lF = eig_desc(PF[l] @ Be @ PF[l])[:, 0]
                lE = eig_desc(Pm @ Be @ Pm)[:, 0]
                tr_all.append(tr)
                lmax_B.append(lB)
                qF.append(lF / lB)
                qE.append(lE / lB)
        tr = np.concatenate(tr_all)
        lB = np.concatenate(lmax_B)
        qf = np.concatenate(qF)
        qe = np.concatenate(qE)
        ok = tr > 0
        disp = np.log10(np.percentile(tr[ok], 95) / np.percentile(tr[ok], 5))
        p = np.percentile
        P(f"| {prob} | {step} | {disp:.2f} | {np.log10(p(lB, 5)):.1f}, {np.log10(np.median(lB)):.1f}, {np.log10(p(lB, 95)):.1f} | "
          f"{100 * np.mean(qf > 1):.0f} % / {100 * np.mean(qf > 10):.0f} % / {100 * np.mean(qf > 100):.0f} % | {np.median(qf):.2f} [{p(qf, 5):.2f}, {p(qf, 95):.2f}] | "
          f"{np.median(qe):.3f} [{p(qe, 5):.3f}, {p(qe, 95):.3f}] |")
        rows.append(dict(problem=prob, step=step, dispersion=float(disp), qF_median=float(np.median(qf)), qF_p95=float(p(qf, 95)), qF_gt10=float(np.mean(qf > 10)), qE_median=float(np.median(qe))))
P("")
(OUT / "analisis_piezo.md").write_text("\n".join(lines), encoding="utf-8")
with open(OUT / "curvatura_F.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---------------------------------------------------------------- 5. condicionamiento dentro de cada arista (regularizado como lo hace E)
P("## 5. Condicionamiento dentro de cada arista, regularizado igual que en E (cond de B/m + 1e-3·I; sin la regularización los bloques casi singulares dan cifras sin sentido)\n")
P("E hace cond = 1 por construcción (P = (B/m + 1e-3·I)^(-1/2)); lo informativo es cuánto reduce F, con su P fija, el condicionamiento real de cada bloque. Mediana [p5, p95] sobre aristas y semillas.\n")
P("| problema | paso | cond de B/m + 1e-3·I | cond tras F | reducción mediana (cond sin / cond con F) | aristas donde F empeora el cond |\n|---|---|---|---|---|---|")
rows5 = []
for prob in ("converse", "direct"):
    for step in (2000, 10000, 20000):
        cB, cF = [], []
        for s in SEEDS:
            z = art(prob, "E", s)
            for l in range(3):
                Pm = z[f"s{step:06d}/state/layers.{l}.pre_P"].astype(float)
                w, V = np.linalg.eigh(Pm)
                Areg = (V * (w ** -2.0)[:, None, :]) @ np.transpose(V, (0, 2, 1))          # = B/m + 1e-3 I
                def cond(M):
                    e = np.linalg.eigvalsh(M)
                    return e[:, -1] / e[:, 0]
                cB.append(cond(Areg))
                cF.append(cond(PF[l] @ Areg @ PF[l]))
        cB, cF = np.concatenate(cB), np.concatenate(cF)
        q = lambda x: f"{np.median(x):.1e} [{np.percentile(x, 5):.1e}, {np.percentile(x, 95):.1e}]"
        red = np.median(cB / cF)
        P(f"| {prob} | {step} | {q(cB)} | {q(cF)} | {red:.1f}x | {100 * np.mean(cF > cB):.0f} % |")
        rows5.append(dict(problem=prob, step=step, condB=float(np.median(cB)), condF=float(np.median(cF)), reduccion=float(red), F_empeora=float(np.mean(cF > cB))))
P("")
(OUT / "analisis_piezo.md").write_text("\n".join(lines), encoding="utf-8")
with open(OUT / "condicionamiento.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows5[0].keys()))
    w.writeheader()
    w.writerows(rows5)
