"""EXP-0030 (post hoc, exploratorio): J dinamica efectiva (Q2), K uso de las aristas (Q6), L probabilidad de mejora y perfiles de desempeno (Q1).
J: movimiento relativo de los coeficientes efectivos por arista cada 1 000 pasos (`settle/*`, del registro) y paso de separacion de B y E (primera evaluacion con l2re_u(E)/l2re_u(B) < 0.5).
K: energia de salida de cada arista ponderada por la ocupacion de sus bases, sqrt(sum_g occ_g c_g^2), con los coeficientes efectivos de s020000 (P c'); concentracion (fraccion de aristas que lleva el 90 % de la energia, Gini) y similitud
   de las funciones de la capa 0 (correlacion) entre brazos con la misma semilla (mismo modelo inicial).
L: P(X < Y) sobre todos los pares entre brazos (empates 1/2) y fraccion de corridas con error <= tau, para el ultimo paso y la metrica robusta (mediana 18 000-20 000); sin intervalos (n <= 5), descripcion de Agarwal et al. (arXiv:2108.13264, resumen)."""
import numpy as np

import common

R = common.cargar()
SEEDS_C = (28, 29, 30, 31, 32)


def hist(r, key="l2re_u"):
    return {h["step"]: h[key] for h in r["json"]["hist"]}


def robusto(r):
    return float(np.median([h["l2re_u"] for h in r["json"]["hist"] if h["step"] >= 18000]))


# ---------------------------------------------------------------- J
out = ["# J. Dinámica efectiva de los coeficientes (post hoc, exploratorio)\n",
       "`mov` = mediana entre aristas del cambio relativo ‖Δc_ef‖/‖c_ef‖ en cada ventana de 1 000 pasos (las tres capas juntas); `acum4k`/`acum8k` = suma de esas medianas hasta el paso 4 000 y 8 000. `separa` = primera evaluación (cada 500 pasos) con l2re_u(E)/l2re_u(B) < 0.5 en la misma semilla; `salida` = primera con l2re_u < 0.5.\n",
       "| problema | semilla | brazo | hw | acum4k | acum8k | mov 1k-2k | mov 5k-6k | mov 15k-16k | salida | separa (E/B) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for p in ("converse", "direct"):
    for sd in SEEDS_C:
        for arm in ("B", "E", "M"):
            if (p, arm, sd) not in R:
                continue
            r = R[(p, arm, sd)]
            z = np.load(r["npz"], allow_pickle=True)
            mov = {}
            for st in range(1000, 20001, 1000):
                vs = [z[f"settle/step{st:06d}/layer{l}"].reshape(-1) for l in range(3) if f"settle/step{st:06d}/layer{l}" in z]
                if vs:
                    mov[st] = float(np.median(np.concatenate(vs)))
            ac = lambda lim: sum(v for k, v in mov.items() if k <= lim)
            h = hist(r)
            salida = next((s for s in sorted(h) if h[s] < 0.5), None)
            sep = ""
            if arm == "E" and (p, "B", sd) in R:
                hb = hist(R[(p, "B", sd)])
                sep = next((s for s in sorted(h) if s in hb and h[s] / hb[s] < 0.5), None)
            out.append(f"| {p} | {sd} | {arm} | {r['hw']} | {ac(4000):.3f} | {ac(8000):.3f} | {mov.get(2000, float('nan')):.4f} | {mov.get(6000, float('nan')):.4f} | {mov.get(16000, float('nan')):.4f} | {salida} | {sep} |")
(common.SALIDA / "J_dinamica.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out), flush=True)


# ---------------------------------------------------------------- K
def c_eff(z, step, l):
    c = z[f"s{step:06d}/state/layers.{l}.coefficients"].astype(float)
    key = f"s{step:06d}/state/layers.{l}.pre_P"
    if key in z:
        P = z[key].astype(float)
        O, I, G = c.shape
        c = np.einsum("kab,kb->ka", P, c.reshape(O * I, G)).reshape(O, I, G)
    return c


def gini(v):
    v = np.sort(np.abs(v))
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum() + 1e-300))


def funcs0(z):
    c = c_eff(z, 20000, 0)
    cen = z["s020000/state/layers.0.centers"].astype(float)
    w = float(np.ravel(z["s020000/state/layers.0.width"])[0])
    g = np.linspace(-1, 1, 101)
    phi = np.exp(-((g[:, None] - cen[None, :]) / w) ** 2)
    return np.einsum("zg,oig->oiz", phi, c).reshape(-1, len(g))


def corr_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-300)


out = ["# K. Uso de las aristas (post hoc, exploratorio)\n",
       "Estados finales. Importancia de arista = sqrt(Σ_g ocupación_g · c_g²) con coeficientes efectivos. `f90` = fracción de aristas que reúne el 90 % de la energía (menor = más concentrada); `gini` = desigualdad. Mediana entre semillas por brazo y capa.\n",
       "| problema | brazo | n | capa | f90 | gini |", "|---|---|---|---|---|---|"]
for p in ("converse", "direct"):
    for arm in ("B", "E", "M"):
        keys = [k for k in R if k[0] == p and k[1] == arm]
        for l in range(3):
            f9, gi = [], []
            for k in keys:
                z = np.load(R[k]["npz"], allow_pickle=True)
                c = c_eff(z, 20000, l)
                occ = z[f"s020000/occupancy/layer{l}"].astype(float)
                imp = np.sqrt((occ[None, :, :] * c ** 2).sum(-1)).reshape(-1)
                e = np.sort(imp ** 2)[::-1]
                f9.append(float((np.cumsum(e) / e.sum() < 0.9).sum() + 1) / len(e))
                gi.append(gini(imp))
            if keys:
                out.append(f"| {p} | {arm} | {len(keys)} | {l} | {np.median(f9):.2f} | {np.median(gi):.2f} |")
out += ["", "## Similitud de las funciones de la capa 0 (correlación mediana entre aristas; misma semilla = mismo modelo inicial)\n", "| problema | par | semillas | correlación mediana |", "|---|---|---|---|"]
for p in ("converse", "direct"):
    for a, b in (("B", "E"), ("B", "M"), ("E", "M")):
        cs = []
        for sd in SEEDS_C:
            if (p, a, sd) in R and (p, b, sd) in R:
                fa = funcs0(np.load(R[(p, a, sd)]["npz"], allow_pickle=True))
                fb = funcs0(np.load(R[(p, b, sd)]["npz"], allow_pickle=True))
                cs.append(float(np.median(corr_rows(fa, fb))))
        if cs:
            out.append(f"| {p} | {a} vs {b} | {len(cs)} | {np.median(cs):.2f} ({min(cs):.2f}–{max(cs):.2f}) |")
    ref = []
    for arm in ("B", "E"):
        ss = [sd for sd in SEEDS_C if (p, arm, sd) in R]
        for i, s1 in enumerate(ss):
            for s2 in ss[i + 1:]:
                fa = funcs0(np.load(R[(p, arm, s1)]["npz"], allow_pickle=True))
                fb = funcs0(np.load(R[(p, arm, s2)]["npz"], allow_pickle=True))
                ref.append(float(np.median(corr_rows(fa, fb))))
    out.append(f"| {p} | mismo brazo, semillas distintas (referencia) | {len(ref)} | {np.median(ref):.2f} ({min(ref):.2f}–{max(ref):.2f}) |")
(common.SALIDA / "K_aristas.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out), flush=True)


# ---------------------------------------------------------------- L
def valores(p, arm, f):
    return [f(R[(p, arm, sd)]) for sd in SEEDS_C if (p, arm, sd) in R]


def pmejora(x, y):
    return float(np.mean([(a < b) + 0.5 * (a == b) for a in x for b in y]))


out = ["# L. Probabilidad de mejora y perfiles de desempeño (post hoc, exploratorio; sin intervalos)\n",
       "`P(X<Y)` = fracción de pares (una corrida de X, una de Y) donde X tiene menor l2re_u (empates ½); 0.5 = sin diferencia. Con n ≤ 5 solo toma valores en pasos de 1/(n·m): se lee como descripción. Perfiles: fracción de corridas con error ≤ tau.\n",
       "| problema | métrica | comparación | P(X<Y) | n |", "|---|---|---|---|---|"]
for p in ("converse", "direct"):
    for nom, f in (("último paso", lambda r: r["json"]["score_final"]["l2re_u"]), ("robusta (18k-20k)", robusto)):
        for a, b in (("E", "B"), ("M", "B"), ("M", "E")):
            xa, xb = valores(p, a, f), valores(p, b, f)
            if xa and xb:
                out.append(f"| {p} | {nom} | {a} frente a {b} | {pmejora(xa, xb):.2f} | {len(xa)}×{len(xb)} |")
out += ["", "| problema | métrica | brazo | n | ≤0.05 | ≤0.10 | ≤0.20 | ≤0.30 | ≤0.50 |", "|---|---|---|---|---|---|---|---|---|"]
for p in ("converse", "direct"):
    for nom, f in (("último paso", lambda r: r["json"]["score_final"]["l2re_u"]), ("robusta (18k-20k)", robusto)):
        for arm in ("B", "E", "M"):
            v = valores(p, arm, f)
            if v:
                out.append(f"| {p} | {nom} | {arm} | {len(v)} | " + " | ".join(f"{np.mean(np.array(v) <= t):.2f}" for t in (0.05, 0.10, 0.20, 0.30, 0.50)) + " |")
(common.SALIDA / "L_perfiles.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out), flush=True)
