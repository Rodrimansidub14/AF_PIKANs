"""EXP-0018: tasa de falso positivo y potencia de las reglas del borrador, por simulacion, con el ruido pareado MEDIDO.

Ruido: desviacion tipica del log de la razon entre puntos finales de L-BFGS desde estados cercanos de la etapa A del EXP-0016 (k = 0.6 y 0.8 frente a 1.00):
Burgers 0.71 (rango 0.44-0.97 segun k), Allen-Cahn 0.38. Es una COTA SUPERIOR del ruido de un cambio de precondicionador desde el MISMO estado (aqui los estados
difieren); el ruido desde el mismo estado no se ha medido.

Reglas (borrador v1): H-CEIL-1 para un brazo (P o H): en ambos problemas, mediana de r <= T1 y r < 1 en >= 4 de 5 semillas. H-CEIL-2 para el brazo que paso:
frente a su control (N para P, N-H para H): mediana de la razon <= T2 y mejora en >= 4 de 5, en ambos problemas. "Techo sostenido" = H-CEIL-1 y H-CEIL-2.
Modelo nulo: log r de cada brazo ~ Normal(0, sigma), independiente entre brazos y semillas. Alternativa: el brazo informativo (P y H) tiene log r ~ Normal(log r_true, sigma),
los controles (N, N-H) ~ Normal(0, sigma).
"""
import json
import math

import numpy as np

rng = np.random.default_rng(20261020)
SIGMA = {"burgers": 0.71, "ac": 0.38}
N = {"burgers": 5, "ac": 5}
SIMS = 40000


def arm_pass(lr, T, need=4):
    r = np.exp(lr)
    return np.median(r) <= T and (r < 1).sum() >= need


def one_sim(T1, T2, r_true, sigma_scale=1.0):
    ok1 = {"P": True, "H": True}
    ok2 = {"P": True, "H": True}
    for pr in ("burgers", "ac"):
        s, n = SIGMA[pr] * sigma_scale, N[pr]
        eff = math.log(r_true)
        lr = {"P": rng.normal(eff, s, n), "H": rng.normal(eff, s, n), "N": rng.normal(0, s, n), "NH": rng.normal(0, s, n)}
        for a, c in (("P", "N"), ("H", "NH")):
            ok1[a] = ok1[a] and arm_pass(lr[a], T1)
            ok2[a] = ok2[a] and arm_pass(lr[a] - lr[c], T2)
    return any(ok1[a] and ok2[a] for a in ("P", "H")), any(ok1.values())


def rate(T1, T2, r_true, scale=1.0):
    res = np.array([one_sim(T1, T2, r_true, scale) for _ in range(SIMS)])
    return res[:, 0].mean(), res[:, 1].mean()


out = {}
print("T1 / T2 | falso positivo (techo sostenido) | solo H-CEIL-1 | potencia si r_verdadero=0.6 | 0.7 | 0.8   (sigma medido; con sigma/2 entre parentesis para r=0.7)")
for T1, T2 in ((0.85, 0.90), (0.80, 0.85), (0.75, 0.85), (0.70, 0.80)):
    fp, fp1 = rate(T1, T2, 1.0)
    pw = [rate(T1, T2, r)[0] for r in (0.6, 0.7, 0.8)]
    pw_half = rate(T1, T2, 0.7, 0.5)[0]
    out[f"{T1}/{T2}"] = dict(false_positive=fp, false_positive_h1_only=fp1, power_r06=pw[0], power_r07=pw[1], power_r08=pw[2], power_r07_half_sigma=pw_half)
    print(f"{T1:.2f}/{T2:.2f} | {fp:.3f} | {fp1:.3f} | {pw[0]:.2f} | {pw[1]:.2f} | {pw[2]:.2f}   ({pw_half:.2f})", flush=True)
json.dump(out, open(__file__.replace("calibrate_thresholds.py", "calibration.json"), "w"), indent=1)
