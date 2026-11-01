"""EXP-0030, bloque A2 (añadido tras la revisión del 2026-10-26; ver Desviaciones): comparación dentro de la MISMA GPU (A100-80) con semillas distintas.
No hay pares M/E con la misma semilla y hardware, pero sí E (31, 32) y B (31, 32) en A100-80 junto con M (28-30) en A100-80. Descriptivo; sin p-valores."""
import numpy as np

import common as c

R = c.cargar()


def rob(r, k="l2re_u", lo=18000):
    return float(np.median([h[k] for h in r["json"]["hist"] if h["step"] >= lo]))


out = ["# EXP-0030, bloque A2: misma GPU (A100-80), semillas distintas (exploratorio, post hoc)\n",
       "u robusto = mediana de las evaluaciones de los pasos 18 000-20 000. Converso. Las semillas de M (28-30) y de B/E (31-32) no coinciden: no es un pareo, es un contraste de bandas dentro del mismo hardware.\n",
       "| brazo | hardware | semillas | u robusto por semilla | mín - máx |", "|---|---|---|---|---|"]
band = {}
for arm in ("B", "E", "M"):
    for hw in ("T4", "H100", "A100-80"):
        s = sorted(k[2] for k, v in R.items() if k[0] == "converse" and k[1] == arm and v["hw"] == hw)
        if s:
            vals = [rob(R[("converse", arm, x)]) for x in s]
            band[(arm, hw)] = vals
            out.append(f"| {arm} | {hw} | {s} | {[round(v, 3) for v in vals]} | {min(vals):.3f} - {max(vals):.3f} |")
allE = [v for (a, h), vs in band.items() if a == "E" for v in vs]
allB = [v for (a, h), vs in band.items() if a == "B" for v in vs]
M = band[("M", "A100-80")]
out += ["", f"- E en todo el hardware (n = {len(allE)}): {min(allE):.3f} - {max(allE):.3f}. E solo en A100-80: {min(band[('E','A100-80')]):.3f} - {max(band[('E','A100-80')]):.3f}.",
        f"- B en todo el hardware (n = {len(allB)}): {min(allB):.3f} - {max(allB):.3f}. B solo en A100-80: {min(band[('B','A100-80')]):.3f} - {max(band[('B','A100-80')]):.3f}.",
        f"- M en A100-80 (n = {len(M)}): {min(M):.3f} - {max(M):.3f}. El mínimo de M ({min(M):.3f}) {'supera' if min(M) > max(allE) else 'no supera'} el máximo de E en cualquier hardware ({max(allE):.3f}).",
        "", "Lectura: la banda de E es angosta y parecida en T4, H100 y A100-80; la «pérdida de la ventaja de E sobre B» en las semillas 31-32 viene de que B salió bien en la A100 (0.039, 0.094), no de que E saliera peor. Las tres semillas de M quedan por encima de la banda de E, también en la misma GPU. El hardware no explica esa diferencia, pero tampoco se puede descartar que el efecto de las semillas 28-30 sobre M difiera del de 31-32 sobre E (semillas distintas). Con n = 2 de E en A100-80, es una observación, no una prueba."]
(c.SALIDA / "A2_mismo_hardware.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
