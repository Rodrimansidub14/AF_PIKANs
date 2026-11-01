"""EXP-0030, bloques A (efecto), B (estabilidad frente a las semillas) y E (sensibilidad retrospectiva). Sin p-valores. Exploratorio y post hoc."""
import itertools
import json

import numpy as np

from common import SALIDA, cargar, final, md_tabla, robusto, salida_llano

R = cargar()
filas, out = [], []
for (pr, a, s), r in sorted(R.items()):
    filas.append(dict(problema=pr, brazo=a, semilla=s, hw=r["hw"], u_final=final(r), phi_final=final(r, "l2re_phi"), u_rob=robusto(r), phi_rob=robusto(r, "l2re_phi"), salida=salida_llano(r),
                      s_por_paso=r["json"]["adam_seconds"] / 20000))
json.dump(filas, open(SALIDA / "A_corridas.json", "w"), indent=1)
out.append("# EXP-0030, bloques A, B y E (exploratorio, post hoc; sin p-valores)\n")
out.append("## A. Efecto por corrida\n")
out.append("`u_rob` y `phi_rob` = mediana de las evaluaciones de los pasos 18 000-20 000 (post hoc). `salida` = primer paso evaluado con l2re_u < 0.5. Hardware siempre visible.\n")
out.append(md_tabla(["problema", "brazo", "semilla", "hardware", "u final", "u rob.", "φ final", "φ rob.", "salida del llano", "s/paso"],
                    [[f["problema"][:4], f["brazo"], f["semilla"], f["hw"], f"{f['u_final']:.3f}", f"{f['u_rob']:.3f}", f"{f['phi_final']:.3f}", f"{f['phi_rob']:.3f}", f"{f['salida']:.0f}", f"{f['s_por_paso']:.3f}"] for f in filas]))

out.append("\n## A. Cola por brazo (mínimo / mediana / máximo)\n")
rows = []
for pr in ("converse", "direct"):
    for a in ("B", "E", "M"):
        sel = [f for f in filas if f["problema"] == pr and f["brazo"] == a]
        if not sel:
            continue
        for k, nom in (("u_final", "u final"), ("u_rob", "u robusto"), ("phi_final", "φ final"), ("salida", "salida del llano")):
            v = [f[k] for f in sel]
            rows.append([pr[:4], a, len(sel), nom, f"{min(v):.3g}", f"{np.median(v):.3g}", f"{max(v):.3g}"])
out.append(md_tabla(["problema", "brazo", "n", "métrica", "mín.", "mediana", "máx."], rows))


def idx(pr):
    return {(f["brazo"], f["semilla"]): f for f in filas if f["problema"] == pr}


out.append("\n## A. Razones pareadas por semilla (valores individuales, mediana y rango)\n")
rows = []
for pr in ("converse", "direct"):
    I = idx(pr)
    for num, den in (("E", "B"), ("M", "E"), ("M", "B")):
        for k, nom in (("u_final", "u final"), ("u_rob", "u robusto"), ("phi_final", "φ final")):
            ps = [(s, I[(num, s)][k] / I[(den, s)][k]) for s in range(28, 33) if (num, s) in I and (den, s) in I]
            if ps:
                v = [x for _, x in ps]
                hw = "mismo hardware" if all(I[(num, s)]["hw"] == I[(den, s)]["hw"] for s, _ in ps) else "hardware distinto en algún par"
                rows.append([pr[:4], f"{num}/{den}", nom, len(ps), ", ".join(f"{s}:{x:.2f}" for s, x in ps), f"{np.median(v):.2f}", f"{min(v):.2f}-{max(v):.2f}", hw])
out.append(md_tabla(["problema", "razón", "métrica", "n", "por semilla", "mediana", "rango", "hardware del par"], rows))

# ------------------------------------------------------------- B. estabilidad frente a las semillas
out.append("\n## B. Estabilidad de la conclusión frente a las semillas (conteo combinatorio, no una prueba)\n")
out.append("Para cada subconjunto de semillas: victorias del numerador (razón < 1) y mediana de la razón. E/B en el converso (n = 5): subconjuntos de 4 y de 3. M/E y M/B en las 3 semillas de M: se deja una fuera.\n")


def estab(I, num, den, semillas, tam, k):
    res = []
    for sub in itertools.combinations(semillas, tam):
        v = [I[(num, s)][k] / I[(den, s)][k] for s in sub]
        res.append((sub, sum(x < 1 for x in v), float(np.median(v))))
    return res


rows = []
Ic = idx("converse")
for tam in (4, 3):
    for k, nom in (("u_final", "u final"), ("u_rob", "u robusto")):
        res = estab(Ic, "E", "B", range(28, 33), tam, k)
        gana = sum(w > tam / 2 for _, w, _ in res)
        rows.append(["converso", "E/B", nom, f"{tam} de 5", len(res), f"E gana (más de la mitad de las semillas) en {gana} de {len(res)}", f"{min(m for _, _, m in res):.2f}-{max(m for _, _, m in res):.2f}"])
for den in ("E", "B"):
    for k, nom in (("u_final", "u final"), ("u_rob", "u robusto")):
        res = estab(Ic, "M", den, (28, 29, 30), 2, k)
        gana = sum(w == 2 for _, w, _ in res)
        rows.append(["converso", f"M/{den}", nom, "2 de 3", len(res), f"M gana las 2 semillas en {gana} de {len(res)}", f"{min(m for _, _, m in res):.2f}-{max(m for _, _, m in res):.2f}"])
Id = idx("direct")
for den in ("E", "B"):
    for k, nom in (("u_final", "u final"), ("u_rob", "u robusto")):
        res = estab(Id, "M", den, (28, 29, 30), 2, k)
        gana = sum(w == 2 for _, w, _ in res)
        rows.append(["directo", f"M/{den}", nom, "2 de 3", len(res), f"M gana las 2 semillas en {gana} de {len(res)}", f"{min(m for _, _, m in res):.2f}-{max(m for _, _, m in res):.2f}"])
out.append(md_tabla(["problema", "razón", "métrica", "semillas", "subconjuntos", "resultado", "rango de la mediana"], rows))

# ------------------------------------------------------------- E. sensibilidad retrospectiva
out.append("\n## E. Sensibilidad retrospectiva (descriptiva, no es varianza explicada)\n")
out.append("Mediana de log10 de la razón sobre la métrica robusta, por factor. **Los factores «patas» y «hardware» están confundidos** (todo M corrió en A100; ver Desviaciones de la spec): el número no los separa.\n")
rows = []
for pr in ("converse", "direct"):
    I = idx(pr)
    for nom, num, den in (("cuerpo (E/B)", "E", "B"), ("patas (M/E)", "M", "E")):
        v = [np.log10(I[(num, s)]["u_rob"] / I[(den, s)]["u_rob"]) for s in range(28, 33) if (num, s) in I and (den, s) in I]
        if v:
            rows.append([pr[:4], nom, len(v), f"{np.median(v):+.2f}", f"{min(v):+.2f} a {max(v):+.2f}"])
for a in ("B", "E"):
    for hw in ("T4", "H100", "A100-80"):
        v = [f["u_rob"] for f in filas if f["problema"] == "converse" and f["brazo"] == a and f["hw"] == hw]
        if v:
            rows.append(["conv", f"{a} en {hw} (semillas {[f['semilla'] for f in filas if f['problema'] == 'converse' and f['brazo'] == a and f['hw'] == hw]})", len(v), f"mediana u rob. {np.median(v):.3f}", f"{min(v):.3f} a {max(v):.3f}"])
out.append(md_tabla(["problema", "factor", "n", "efecto", "rango"], rows))
out.append("\nMorris queda diferido hasta que haya cómputo (ver spec).")
(SALIDA / "A_B_E_resultados.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
