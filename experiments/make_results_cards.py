#!/usr/bin/env python3
"""Regenerates experiments/<problem>/RESULTS.md from the per-seed CSV/JSON in each problem's results/ folders.

Seeds are the unit of dispersion (mean ± sample sd). Both optimisation stages are reported and never mixed.
Run: py -3.13 experiments/make_results_cards.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E8 = "EXP-0008-baseline-v2-competitive"
STATUS = ("Exploratorio. `scientific_accepted=false`, `reference_accepted=false`. Estado frente al baseline v1 firmado el 2026-10-03: "
          "ver `evals/baselines/baseline_v1_exploratory.md`.")


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ms(vals):
    vals = [float(v) for v in vals]
    return f"{st.mean(vals):.3e} ± {st.stdev(vals):.2e}" if len(vals) > 1 else f"{vals[0]:.3e}"


def write(problem, text):
    p = ROOT / problem / "RESULTS.md"
    p.write_text(text.rstrip() + "\n", encoding="utf-8")
    print("wrote", p)


def card_heat():
    d = json.load(open(ROOT / f"heat/{E8}/results/heat-lbfgs-redo-20260930/results-summary.json", encoding="utf-8"))
    write("heat", f"""# Heat 1D — resultados del baseline (EXP-0008)

{STATUS}

**Contrato.** u_t = 0.1 u_xx en [0,1]², solución exacta sin(πx)e^(−0.1π²t), residuo escalado por 1.1. RBF-PIKAN `rbf_fixed`, float64, lote completo de
4 096 puntos interiores, Adam 20 000 pasos (lr 1e-3) y L-BFGS strong-Wolfe (lr 1, historia 100, bloques de 25 iteraciones con 125 evaluaciones).
Semillas 901–905, pools Sobol pareados. Puntuado en la malla analítica 512×256.

| Etapa | L2 relativo (media ± sd, n=5) |
|---|---|
| Fin de Adam | {ms([x['adam_final_rel_l2'] for x in d])} |
| + L-BFGS completo | **{ms([x['rel_l2_512x256'] for x in d])}** |

**Lectura.** El L-BFGS elimina la dispersión entre semillas y llega a un piso de ≈1.9e-6 en ~50 iteraciones, limitado por la discretización
(4 096 puntos), no por la optimización. Un mecanismo solo puede mostrar efecto medible en la etapa Adam o en el costo para llegar al piso.
El valor archivado 2.06e-6 (L-BFGS truncado) queda supersedido.

**Comparadores.** DeepXDE α=0.1 (no pareado): 5.81e-4; B-spline S2 con L-BFGS: 1.416e-6 (seguimiento parcial).

**Artefactos.** `{E8}/results/heat-lbfgs-redo-20260930/`, figura `{E8}/figures/heat-lbfgs-stages-and-curves.png`, checkpoints con hashes en `checkpoints/`.
**Siguiente experimento.** EXP-0011 (RAD sobre este control), pre-registrado y firmado el 2026-10-03.
""")


def card_burgers():
    r1 = rows(ROOT / f"burgers/{E8}/results/burgers-close-20261003/burgers-close-per-seed.csv")
    r2 = rows(ROOT / f"burgers/{E8}/results/burgers-f64-lbfgs-20261003/burgers-f64-per-seed.csv")
    r3 = rows(ROOT / f"burgers/{E8}/results/burgers-big-20261003/burgers-big-per-seed.csv")
    g = lambda R, arm, w, k: [x[k] for x in R if x["arm"] == arm and x["width"] == w]
    fem = ROOT / f"burgers/{E8}/results/burgers-close-20261003/burgers-vs-pinnacle-fem.csv"
    femtxt = ""
    if fem.exists():
        rf = rows(fem)
        femtxt = "\n**Contra la referencia FEM de PINNacle** (101×11; comparación directa con lo publicado, 1.33e-2 ± 5.3e-5 con L-BFGS): " + "; ".join(
            f"{a}{w} {ms([x['l2_fem_lbfgs'] for x in rf if x['arm'] == a and x['width'] == w])}" for a, w in (("mlp", ""), ("rbf", "16"), ("rbf", "48"))) + ". Es el piso de la referencia: no discrimina entre modelos.\n"
    write("burgers", f"""# Burgers 1D (PINNacle) — resultados del baseline (EXP-0008)

{STATUS}

**Contrato.** u_t + u u_x = (0.01/π) u_xx en [−1,1]×[0,1], IC −sin(πx), BC homogéneas duras; referencia Cole–Hopf (Gauss–Hermite 128) 1025×129, evaluador del verificador.
8 192 puntos Sobol por semilla compartidos entre brazos, Adam 20 000 pasos (lr 1e-3, float32) y L-BFGS strong-Wolfe. Semillas 301–305.

| Brazo | Parámetros | Fin de Adam | L-BFGS float32 (600 s) | L-BFGS float64 (900 s) | **Grande**: 16 384 pts, 24k Adam, L-BFGS f64 1 800 s |
|---|---|---|---|---|---|
| MLP 5×100 (PINNacle Vanilla) | 40 801 | {ms(g(r1,'mlp','','l2_adam'))} | {ms(g(r1,'mlp','','l2_lbfgs'))} | {ms(g(r2,'mlp','','l2_lbfgs'))} | {ms(g(r3,'mlp','','l2_lbfgs'))} |
| RBF-PIKAN ancho 16 | 3 648 | {ms(g(r1,'rbf','16','l2_adam'))} | {ms(g(r1,'rbf','16','l2_lbfgs'))} | — | — |
| RBF-PIKAN ancho 48 | 29 376 | {ms(g(r1,'rbf','48','l2_adam'))} | {ms(g(r1,'rbf','48','l2_lbfgs'))} | {ms(g(r2,'rbf','48','l2_lbfgs'))} | **{ms(g(r3,'rbf','48','l2_lbfgs'))}** |
{femtxt}
**Lectura.** El MLP reproduce el Vanilla publicado (1.45e-2) al final de Adam. RBF48 empata con el MLP tras L-BFGS (pareado float64 −1.2e-4 ± 4.1e-4, RBF gana 3/5): no hay ventaja demostrada.
float64 mejora ~25–30 % sobre float32; todas las corridas de float64 terminaron por tope de tiempo. RBF16 es peor (~3×).
La corrida grande (16 384 puntos, 24 000 pasos de Adam, L-BFGS float64 1 800 s; sesión de 4.6 h) baja ~4× más: MLP 1.05e-4 ± 2.2e-5, RBF48 **7.16e-5 ± 4.8e-5**; pareado RBF48 − MLP −3.3e-5 ± 6.4e-5 (RBF gana 4/5): ventaja pequeña, sin significancia con n=5. Todas terminaron por tope de tiempo de L-BFGS.

**Artefactos.** `{E8}/results/burgers-close-20261003/`, `burgers-f64-lbfgs-20261003/`, figura `{E8}/figures/burgers-l2-by-stage.png`.
**Límites.** Sin SOTA de métodos especializados reproducido; el control del baseline es RBF48.
""")


def card_ac():
    r = rows(ROOT / f"allen-cahn/{E8}/results/ac-rerun-20260930/ac-rerun-per-seed.csv")
    g = lambda a, k: [x[k] for x in r if x["arm"] == a]
    write("allen-cahn", f"""# Allen–Cahn 1D — resultados del baseline (EXP-0008)

{STATUS}

**Contrato (DeepXDE).** u_t = 0.001 u_xx + 5(u − u³) en [−1,1]×[0,1], IC x²cos(πx), transformada dura; referencia `Allen_Cahn.mat` (DeepXDE, tercero; **aceptada por el investigador el 2026-10-01**), malla 201×101.
MLP 3×20 (921 parámetros) frente a RBF-PIKAN ancho 16; 8 000 puntos Sobol, Adam 40 000 pasos (float32) + L-BFGS (tope 300 s). Semillas 601–605.

| Brazo | Fin de Adam | + L-BFGS |
|---|---|---|
| MLP 3×20 | {ms(g('mlp','l2_adam'))} | {ms(g('mlp','l2_lbfgs'))} |
| RBF-PIKAN 16 | {ms(g('rbf','l2_adam'))} | **{ms(g('rbf','l2_lbfgs'))}** |

**Lectura.** El RBF gana en las 5 semillas, en ambas etapas. DeepXDE oficial (no pareado): 2.00e-2 ± 6.1e-3. Las referencias generadas por agente (FD8192/ETDRK4) siguen rechazadas y no cuentan.

**Artefactos.** `{E8}/results/ac-rerun-20260930/`, figura `{E8}/figures/ac-l2-by-stage.png`.
""")


def card_ns():
    ns = rows(ROOT / f"navier-stokes/{E8}/results/ns2d-cg-20260930/ns2dcg-per-seed.csv")
    dl = rows(ROOT / f"navier-stokes/{E8}/results/dfg-long-20261001/dfg-long-per-seed.csv")
    nsrows = []
    for a in sorted({x["arm"] for x in ns}):
        for pr in sorted({x["params"] for x in ns if x["arm"] == a}, key=int):
            v = [x["flat"] for x in ns if x["arm"] == a and x["params"] == pr]
            nsrows.append(f"| {a} ({int(pr):,} par., n={len(v)}) | {ms(v)} |")
    dlrows = []
    for a in ("mlp", "rbf"):
        for stage, lab in (("adam_final", "Adam"), ("lbfgs_final", "+ L-BFGS")):
            R = [x for x in dl if x["arm"] == a and x["stage"] == stage]
            dlrows.append(f"| {a} {lab} | {ms([x['u'] for x in R])} | {ms([x['Cd_err'] for x in R])} |")
    write("navier-stokes", f"""# Navier–Stokes (NS2d-CG y DFG Re=20) — resultados del baseline (EXP-0008)

{STATUS}

## NS2d-CG (PINNacle, backward step, Re=100)
Referencia COMSOL `ns_0_obstacle.dat` (sha256 13e1d9fa…). MLP 5×100 frente a RBF-PIKAN; Adam 20 000 pasos (float32), sin L-BFGS. Semillas 701–705 (RBF16: 701–703). L2RE plano sobre las tres salidas
(la agregación de PINNacle no es verificable en su código).

| Brazo | L2RE plano, fin de Adam |
|---|---|
{chr(10).join(nsrows)}

PINNacle publica vanilla 0.119 y LAAF 0.0824. RBF gana pareado; sin L-BFGS corrido.

## DFG 2D-1 Re=20
- **Cohortes previas (control del v1):** RBF-PIKAN 301–305 tras L-BFGS: Cd 5.124 ± 0.053 (error 8.16 %), Δp error 9.89 %; MLP 5×64 (501–505): Cd error 5.3 %. Fórmula de fuerzas σ·n corregida el 2026-09-30. Campo contra FEM Taylor–Hood L3 (referencia de trabajo aceptada con Cl absoluto).
- **Receta de la tesis de Daniel (2 000 Adam + 1 000 L-BFGS):** ningún brazo resuelve (MLP u ≈ 0.88, RBF ≈ 0.51).
- **Receta larga (informativa, no converge; L-BFGS terminó por tope de tiempo):**

| Brazo / etapa | u (L2 rel. vs FEM L3) | error de Cd |
|---|---|---|
{chr(10).join(dlrows)}

**Artefactos.** `{E8}/results/{{ns2d-cg-20260930,dfg-daniel-cfg-20260930,dfg-long-20261001}}/`, figuras `ns2dcg-flat-l2re.png` y `dfg-long-by-stage.png`.
""")


def card_piezo():
    write("piezo", """# Piezoelectricidad (bimorfo PVDF, formulación indirecta) — estado (EXP-0009)

Exploratorio. `scientific_accepted=false`. Referencia de trabajo: FEM de Daniel González con `poling_sign=−1` (FEM del mismo grupo; falta contraste de terceros).

**Resumen honesto.** Con nuestra propia formulación mixta de 8 campos (reimplementación propia), ningún brazo RBF-PIKAN salió de la solución nula en 12+ variantes; el MLP solo aprendió la flexión
con pesos adaptativos + separación de escalas de v (u ≈ v ≈ 0.19 al final de Adam, una semilla) y el L-BFGS siempre destruyó la solución. El mismo fallo está documentado por el propio código de Daniel
(`scaling.py`: "the trivial field u = v = 0 is the only field that pays nothing"). Hallazgos: la convención de signos físicamente consistente es e31 y e33 de signo opuesto (los residuos de equilibrio del FEM lo confirman);
la tabla `tab:pvdf` de la metodología coincide con la inversión de e33 del cargador de Daniel, no con la física.

**Con la receta del paper de Daniel** (commit e7856f1, 2026-10-02: 20 000 Adam, float64, formulación por capas con interfaz, SiLU, escalas por campo, remuestreo cada 200 épocas, sin L-BFGS) ejecutada con su MLP y con un tronco RBF-PIKAN
intercambiado (`daniel_arm.py`), semilla del paper 20260728, evaluador propio contra el FEM:

| Brazo (n=5 semillas: 20260728–32) | Parámetros | u | v | φ | Tiempo por corrida |
|---|---|---|---|---|---|
| MLP 3×50 SiLU | 5 758 | 0.136 ± 0.007 | 0.186 ± 0.011 | 0.046 ± 0.000 | ~4 000 s |
| RBF-PIKAN 16 | 5 376 | **0.092 ± 0.021** | **0.104 ± 0.048** | 0.048 ± 0.003 | ~9 400 s |

Diferencia pareada RBF − MLP: u −0.054, −0.003, −0.073, −0.036, −0.055; v −0.104, −0.006, −0.152, −0.083, −0.066 (el RBF gana 5/5 en ambas). Por semilla en `daniel-recipe-per-seed.csv`.

Con la receta correcta el RBF aprende la flexión y queda por debajo del MLP en u y v en las 5 semillas, con mayor dispersión y ~2.4× el costo; la trayectoria del RBF aún mejora al final de las 20 000 épocas (no convergido). El fallo de nuestra reimplementación propia era de formulación (capas con interfaz, escalas por campo, enrutamiento de gradiente), no del RBF.

Detalle completo, tablas por variante y diagnóstico: `EXP-0009-piezo-indirect-rbf/spec.md`. Registro de fallos: `registry/failures.jsonl` (`piezo-null-state-20261003`).
""")


if __name__ == "__main__":
    for fn in (card_heat, card_burgers, card_ac, card_ns, card_piezo):
        fn()
