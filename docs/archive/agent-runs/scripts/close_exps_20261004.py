import json
import pathlib
import re

ROOT = pathlib.Path(r"C:\Users\rodri\Documents\AF_PIKANs")
EXP = ROOT / "experiments"
D = "2026-10-04"
BY = 'Rodrigo Mansilla (chat, 2026-10-04: "apruebo" / "cierro" / "aprobados")'
BL = "`evals/baselines/baseline_v1_exploratory.*` (2026-10-03) y `baseline_v2_exploratory.*` (2026-10-04)"


def w(path, txt):
    p = EXP / path
    p.write_text(txt.strip() + "\n", encoding="utf-8")


def closure(exp, decision, title, body):
    return f"""---
experiment_id: {exp}
decision: {decision}
decided_at: {D}
decided_by: {BY}
supersedes:
---

# EDR — {exp}: {title}

Cierre **de orden**: redactado por un agente y aprobado por el investigador. No cambia ningún número ni es un claim; los baselines vigentes son {BL}.

{body}
"""


w("heat/EXP-0011-heat-rad-adam/decision.md", f"""---
experiment_id: EXP-0011
decision: rejected        # accepted | rejected | inconclusive | deferred
decided_at: {D}
decided_by: {BY}
supersedes:
---

# EDR — EXP-0011: RAD sobre el control RBF de Heat

**Decisión: rechazada, por la regla pre-registrada y firmada el 2026-10-03.** Aprobada por el investigador.

- **Criterio primario cumplido:** razón RAD/control al final de Adam, media geométrica 0.088 (umbral ≤ 0.80), 5/5 semillas ≤ 0.80.
- **Guardia violada:** error tras L-BFGS completo 4.9× el del control (límite 2.0×). La condición de falsación (c) rechaza la hipótesis.
- **Alcance:** se rechaza H-RAD-HEAT-1 con esta configuración (RAD potencia 2, 50 % uniforme, refresco 1 000), en este control, con n = 5. No es un resultado general sobre RAD.
- **Seguimiento:** EXP-0012 (cerrado, ver su decisión). Patrón resultante en `experiments/INDEX.md`.
- Resultados: `results.md`. Nada de esto es un claim (gate `scientific_claim`).
""")

w("heat/EXP-0012-heat-rad-uniform-lbfgs/decision.md", f"""---
experiment_id: EXP-0012
decision: rejected        # accepted | rejected | inconclusive | deferred
decided_at: {D}
decided_by: {BY}
supersedes:
---

# EDR — EXP-0012: ¿el déficit tras L-BFGS venía de los puntos de L-BFGS?

**Decisión: rechazada por la regla pre-registrada (media geométrica 2.025 > 2.0; 2/5 semillas ≤ 2.0). El investigador cierra el hilo RAD en Heat (opción A) y no abre EXP-0013.**

## Patrón registrado explícitamente

> **En el control RBF de Heat, RAD mejora una etapa y empeora la otra.** Reduce el error al final de Adam ~11× (razón 0.088, 5/5 semillas) y deja un estado que L-BFGS completo no refina al piso del control: ~4.9× peor si L-BFGS corre sobre los puntos RAD, y ~2.0× peor (2.025) si corre sobre el pool uniforme. Corregir los puntos de L-BFGS recupera ~2.5× pero no cierra la brecha.

Límites del patrón (declarados): n = 5 semillas, un problema (Heat), una configuración de RAD; en las 5 corridas L-BFGS terminó por tope (2 de iteraciones, 3 de tiempo), así que no se probó si más iteraciones lo cierran. Eso no se hizo ni se hará dentro de este hilo.

## Consecuencias para el diseño del primer mecanismo (reglas, no resultados)
1. Un mecanismo se evalúa **en las dos etapas** (fin de Adam y tras L-BFGS); mejorar solo Adam no basta. La guardia de L-BFGS es vinculante.
2. L-BFGS se evalúa sobre un pool **fijo y declarado** de antemano (no sobre los puntos que el mecanismo concentra).
3. El mecanismo se pre-registra con esa guardia ya incorporada.

Resultados: `results.md`. Nada de esto es un claim.
""")

C = {
    "heat/EXP-0001-heat1d-baseline": ("EXP-0001", "superseded", "Heat 1D baseline (borrador inicial)",
        "- **Estado real:** borrador que nunca se firmó ni se ejecutó con este id.\n"
        "- **Etiqueta:** cancelado y sustituido. El baseline de Heat lo establecen las campañas de EXP-0007 y EXP-0008 (control RBF: 2.197e-4 al final de Adam, 1.857e-6 tras L-BFGS completo), congeladas en v1.\n"
        "- `decision` = `superseded` (extiende la lista `accepted | rejected | inconclusive | deferred` de la plantilla)."),
    "heat/EXP-0002-heat-burgers-sobol": ("EXP-0002", "closed_exploratory_record", "Heat + Burgers, primera campaña Sobol",
        "- **Estado real:** campaña exploratoria sin spec ni pre-registro, solo manifiestos (`manifest.json`, `manifest-v3-t4x2.json`) y `parameter-decisions.md`.\n"
        "- **Etiqueta:** registro exploratorio, no confirmatorio. Sus cifras fueron superadas por EXP-0007 y EXP-0008 (Heat) y por las campañas `burgers-close`, `burgers-f64` y `burgers-big` (Burgers).\n"
        "- Los cuadernos de resultados viven en `notebooks/heat/EXP-0002-heat-burgers-sobol/`. EXP-0005 (pantalla de arquitectura, sin carpeta propia) queda en la misma categoría."),
    "allen-cahn/EXP-0003-allen-cahn-rbf-sobol": ("EXP-0003", "closed_exploratory_record", "Allen–Cahn RBF Sobol, configuración v1",
        "- **Estado real:** configuración exploratoria v1; el propio spec dice \"no es preregistro confirmatorio\".\n"
        "- **Etiqueta:** registro exploratorio. Las cifras contra referencias generadas por un agente (FD8192/ETDRK4) **no cuentan**. El baseline vigente de Allen–Cahn usa `Allen_Cahn.mat` (DeepXDE), aceptado el 2026-10-01: RBF16 2.686e-3 ± 9.5e-4 tras L-BFGS, MLP 1.773e-2 ± 8.7e-3."),
    "burgers/EXP-0006-burgers-sobol-rad": ("EXP-0006", "rejected", "Burgers: RAD contra Sobol (resultado negativo)",
        "- **Estado real:** pre-registro preparado pero **nunca firmado**; sus criterios (razón geométrica ≤ 0.8) eran \"propuestos, NO activos\". Se ejecutó bajo autorización exploratoria.\n"
        "- **Resultado:** RAD sobre el RBF de esa campaña salió ~5.9× peor que Sobol (IC95 2.8–12.1; 5/5 parejas), con una base no convergida y un diseño frágil. Resultado negativo exploratorio, sin criterio activado.\n"
        "- **Etiqueta:** cerrado como resultado negativo. Coincide con el patrón de EXP-0011 y EXP-0012 (RAD no ayuda sobre estos controles). El baseline vigente de Burgers es el RBF48 de v2."),
    "heat/EXP-0007-heat-stepmatched-deepxde": ("EXP-0007", "closed_exploratory_record", "Heat step-matched (RBF vs B-spline) con presupuesto DeepXDE",
        "- **Estado real:** contrato `PENDING_HUMAN_SIGNATURE`, nunca firmado; se ejecutó bajo autorización exploratoria (campaña S2 y seguimiento).\n"
        "- **Etiqueta:** registro exploratorio. El L-BFGS archivado de S2 (2.06e-6) estaba truncado por un defecto del arnés y **queda supersedido** por el L-BFGS completo (1.857e-6 ± 1.8e-7, `heat-lbfgs-redo-20260930`), que es lo que figura en v1. El B-spline de esta campaña no entra al baseline."),
    "baseline-v2/EXP-0008-baseline-v2-competitive": ("EXP-0008", "closed_superseded_by_baselines", "Plan del baseline competitivo",
        "- **Estado real:** paquete de plan, recetas y evaluadores con `status: draft`; sus campañas (Heat L-BFGS, AC, NS2d-CG, DFG, Burgers) se ejecutaron bajo aprobaciones de sesión.\n"
        "- **Etiqueta:** cerrado. Sus resultados quedaron consolidados y congelados en v1 y v2; el estado vigente está en `experiments/<problema>/RESULTS.md`."),
    "piezo/EXP-0009-piezo-indirect-rbf": ("EXP-0009", "closed_baseline_recorded", "Piezo indirecto: problema, referencia y control",
        "- **Estado real:** spec en borrador; el control está registrado en v2: receta de Daniel, n = 5, RBF16 u/v/φ 0.092/0.104/0.048 contra MLP 0.136/0.186/0.046; RBF gana 5/5 pareado en u y v, 2.3× el costo, no convergido.\n"
        "- **Etiqueta:** cerrado como baseline. La reimplementación propia (`piezo_pinn.py`) queda como diagnóstico negativo.\n"
        "- **Abierto (no es un cierre):** el FEM es del mismo grupo (sin contraste de terceros) y la licencia de Daniel no está confirmada. El mecanismo en piezo irá en un EXP nuevo, después de DFG."),
}
for k, (exp, dec, t, b) in C.items():
    w(k + "/decision.md", closure(exp, dec, t, b))

for k in ("heat/EXP-0001-heat1d-baseline", "baseline-v2/EXP-0008-baseline-v2-competitive",
          "piezo/EXP-0009-piezo-indirect-rbf", "heat/EXP-0012-heat-rad-uniform-lbfgs"):
    p = EXP / k / "spec.md"
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"^status:\s*\w+", "status: decided", s, count=1, flags=re.M)
    p.write_text(s, encoding="utf-8")

idx = EXP / "INDEX.md"
s = idx.read_text(encoding="utf-8")
s += """
## Estado de los EXP (2026-10-04)

| EXP | Estado | Decisión |
|---|---|---|
| 0001 | decidido | superseded (nunca se corrió) |
| 0002 / 0005 | cerrados | registro exploratorio |
| 0003 | cerrado | registro exploratorio (referencia AC vigente: `Allen_Cahn.mat`) |
| 0004 (DFG) | borrador | abierto: se retoma en la fase de DFG |
| 0006 | cerrado | rechazado (resultado negativo exploratorio de RAD en Burgers) |
| 0007 | cerrado | registro exploratorio (L-BFGS truncado supersedido) |
| 0008 | cerrado | superseded por los baselines v1 y v2 |
| 0009 | cerrado | baseline de piezo registrado en v2 |
| 0011 | decidido | **rechazada** por su regla pre-registrada |
| 0012 | decidido | **rechazada**; hilo RAD en Heat cerrado |

## Patrón registrado (EXP-0011 y EXP-0012)

En el control RBF de Heat, **RAD mejora una etapa y empeora la otra**: ~11× menos error al final de Adam, y un estado que tras L-BFGS completo queda 2–5× por encima del control (4.9× con L-BFGS sobre los puntos RAD, 2.0× sobre el pool uniforme). Reglas derivadas para el primer mecanismo: se evalúa en las dos etapas, con L-BFGS sobre un pool fijo declarado de antemano, y la guardia de L-BFGS es vinculante.
"""
idx.write_text(s, encoding="utf-8")

reg = ROOT / "registry"
with open(reg / "proposals.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"proposal_id": "EXP-0012-heat-rad-uniform-lbfgs", "date": D, "status": "closed_rejected",
                        "outcome": "geomean 2.025 > 2.0; 2/5 seeds <= 2.0; hilo RAD en Heat cerrado por decision del investigador (no EXP-0013)",
                        "pattern": "RAD mejora fin de Adam ~11x y empeora el estado tras L-BFGS (2.0-4.9x)", "recorded_by": "agent"}, ensure_ascii=False) + "\n")
with open(reg / "failures.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"failure_id": "rad-stage-tradeoff-heat-20261004", "experiment": "EXP-0011/EXP-0012", "cluster": "mechanism",
                        "description": "RAD sobre el control RBF de Heat mejora fin de Adam (razon 0.088) pero deja un estado que L-BFGS no refina al piso del control (4.9x con puntos RAD, 2.0x con pool uniforme)",
                        "diagnosis": "hipotesis parcialmente confirmada: los puntos de L-BFGS explican ~2.5x, no toda la brecha; no probado si mas iteraciones cierran la brecha",
                        "rules": ["evaluar en las dos etapas", "L-BFGS sobre pool fijo declarado", "guardia de L-BFGS vinculante"],
                        "recorded": D, "recorded_by": "agent"}, ensure_ascii=False) + "\n")
print("ok")
