"""EXP-0030, bloque F (incertidumbre entre semillas sobre la malla de referencia) y bloque G (fichas, proveniencia, dominio, fallos). Exploratorio, post hoc."""
import numpy as np
from scipy.stats import spearmanr

from common import RES_IN, SALIDA, cargar, md_tabla

R = cargar()
out = ["# EXP-0030, bloques F y G (exploratorio, post hoc)\n", "## F. Incertidumbre entre semillas sobre la malla de referencia\n",
       "Campos predichos en el paso 20 000 sobre la malla de 4 221 nodos de la referencia FEM. Dispersión = rango intercuartílico entre semillas por nodo; error = |mediana entre semillas − referencia|. Spearman entre el mapa de dispersión y el de error: **solo descripción** (n = 3 o 5 semillas; los nodos no son independientes).\n"]
rows = []
for pr in ("converse", "direct"):
    ref = np.load(RES_IN / "piezo-exp0029-final/piezo_exp0029" / f"fem_{pr}_reference.npz")
    for a in ("B", "E", "M"):
        ks = sorted(k for k in R if k[0] == pr and k[1] == a)
        F = []
        for k in ks:
            z = np.load(R[k]["npz"], allow_pickle=True)
            xy, v = z["s020000/fields/xy"].astype(float), z["s020000/fields/values"].astype(float)
            assert np.allclose(xy, ref["xy"], rtol=1e-4, atol=1e-9), "la malla guardada no coincide con la de la referencia"
            F.append(v)
        F = np.stack(F)                                       # (semillas, nodos, 8)
        for j, nm in ((0, "u"), (1, "v"), (2, "φ")):
            med = np.median(F[:, :, j], axis=0)
            iqr = np.percentile(F[:, :, j], 75, axis=0) - np.percentile(F[:, :, j], 25, axis=0)
            r0 = ref[("phi" if nm == "φ" else nm)].astype(float)
            err = np.abs(med - r0)
            esc = float(np.sqrt(np.mean(r0 ** 2)))
            rows.append([pr[:4], a, len(ks), nm, f"{np.sqrt(np.mean(err ** 2)) / esc:.3f}", f"{np.median(iqr) / esc:.3f}", f"{spearmanr(iqr, err)[0]:+.2f}"])
out.append(md_tabla(["problema", "brazo", "n semillas", "campo", "error del campo mediano (rel. L2)", "dispersión mediana (IQR / RMS de la referencia)", "Spearman dispersión–error"], rows))

# ----------------------------------------------------------------------- G
out.append("\n## G. Fichas, proveniencia, dominio y fallos\n")
out.append("### Ficha por corrida (de `meta_json` y `run_card`)\n")
rows = []
for (pr, a, s), r in sorted(R.items()):
    j, m = r["json"], r["meta"]
    v = m.get("versions", {})
    rows.append([pr[:4], a, 20260700 + s, r["hw"], v.get("torch", "?"), (v.get("python", "?")), f"{j['adam_seconds'] / 20000:.3f}", f"{j['adam_seconds'] / 3600:.2f}",
                 (m.get("git_commit") or "?")[:8], "sí" if "run_card" in j else "no"])
out.append(md_tabla(["problema", "brazo", "semilla", "hardware", "torch", "python", "s/paso", "horas de proceso", "commit", "ficha de IA responsable"], rows))
out.append("\n`horas de proceso` es el tiempo de pared de un proceso; en Modal corrían 4 procesos por GPU, y en Kaggle uno por GPU. El costo en USD de la ronda de M **no se verificó** (el investigador reportó un gasto ≈ 11 USD mayor que la estimación del agente; probable causa: las A100 resultaron de 80 GB, más caras que la tarifa supuesta; sin confirmar).\n")
out.append("""### Proveniencia
- **Escribió el agente:** los scripts de este análisis (`analysis/`), las tablas generadas y este texto.
- **Decidió una persona:** qué se corre y en qué hardware, la firma de EXP-0029, la métrica robusta como secundaria (post hoc), el cierre del cómputo y la lectura de resultados.
- **Los datos de entrada** no se modificaron.

### Dominio de validez
Un bimorfo PVDF, dos problemas (converso, directo), receta del artículo (float64, Adam 20 000 pasos, sin L-BFGS, pesos fijos), n ≤ 5 semillas (M: 3), hardware mixto. No se afirma nada fuera de ese dominio.

### Catálogo de fallos
- M converso, semillas 31 y 32: sin memoria (CUDA OOM, A100 de 40 GB con 4 procesos y registro de bloques completos). No existen.
- E converso semilla 32: l2re_u 0.14 → 0.09 → 0.52 en los pasos 10 000, 15 000, 20 000 (salto al final).
- Saltos de pérdida: solo medibles por paso en las corridas de EXP-0029 (A100); B y E semillas 28-30 tienen la pérdida cada 100 pasos.

### Limitaciones de comparabilidad
- **Hardware × brazo:** B y E semilla 28 en T4, semillas 29-30 en H100, semillas 31-32 en A100-80GB; M converso y directo 28 en A100-80GB, M directo 29-30 en A100-40GB. No hay pares M/E con el mismo hardware.
- El pareo por semilla no es exacto en float64 entre dispositivos.
- n distinto por brazo y por problema.
- La semilla 28 de B y E es la del artículo; en el converso 31-32 no hay M.
""")
(SALIDA / "F_G_incertidumbre_fichas.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out)[:6000])
