#!/usr/bin/env python3
"""properties_terms.csv -> properties_terms.md (NS con residual_terms, DFG, y el stand-in del piezo; mediana [min, max] entre semillas)."""
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROBLEMS = ["ns_terms", "dfg", "piezo_standin"]
SKIP = ("loss0_",)


def fmt(v):
    return "0" if v == 0 else f"{v:.3g}"


def main():
    d = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(HERE / "properties_terms.csv", newline="")):
        d[r["problema"]][r["propiedad"]].append(float(r["valor"]))
    keys = []
    for p in PROBLEMS:
        for k in d[p]:
            if k not in keys:
                keys.append(k)
    terms = [k for k in keys if k.startswith("loss0_") and k not in ("loss0_pde", "loss0_bc", "loss0_terms")]
    main_keys = [k for k in keys if k not in terms]
    out = ["# EXP-0028 Fase 2 (ampliacion) - propiedades con `residual_terms` (PDE + borde + pesos), sin entrenar, CPU", "",
           "Semillas 4001-4010, n = 10 por problema. Celda = mediana [min, max]. Generado por `make_table_terms.py` desde `properties_terms.csv`, medido con `measure_properties_terms.py`.",
           "Modelo, semilla, pools y puntos del GN identicos a la fase 1c (`run_phase1c.py`, `run_phase1c_dfg.py` importados sin modificar; GN = 2048 por clave del pool, bloques por arista, chunk 16).",
           "- `ns_terms`: NS2d-CG repetido con los 8 terminos (3 PDE + 5 de borde, peso 1); la fase 2 original (`properties.md`, columna `ns`) usaba solo PDE para cond(B) (por eso `pde_condB_*` debe coincidir con ella: 7.2e3 frente a 7.21e3).",
           "- `dfg`: DFG 2D-1 Re20 con `DFGTerms` (13 terminos: 3 PDE + inlet 2 + paredes y cilindro 6 + salida 2; pesos 1 y 1/2), modelo `DFGRBFPIKAN` con `output_scales` dentro del forward, pools completos de `generate_dfg_pools` (interior 16 384, 512 por frontera).",
           "- `piezo_standin`: **NO es el problema del paper.** Es la reimplementacion propia `experiments/piezo/EXP-0009-piezo-indirect-rbf/piezo_pinn.py` (2 entradas xi=x/L, eta=y/H; 8 salidas; bordes duros en u, v, phi; escalas SC_*; terminos constitutivos + equilibrio + libres de traccion; pesos 1, sin pesos adaptativos, sin enrutado stop-gradient, sin caracteristicas de fase), RBF16 de 11 bases, float64, 2048 interiores + 256 por borde (receta del paper). El adaptador `piezo_terms` esta en `measure_properties_terms.py`; reproduce `loss_fn` de `piezo_pinn.py` (error relativo ~1e-16). Sirve para medir propiedades de la fisica y del reparto de escalas, no para predecir el codigo de Daniel.", ""]
    out += ["| propiedad | " + " | ".join(PROBLEMS) + " |", "|---|---|---|---|"]
    for title, ks in (("principales", main_keys), ("perdida inicial por termino", terms)):
        if title != "principales":
            out += ["", "**Perdida inicial por termino (w_t * media r_t^2, pool completo)**", "", "| termino | " + " | ".join(PROBLEMS) + " |", "|---|---|---|---|"]
        for k in ks:
            cells = []
            for p in PROBLEMS:
                v = d[p].get(k)
                cells.append("-" if not v else fmt(v[0]) if max(v) == min(v) else f"{fmt(statistics.median(v))} [{fmt(min(v))}, {fmt(max(v))}]")
            if any(c != "-" for c in cells):
                out.append(f"| `{k}` | " + " | ".join(cells) + " |")
    out += ["", "## Definiciones", "",
            "- `condB_*`, `P_*`: como la fase 2 original (B/m + 1e-3 I por arista), pero con B = suma sobre TODOS los terminos (con pesos); `pde_condB_*` solo con los terminos de interior.",
            "- `gn_trace_bc_over_pde`: traza total de los bloques GN de los terminos de borde / los de interior (cuanto del precondicionador lo dicta el borde).",
            "- `resid_conc_top10(_sq)`: fraccion de la masa de sqrt(sum_k r_k^2) (o r^2) en el 10 % de puntos de interior con mayor valor, pool completo; `resid_conc_all_sq_top10`: con la contribucion w_t r^2/N_t de todos los puntos de todos los terminos (muy alta = la perdida inicial la dicta un grupo pequeno de puntos/terminos).",
            "- `bc_over_pde`, `loss0_*`: perdida inicial por terminos con pesos, pool completo. `term_loss_ratio_max_min`: max/min de la perdida por termino (disparidad de escalas entre terminos). `terms_vs_loss_relerr`: adaptador frente a la perdida de entrenamiento (en float32 ~1e-7 es redondeo).",
            "- `output_scale_ratio`: max/min de las escalas caracteristicas de las salidas (unidades mezcladas); piezo: u 3e-7, v 3.3e-5, phi 100 V, tensiones 6e3, D 1e-5; `output_scale_ratio_uv_phi` = 100 V / 3e-7 m. DFG: U_MEAN/PRESSURE_SCALE; NS: 1 (sin escalas).",
            "- `peclet` = Re = U L / nu (NS: 100; DFG: U_MEAN D / nu = 0.2*0.1/0.001 = 20). Piezo: no aplica (sin adveccion).",
            "- `edge_trace_*`, `layer_trace_*`: solo 3 semillas, ver `properties_edge_scale.csv`."]
    (HERE / "properties_terms.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("escrito properties_terms.md")


if __name__ == "__main__":
    main()
