#!/usr/bin/env python3
"""Fase 2: properties.csv -> properties.md (tabla problema x propiedad: mediana [min, max] entre semillas)."""
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBLEMS = ["ac_d1e-3", "ac_d2e-4", "burgers", "kpp", "ns"]
GROUPS = [
    ("(1) Condicionamiento de los bloques por arista (B_n + 1e-3 I) y de P", ["condB_min", "condB_median", "condB_max", "condB_l0_median", "condB_l1_median", "condB_l2_median", "P_sv_min", "P_sv_max", "P_cond_median", "n_edges", "n_edges_dead"]),
    ("(2) Gram uniforme de las 11 bases (covariable de F)", ["gram_cond_raw", "gram_cond_reg"]),
    ("(3) Concentracion del residuo inicial (0.10 = uniforme)", ["resid_conc_top10", "resid_conc_top10_sq", "resid_rms0"]),
    ("(4) Numeros adimensionales del PDE (constantes)", ["damkohler", "peclet", "front_width", "front_over_rbf", "rbf_width_x"]),
    ("(5) Borde frente a PDE en la perdida inicial", ["bc_over_pde", "loss0_pde", "loss0_bc"]),
    ("(6) Salidas y ecuaciones acopladas", ["n_outputs", "n_coupled_eqs", "n_loss_terms", "model_width"]),
    ("(7) Perdida y gradiente iniciales", ["loss0", "grad_norm0", "grad_norm0_coef"]),
]


def fmt(v):
    if v == 0:
        return "0"
    return f"{v:.3g}"


def main():
    d = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(HERE / "properties.csv", newline="")):
        d[r["problema"]][r["propiedad"]].append(float(r["valor"]))
    probs = [p for p in PROBLEMS if p in d]
    n = {p: len(next(iter(d[p].values()))) for p in probs}
    out = ["# EXP-0028 Fase 2 - propiedades a la inicializacion (sin entrenar, CPU, float32)", "",
           "Semillas 4001-4010 (n por problema: " + ", ".join(f"{p}={n[p]}" for p in probs) + "). Celda = mediana [min, max] entre semillas; '-' = no aplica. "
           "Modelo y puntos identicos al entrenamiento de EXP-0023 (RBF16 en AC/KPP/NS, RBF48 en Burgers, 11 bases, ancho 0.35; NS con ancho oculto 16, no fijado por el spec). "
           "Generado por `make_table.py` desde `properties.csv`; medido con `measure_properties.py`.", ""]
    for title, props in GROUPS:
        out += [f"## {title}", "", "| propiedad | " + " | ".join(probs) + " |", "|---|" + "---|" * len(probs)]
        for k in props:
            cells = []
            for p in probs:
                v = d[p].get(k)
                if not v:
                    cells.append("-")
                elif max(v) == min(v):
                    cells.append(fmt(v[0]))
                else:
                    cells.append(f"{fmt(statistics.median(v))} [{fmt(min(v))}, {fmt(max(v))}]")
            if any(c != "-" for c in cells):
                out.append(f"| `{k}` | " + " | ".join(cells) + " |")
        out.append("")
    out += (HERE / "notas.md").read_text(encoding="utf-8").splitlines() if (HERE / "notas.md").exists() else []
    (HERE / "properties.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("escrito properties.md")


if __name__ == "__main__":
    main()
