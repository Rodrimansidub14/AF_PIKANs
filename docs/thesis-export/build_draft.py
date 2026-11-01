#!/usr/bin/env python3
"""Rellena las tablas del borrador `mecanismo-preliminar.template.md` desde los CSV de EXP-0013 y genera `mecanismo-preliminar.md` (+ CSV en tables/)."""
import shutil
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parents[1] / "experiments/diagnostics/EXP-0013-run-diagnosis"
A = R / "results/analysis"
shutil.copy2(R / "figures/matrix_signal_problem.png", HERE / "figures/matrix_signal_problem.png")

def md(df): return df.to_markdown(index=False)

base = pd.DataFrame([
    ["Heat 1D", "analítica", "RBF, 3 648 par., centros [−1.05,1.05], σ=0.21, float64", "2.2e-4 ± 1.8e-4", "1.86e-6 ± 1.8e-7"],
    ["Burgers 1D", "Cole–Hopf (GH128, 1025×129)", "RBF48, 29 376 par., σ=0.35, 8–16 k puntos Sobol", "5.4e-3 ± 2.0e-3 (v2)", "7.2e-5 ± 4.8e-5 (v2); MLP 1.05e-4 ± 2.2e-5"],
    ["Allen–Cahn 1D", "`Allen_Cahn.mat` (DeepXDE)", "RBF16, σ=0.35, float32", "2.84e-2 ± 7.4e-3", "2.69e-3 ± 9.5e-4; MLP 1.77e-2 ± 8.7e-3"],
    ["NS2d-CG", "COMSOL (12 910 nodos)", "RBF56 (40 992 par.), float32", "0.0655 ± 0.0058 (L2RE plano)", "no corrido; MLP 0.0977 ± 0.0067"],
    ["DFG 2D-1 Re=20", "FEATFLOW (QoI) y FEM L3", "RBF 2-16-16-3, cohortes 301–305 y 501–505", "—", "error de Cd 8.2 % y 6.2 %; Δp 9.9 % y 7.3 %"],
    ["Piezo (bimorfo PVDF)", "FEM P2, `poling_sign=−1`", "RBF16 (5 376 par.), receta de la referencia, float64, n = 5", "u/v/φ: 0.092/0.104/0.048", "sin L-BFGS; MLP 0.136/0.186/0.046"],
], columns=["Problema", "Referencia", "Control RBF", "Fin de Adam (L2 rel.)", "Tras L-BFGS (L2 rel.)"])
base.to_csv(HERE / "tables/baselines.csv", index=False)

q1 = pd.read_csv(A / "q1_residual_vs_error.csv")
q1 = q1[q1["arm"].isin(["mlp", "rbf"])]
q1 = q1[["problem", "arm", "stage", "n", "spearman_res_err_median", "top5_overlap_median", "steep5_err_conc", "steep5_res_conc"]].round(3)
q1.columns = ["Problema", "Brazo", "Etapa", "n", "Spearman residuo–error", "Solape 5 % sup.", "Conc. error (steep5)", "Conc. residuo (steep5)"]
q1.to_csv(HERE / "tables/q1_residuo_vs_error.csv", index=False)

q5 = pd.read_csv(A / "q5_scale_ratio.csv")[["problem", "stage", "scale/ratio_struct_to_spacing_dim0", "struct/width_dim0", "err/all_rms_normalized"]].round(4)
q5.columns = ["Problema", "Etapa", "Razón estructura/espaciado (dim 0)", "Anchura de estructura (dim 0)", "Error normalizado (mediana)"]
q5.to_csv(HERE / "tables/q5_estructura_vs_bases.csv", index=False)

ml = pd.read_csv(A / "loso_ml.csv")
piv = ml.pivot_table(index=["problem", "stage"], columns="model", values="r2").reset_index()
piv = piv[["problem", "stage", "rf", "ridge", "cond_mean"]].round(3)
n = ml[ml.model == "rf"][["problem", "stage", "n"]]
piv = piv.merge(n, on=["problem", "stage"])
piv.columns = ["Problema", "Etapa", "R² bosque (señales)", "R² cresta (señales)", "R² solo condición", "n"]
piv.to_csv(HERE / "tables/loso_ml.csv", index=False)

c = pd.read_csv(A / "candidate_signals.csv")
c = c[c.meets_b][["stage", "signal", "problems", "direction", "mean_stability_tau", "meets_c"]].round(3)
c["tipo"] = c["signal"].apply(lambda s: "variable de presupuesto (confusor)" if s.startswith("train/") else "señal del residuo / estructura")
c.columns = ["Etapa", "Señal", "Problemas donde cumple (b)", "Dirección", "τ medio", "cumple (c)", "Tipo"]
c.to_csv(HERE / "tables/candidatas.csv", index=False)

t = (HERE / "mecanismo-preliminar.template.md").read_text(encoding="utf-8")
for k, df in (("TABLE_BASELINES", base), ("TABLE_Q1", q1), ("TABLE_Q5", q5), ("TABLE_LOSO", piv), ("TABLE_CANDIDATES", c)):
    t = t.replace("{{" + k + "}}", md(df))
(HERE / "mecanismo-preliminar.md").write_text(t, encoding="utf-8")
print("ok", len(t.split()), "palabras")
