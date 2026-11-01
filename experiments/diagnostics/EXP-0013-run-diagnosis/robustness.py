#!/usr/bin/env python3
"""EXP-0013 — comprobación de robustez POST-HOC (no pre-registrada): la misma correlación de rangos, pero centrada y permutada DENTRO de cada condición
(campaña|variante|brazo|ancho|etapa), es decir, solo entre semillas de una misma condición. Elimina la confusión por presupuesto/campaña/brazo.
Se aplica a las señales que cumplen (b) y (b)+(c) en el análisis pre-registrado. NO cambia el veredicto del criterio; lo matiza.
"""
import numpy as np, pandas as pd
from analyze import load, within_rank_corr, holm
from common import RESULTS, log

d = load()
c = pd.read_csv(RESULTS / "analysis" / "candidate_signals.csv")
mat = pd.read_csv(RESULTS / "analysis" / "matrix_signal_problem.csv")
sigs = sorted(set(c[c["meets_b"]]["signal"]))
rows = []
for (prob, stage), sub in d.groupby(["problem", "stage"]):
    y = sub["y"].values
    g = sub["cond"].values
    for s in sigs:
        if s not in sub.columns:
            continue
        x = pd.to_numeric(sub[s], errors="coerce").values
        rho, p, n = within_rank_corr(x, y, g)
        base = mat[(mat.problem == prob) & (mat.stage == stage) & (mat.signal == s)]
        rows.append({"problem": prob, "stage": stage, "signal": s, "n": n, "rho_within_condition": rho, "p_perm": p,
                     "rho_prereg(seed-blocks)": float(base["rho_w"].iloc[0]) if len(base) else np.nan})
r = pd.DataFrame(rows)
r["p_holm_within_signals"] = np.nan
for (prob, stage), sub in r.groupby(["problem", "stage"]):
    r.loc[sub.index, "p_holm_within_signals"] = holm(sub["p_perm"].values)
r.to_csv(RESULTS / "analysis" / "robustness_within_condition.csv", index=False)
log("robustness", rows=len(r))
pd.set_option("display.width", 250)
print(r.round(3).sort_values(["signal", "problem"]).to_string())
