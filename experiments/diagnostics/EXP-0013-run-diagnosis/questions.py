#!/usr/bin/env python3
"""EXP-0013 — tablas descriptivas por pregunta diagnóstica Q1–Q6 (CPU). Lee los CSV consolidados; no recalcula señales.

Salidas: results/analysis/q1..q6_*.csv y results/analysis/questions-summary.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analyze import load
from common import RESULTS, log

OUT = RESULTS / "analysis"
OUT.mkdir(exist_ok=True)


def arm_key(r):
    a = r["arm"]
    return a


def main():
    d = load()
    d["arm_w"] = d["arm"] + d["width"].astype(int).map(lambda w: f"@{w}" if w else "")
    md = ["# Preguntas diagnósticas Q1–Q6 — tablas descriptivas (EXP-0013)", "", "Exploratorio. n = 5 semillas por celda; medianas e intervalos por semilla, no inferencia. `audit/` y `err/` usan la referencia; el resto no.", ""]

    # ---------------- Q1: residuo vs error
    g = d.groupby(["problem", "arm", "stage"])
    q1 = g.agg(n=("id", "count"), spearman_res_err_median=("audit/spearman_res_err", "median"), spearman_res_err_min=("audit/spearman_res_err", "min"),
               spearman_res_err_max=("audit/spearman_res_err", "max"), top5_overlap_median=("audit/top5_overlap", "median"),
               err_mass_in_top5res_median=("audit/err_mass_in_top5res", "median"), res_mass_in_top5err_median=("audit/res_mass_in_top5err", "median"),
               res_gini_median=("res/all/gini", "median"), err_gini_median=("err/gini", "median")).reset_index()
    if "region/steep5/err_concentration" in d.columns:
        q1 = q1.merge(g.agg(steep5_err_conc=("region/steep5/err_concentration", "median"), steep5_res_conc=("region/steep5/res_concentration", "median")).reset_index(), on=["problem", "arm", "stage"])
    q1.to_csv(OUT / "q1_residual_vs_error.csv", index=False)
    md += ["## Q1 — ¿Dónde está el error y dónde está el residuo?", "", "Mediana por (problema, brazo, etapa) del Spearman residuo–error, solape de los 5 % superiores, y concentración (masa / fracción de puntos) del error y del residuo en la región steep5 (5 % de puntos con mayor gradiente de la referencia). Un solape esperado por azar es 0.05.", "",
           q1[["problem", "arm", "stage", "n", "spearman_res_err_median", "top5_overlap_median", "steep5_err_conc", "steep5_res_conc"]].round(3).to_markdown(index=False), ""]

    # ---------------- Q2: capacidad vs optimización
    cols = ["train/adam_loss_end", "train/lbfgs_loss_end", "jac/effective_rank_per_param", "basis/l0/dead_fraction", "param/count", "err/all_rms_normalized"]
    cols = [c for c in cols if c in d.columns]
    q2 = d.groupby(["problem", "campaign", "variant", "arm_w", "stage"])[cols].median().reset_index()
    q2.to_csv(OUT / "q2_capacity_vs_optimization.csv", index=False)
    rho = []
    for (p, st), sub in d.groupby(["problem", "stage"]):
        for lc in ("train/adam_loss_end", "train/lbfgs_loss_end"):
            if lc in sub.columns and sub[lc].notna().sum() >= 8:
                r = stats.spearmanr(sub[lc], sub["err/all_rms_normalized"], nan_policy="omit")[0]
                rho.append({"problem": p, "stage": st, "loss_column": lc, "n": int(sub[lc].notna().sum()), "spearman_loss_vs_error": r})
    pd.DataFrame(rho).to_csv(OUT / "q2_loss_vs_error.csv", index=False)
    md += ["## Q2 — ¿Capacidad u optimización?", "", "Spearman entre la pérdida final registrada y el error (dentro de cada problema y etapa, sin quitar el efecto de brazo ni campaña): si la pérdida ordena el error, la optimización de la pérdida limita el resultado; si no, la pérdida no es un buen indicador del error.", "",
           pd.DataFrame(rho).round(3).to_markdown(index=False) if rho else "(sin datos)", "",
           "Capacidad (RBF): ver `q2_capacity_vs_optimization.csv` (rango efectivo del jacobiano por parámetro, bases muertas, parámetros). Evidencia externa a este estudio: el ajuste directo del RBF de 4 032 parámetros al campo FEM de DFG llega a 1–5 % (`local-20260930/capacity-fit-dfg.json`), por lo que el límite en DFG es de optimización y no de capacidad.", ""]

    # ---------------- Q3: Adam -> L-BFGS (pareado por campaña, brazo, semilla)
    key = ["problem", "cgroup", "variant", "arm_w", "seed"]
    a = d[d["stage"] == "adam_final"].set_index(key)
    b = d[d["stage"] == "lbfgs_final"].set_index(key)
    common = a.index.intersection(b.index)
    t = pd.DataFrame(index=common)
    t["err_ratio_lbfgs_over_adam"] = b.loc[common, "err/all_rms_normalized"] / a.loc[common, "err/all_rms_normalized"]
    for c, nm in (("res/all/rms", "res_rms_ratio"), ("res/all/gini", "res_gini_delta"), ("res/all/top5_mass", "res_top5_delta"), ("audit/spearman_res_err", "spearman_delta"), ("err/gini", "err_gini_delta")):
        if c in d.columns:
            t[nm] = (b.loc[common, c] / a.loc[common, c]) if nm.endswith("ratio") else (b.loc[common, c] - a.loc[common, c])
    t = t.reset_index()
    t.to_csv(OUT / "q3_adam_to_lbfgs_paired.csv", index=False)
    s3 = t.groupby(["problem", "cgroup", "variant", "arm_w"]).agg(n=("seed", "count"), err_ratio_median=("err_ratio_lbfgs_over_adam", "median"), res_rms_ratio=("res_rms_ratio", "median"),
                                                        res_gini_delta=("res_gini_delta", "median"), res_top5_delta=("res_top5_delta", "median"), spearman_delta=("spearman_delta", "median")).reset_index()
    cv = d.groupby(["problem", "cgroup", "variant", "arm_w", "stage"])["err/all_rms_normalized"].agg(lambda x: np.std(x, ddof=1) / np.mean(x) if len(x) > 1 else np.nan).unstack("stage")
    cv.columns = [f"err_cv_{c}" for c in cv.columns]
    s3 = s3.merge(cv.reset_index(), on=["problem", "cgroup", "variant", "arm_w"], how="left")
    s3.to_csv(OUT / "q3_adam_to_lbfgs_summary.csv", index=False)
    md += ["## Q3 — ¿Qué cambia de Adam a L-BFGS?", "", "Pareado por (campaña, brazo, semilla). `err_ratio_median` = error tras L-BFGS / error al final de Adam; el CV es la dispersión relativa entre semillas del error en cada etapa.", "",
           s3.round(4).to_markdown(index=False), ""]

    # ---------------- Q4: modos de falla
    fm = pd.read_csv(RESULTS / "failure_modes.csv").fillna("")
    fm = fm.merge(d[["id"]], on="id")
    rows = []
    for (p, a_, s), g_ in fm.groupby(["problem", "arm", "stage"]):
        c = {}
        for m in g_["modes"]:
            for tg in [x for x in str(m).split(";") if x]:
                c[tg] = c.get(tg, 0) + 1
        rows.append({"problem": p, "arm": a_, "stage": s, "n": len(g_), **{k: v for k, v in sorted(c.items())}})
    q4 = pd.DataFrame(rows).fillna(0)
    q4.to_csv(OUT / "q4_failure_modes.csv", index=False)
    md += ["## Q4 — Modos de falla (reglas fijas en `make_cards.py`; no causales)", "", q4.to_markdown(index=False), "",
           "Nota: `low_jacobian_rank_per_param` (umbral 0.002) se activa en casi todos los modelos grandes y no discrimina; no se usa como evidencia.", ""]

    # ---------------- Q5: escala de estructura vs bases
    c5 = [c for c in d.columns if c.startswith("scale/ratio_struct_to_spacing_dim0") or c.startswith("scale/ratio_struct_to_width_dim0") or c == "struct/width_dim0"]
    rbf = d[(d["kind"] == "rbf")]
    cols = c5 + ["err/all_rms_normalized", "region/steep5/err_concentration"]
    q5 = rbf.groupby(["problem", "stage"])[[c for c in cols if c in rbf.columns]].median().reset_index()
    q5.to_csv(OUT / "q5_scale_ratio.csv", index=False)
    md += ["## Q5 — Anchura de la estructura frente al espaciado de las bases RBF de la capa 0 (dimensión 0 = x)", "",
           "Razón < 1: la estructura del propio modelo es más fina que el espaciado de las bases de la capa 0. Las capas ocultas pueden afinar la estructura (la capa 0 no es el único lugar de resolución); es un indicio, no una prueba.", "", q5.round(4).to_markdown(index=False), ""]

    # ---------------- Q6: ¿qué señales de Adam anticipan la ganancia de L-BFGS?
    sig6 = [c for c in ("train/adam_log_slope_last10", "train/adam_rel_gain_last20pct", "res/all/gini", "res/all/top5_mass", "align/spearman_res_gradnorm", "jac/effective_rank_per_param") if c in d.columns]
    rows = []
    t2 = t.merge(a.reset_index()[key + sig6], on=key, how="left")
    for p, sub in t2.groupby("problem"):
        for c in sig6:
            ok = sub[[c, "err_ratio_lbfgs_over_adam"]].dropna()
            if len(ok) >= 8 and ok[c].nunique() > 2:
                rows.append({"problem": p, "adam_stage_signal": c, "n": len(ok), "spearman_with_lbfgs_gain": stats.spearmanr(ok[c], np.log10(ok["err_ratio_lbfgs_over_adam"]))[0]})
    q6 = pd.DataFrame(rows)
    q6.to_csv(OUT / "q6_adam_signals_vs_lbfgs_gain.csv", index=False)
    md += ["## Q6 — ¿Anticipa alguna señal de Adam cuánto ganará L-BFGS?", "", "Spearman (descriptivo, sin corrección de múltiples pruebas ni bloques de semilla) entre una señal medida al final de Adam y log10(error L-BFGS / error Adam) pareado. Valores negativos = más ganancia de L-BFGS.", "",
           q6.round(3).to_markdown(index=False) if len(q6) else "(sin datos suficientes)", ""]
    (OUT / "questions-summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("questions", tables=6)
    print("ok")


if __name__ == "__main__":
    main()
