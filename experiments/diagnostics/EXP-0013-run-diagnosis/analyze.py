#!/usr/bin/env python3
"""EXP-0013 — análisis estadístico pre-registrado (CPU). Une señales (sin referencia) con etiquetas (resultado) y aplica el criterio fijado en el spec.

Unidad: corrida = (problema, campaña, brazo, semilla, etapa). Dependencia: filas de la misma semilla comparten pools => permutaciones POR BLOQUES de semilla.
Estadístico: correlación de rangos (Spearman) centrada dentro de cada semilla (ρ_w). p por permutación de la señal dentro de cada bloque de semilla (B=5000).
Holm por familia = todas las pruebas (señal) dentro de un (problema, etapa). "Intervalo de permutación que excluye 0" se implementa como p_Holm < 0.05.
Criterio de señal candidata (vinculante, del spec): (a) sin referencia; (b) misma dirección en >= 2 problemas con |ρ_w| >= 0.5 y p_Holm < 0.05;
(c) estabilidad entre semillas: τ de Kendall medio >= 0.5 del ranking de condiciones entre pares de semillas; (d) costo < 10 % del paso (ver cost_bench.py).
Salidas: results/analysis/*.csv, results/analysis/*.md, figuras en figures/.
"""
from __future__ import annotations

import json
import re
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import HERE, RESULTS, log

warnings.filterwarnings("ignore")
OUT = RESULTS / "analysis"
OUT.mkdir(exist_ok=True)
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261005)
B_PERM = 5000
SIG_PREFIX = ("res/", "struct/", "align/", "basis/", "scale/", "jac/", "param/", "gradterm/", "train/", "heat/", "burgers/", "ac/", "dfg/", "piezo/")
COMMON_PREFIX = ("res/all/", "struct/", "align/", "basis/", "scale/", "jac/", "param/", "train/")


def load():
    s = pd.read_csv(RESULTS / "signals_wide.csv")
    l = pd.read_csv(RESULTS / "labels_wide.csv")
    d = s.merge(l.drop(columns=["problem", "campaign", "arm", "width", "seed", "stage"]), on="id", how="inner")
    # EXCLUSIÓN DECLARADA: las réplicas del control de EXP-0011 se corrieron con L-BFGS de 0 s; su "etapa L-BFGS" duplica la etapa Adam.
    dup = (d["campaign"] == "EXP-0011") & (d["arm"] == "rbf(ctl-replicate)") & (d["stage"] == "lbfgs_final")
    log("analysis_exclusion", rows=int(dup.sum()), reason="EXP-0011 ctl-replicate lbfgs_final duplica adam_final (lbfgs-seconds=0)")
    d = d[~dup].copy()
    d["width"] = d["width"].fillna(0)
    d["variant"] = d["path"].str.extract(r"/(corrected|literal)-")[0].fillna("")
    cg = {"cohort-A-lbfgs": "cohort-A", "cohort-A-r3r4-adam": "cohort-A", "cohort-B-pilot-adam": "cohort-B", "cohort-B-lbfgs": "cohort-B",
          "S2-control-adam": "heat-S2-control", "heat-lbfgs-redo": "heat-S2-control"}
    d["cgroup"] = d["campaign"].map(cg).fillna(d["campaign"])
    d["cond"] = d["cgroup"] + "|" + d["variant"] + "|" + d["arm"] + "|" + d["width"].astype(int).astype(str) + "|" + d["stage"]
    d["y"] = np.log10(d["err/all_rms_normalized"].astype(float).clip(lower=1e-12))
    # dimensiones sin nombre propio para comparar entre problemas
    ren = {}
    for c in d.columns:
        m = re.match(r"scale/(ratio_struct_to_spacing|ratio_struct_to_width|spacing_phys|width_phys)_(x)$", c)
        if m:
            ren[c] = f"scale/{m.group(1)}_dim0"
        m = re.match(r"scale/(ratio_struct_to_spacing|ratio_struct_to_width|spacing_phys|width_phys)_(t|y)$", c)
        if m:
            ren[c] = f"scale/{m.group(1)}_dim1"
        m = re.match(r"struct/width_(x)$", c)
        if m:
            ren[c] = "struct/width_dim0"
        m = re.match(r"struct/width_(t|y)$", c)
        if m:
            ren[c] = "struct/width_dim1"
    d = d.rename(columns=ren)
    return d


def signal_cols(d):
    skip = ("train/meta",)
    cols = [c for c in d.columns if c.startswith(SIG_PREFIX) and not c.startswith(skip)]
    return [c for c in cols if pd.api.types.is_numeric_dtype(d[c])]


def within_rank_corr(x, y, groups):
    """ρ_w: Pearson de rangos centrados por semilla. Devuelve (ρ_w, p_perm, n_efectivo)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], groups[ok]
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, int(ok.sum())
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    ug = np.unique(g)
    idx = [np.where(g == u)[0] for u in ug if (g == u).sum() >= 2]
    if len(idx) < 3:
        return np.nan, np.nan, int(ok.sum())

    def center(r):
        c = r.copy()
        for ii in idx:
            c[ii] = r[ii] - r[ii].mean()
        return c

    def stat(rxc):
        a, b = rxc - rxc.mean(), cy - cy.mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float((a * b).sum() / den) if den > 0 else np.nan

    cy = center(ry)
    obs = stat(center(rx))
    cnt = 0
    it = 0
    for it in range(1, B_PERM + 1):
        rp = rx.copy()
        for ii in idx:
            rp[ii] = rx[ii][RNG.permutation(len(ii))]
        v = stat(center(rp))
        if np.isfinite(v) and abs(v) >= abs(obs) - 1e-12:
            cnt += 1
        if cnt >= 250 and it >= 500:      # parada temprana: p claramente > 0.05 (no afecta a Holm de los p pequeños)
            break
    return obs, (cnt + 1) / (it + 1), int(ok.sum())


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m = len(order)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        out[i] = min(1.0, run)
    return out


def stability_tau(sub, col):
    """τ de Kendall medio entre pares de semillas del ranking de condiciones según la señal."""
    piv = sub.pivot_table(index="cond", columns="seed", values=col, aggfunc="mean")
    piv = piv.dropna(how="any", axis=0)
    if piv.shape[0] < 3 or piv.shape[1] < 2:
        return np.nan
    taus = []
    for a, b in combinations(piv.columns, 2):
        t = stats.kendalltau(piv[a], piv[b])[0]
        if np.isfinite(t):
            taus.append(t)
    return float(np.mean(taus)) if taus else np.nan


def main():
    d = load()
    log("analysis_start", n_rows=len(d), problems=sorted(d["problem"].unique().tolist()))
    sigs = signal_cols(d)
    rows = []
    for (prob, stage), sub in d.groupby(["problem", "stage"]):
        groups = sub["seed"].astype(str).values
        y = sub["y"].values
        ps = []
        recs = []
        for c in sigs:
            x = pd.to_numeric(sub[c], errors="coerce").values
            if np.isfinite(x).sum() < 8:
                continue
            rho, p, n = within_rank_corr(x, y, groups)
            recs.append({"problem": prob, "stage": stage, "signal": c, "n": n, "rho_w": rho, "p_perm": p, "stability_tau": stability_tau(sub, c)})
            ps.append(p)
        ph = holm(ps)
        for r, v in zip(recs, ph):
            r["p_holm"] = v
            rows.append(r)
    mat = pd.DataFrame(rows)
    mat["meets_b_cell"] = (mat["rho_w"].abs() >= 0.5) & (mat["p_holm"] < 0.05)
    mat.to_csv(OUT / "matrix_signal_problem.csv", index=False)
    log("analysis_matrix", rows=len(mat), cells_meeting=int(mat["meets_b_cell"].sum()))

    # --- criterio (b): misma dirección en >= 2 problemas (por etapa) ; (c) estabilidad
    cand = []
    for (stage, sig), sub in mat.groupby(["stage", "signal"]):
        good = sub[sub["meets_b_cell"]]
        pos, neg = int((good["rho_w"] > 0).sum()), int((good["rho_w"] < 0).sum())
        n_same = max(pos, neg)
        tau = sub["stability_tau"].mean()
        cand.append({"stage": stage, "signal": sig, "problems_tested": sub["problem"].nunique(), "problems_meeting_b": n_same, "direction": ("+" if pos >= neg else "-") if n_same else "",
                     "problems": ",".join(good["problem"].tolist()), "mean_abs_rho": float(sub["rho_w"].abs().mean()), "mean_stability_tau": tau,
                     "meets_b": n_same >= 2, "meets_c": bool(np.isfinite(tau) and tau >= 0.5)})
    cand = pd.DataFrame(cand)
    cand["meets_b_and_c"] = cand["meets_b"] & cand["meets_c"]
    cand = cand.sort_values(["meets_b_and_c", "meets_b", "problems_meeting_b", "mean_abs_rho"], ascending=False)
    cand.to_csv(OUT / "candidate_signals.csv", index=False)
    log("analysis_candidates", meets_b=int(cand["meets_b"].sum()), meets_b_and_c=int(cand["meets_b_and_c"].sum()))
    print(cand.head(25).to_string())
    return d, mat, cand


if __name__ == "__main__":
    main()
