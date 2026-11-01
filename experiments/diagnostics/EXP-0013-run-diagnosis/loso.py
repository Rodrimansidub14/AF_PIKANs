#!/usr/bin/env python3
"""EXP-0013 — validación dejando una semilla fuera (leave-one-seed-out) pedida por el spec.
(1) Estabilidad de la dirección de cada señal candidata (cumple (b)) al excluir cada semilla.
(2) ML básico (Ridge y bosque aleatorio) por (problema, etapa): predice log10 del error con señales SIN referencia, comparado con una línea base que solo conoce la condición
    (media de la condición en el entrenamiento). Importancia por permutación en el pliegue de validación. Sin selección de características con datos de validación.
Excluye variables de presupuesto/diseño (train/*) para que el modelo no use la identidad de la campaña.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from analyze import load, signal_cols, within_rank_corr
from common import RESULTS, log

d = load()
cand = pd.read_csv(RESULTS / "analysis" / "candidate_signals.csv")
sigs_b = sorted(set(cand[cand["meets_b"]]["signal"]))
OUT = RESULTS / "analysis"

# ---- (1) estabilidad de la dirección al dejar una semilla fuera
rows = []
for (prob, stage), sub in d.groupby(["problem", "stage"]):
    seeds = sorted(sub["seed"].unique())
    for s in sigs_b:
        if s not in sub.columns:
            continue
        full, _, _ = within_rank_corr(pd.to_numeric(sub[s], errors="coerce").values, sub["y"].values, sub["seed"].astype(str).values)
        if not np.isfinite(full):
            continue
        signs = []
        for k in seeds:
            f = sub[sub["seed"] != k]
            r, _, _ = within_rank_corr(pd.to_numeric(f[s], errors="coerce").values, f["y"].values, f["seed"].astype(str).values)
            signs.append(np.sign(r) if np.isfinite(r) else 0)
        same = int(sum(1 for z in signs if z == np.sign(full)))
        rows.append({"problem": prob, "stage": stage, "signal": s, "rho_full": full, "folds": len(seeds), "folds_same_sign": same, "direction_holds_loso": same == len(seeds)})
r1 = pd.DataFrame(rows)
r1.to_csv(OUT / "loso_direction.csv", index=False)

# ---- (2) ML con LOSO
sigcols = [c for c in signal_cols(d) if not c.startswith("train/") and c != "jac/cond"]   # jac/cond: valores numéricos sin significado (ver catálogo)
rows2, imps = [], []
for (prob, stage), sub in d.groupby(["problem", "stage"]):
    X = sub[sigcols].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().mean() > 0.95]
    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] < 3 or sub["seed"].nunique() < 4 or len(sub) < 10:
        continue
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0.0)
    y = sub["y"].values
    seeds = sub["seed"].values
    cond = sub["cond"].values
    preds = {"ridge": np.zeros(len(y)), "rf": np.zeros(len(y)), "cond_mean": np.zeros(len(y))}
    for k in np.unique(seeds):
        te = seeds == k
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        xtr, xte = np.nan_to_num(sc.transform(X[tr])), np.nan_to_num(sc.transform(X[te]))
        rd = Ridge(alpha=10.0).fit(xtr, y[tr])
        preds["ridge"][te] = rd.predict(xte)
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(xtr, y[tr])
        preds["rf"][te] = rf.predict(xte)
        mu = pd.Series(y[tr]).groupby(cond[tr]).mean()
        preds["cond_mean"][te] = [mu.get(c, y[tr].mean()) for c in cond[te]]
        # importancia por permutación en el pliegue (RF)
        base = np.mean((rf.predict(xte) - y[te]) ** 2)
        for j, col in enumerate(X.columns):
            xp = xte.copy()
            xp[:, j] = np.random.default_rng(j).permutation(xp[:, j])
            imps.append({"problem": prob, "stage": stage, "fold_seed": k, "signal": col, "delta_mse": float(np.mean((rf.predict(xp) - y[te]) ** 2) - base)})
    for m, p in preds.items():
        rows2.append({"problem": prob, "stage": stage, "model": m, "n": len(y), "features": X.shape[1], "spearman_pred_true": stats.spearmanr(p, y)[0],
                      "r2": 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2), "rmse_log10": float(np.sqrt(np.mean((p - y) ** 2)))})
r2 = pd.DataFrame(rows2)
r2.to_csv(OUT / "loso_ml.csv", index=False)
im = pd.DataFrame(imps)
if len(im):
    agg = im.groupby(["problem", "stage", "signal"]).agg(mean_delta_mse=("delta_mse", "mean"), folds_positive=("delta_mse", lambda x: int((x > 0).sum())), folds=("delta_mse", "count")).reset_index()
    agg = agg.sort_values(["problem", "stage", "mean_delta_mse"], ascending=[True, True, False])
    agg.groupby(["problem", "stage"]).head(8).to_csv(OUT / "loso_ml_importance_top8.csv", index=False)
log("loso", direction_rows=len(r1), ml_rows=len(r2))
pd.set_option("display.width", 250)
print(r1.round(3).to_string())
print(r2.round(3).to_string())
