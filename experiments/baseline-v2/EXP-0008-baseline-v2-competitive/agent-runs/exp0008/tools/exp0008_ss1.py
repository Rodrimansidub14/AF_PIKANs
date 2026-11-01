"""SS-1 (signal gate) on dense feature grids + label grids. CPU only, unit of analysis = seed.

Models: M0 residual | M1 + geometry | M2 + derivative signals | M3 + RBF/network signals.
Leave-one-seed-out ridge regression on log|error|; metric = AUC of the top error decile in the held-out seed.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

EPS = 1e-12


def auc(score, positive):
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n1 = positive.sum(); n0 = len(score) - n1
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def design(d, model):
    cols = [np.log(np.abs(d["residual"]) + EPS)]
    if model >= 1:
        dist = d.get("interface_distance", np.zeros_like(d["x"]))
        dist = np.where(np.isfinite(dist), dist, np.nanmax(dist[np.isfinite(dist)]) if np.isfinite(dist).any() else 0.0)
        cols += [d["t"], np.abs(d["x"]), np.log(dist + 1e-3)]
    if model >= 2:
        cols += [np.log(np.abs(d["u_x"]) + EPS), np.log(np.abs(d["u_xx"]) + EPS), np.log(d["grad_residual_norm"] + EPS)]
    if model >= 3:
        for k in ("basis0_entropy", "basis0_peak", "saturation_l0", "saturation_l1"):
            if k in d: cols.append(d[k])
        cols.append(np.log(d["sensitivity"] + EPS))
    return np.stack(cols, 1)


def ridge_fit(X, y, lam=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-12; Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Xn: ((Xn - mu) / sd) @ w + y.mean()


def load(run_dirs, tag):
    out = []
    for rd in run_dirs:
        f = Path(rd) / f"dense_signals_{tag}.npz"; l = Path(rd) / f"labels_{tag}.npz"
        if not (f.exists() and l.exists()): continue
        d = dict(np.load(f)); d["error"] = np.load(l)["abs_error"]; out.append(d)
    return out


def analyse(data, subset_only_for_m3=True):
    n = len(data); res = {"n_seeds": n, "spearman_residual_error": [], "auc": {f"M{m}": [] for m in range(4)}}
    for d in data:
        res["spearman_residual_error"].append(float(spearmanr(np.abs(d["residual"]), d["error"])[0]))
    for hold in range(n):
        tr = [d for i, d in enumerate(data) if i != hold]; te = data[hold]
        for m in range(4):
            if m == 3:
                keep_tr = [np.isfinite(d["sensitivity"]) for d in tr]; keep_te = np.isfinite(te["sensitivity"])
            else:
                keep_tr = [np.ones(len(d["x"]), bool) for d in tr]; keep_te = np.ones(len(te["x"]), bool)
            X = np.concatenate([design(d, m)[k] for d, k in zip(tr, keep_tr)])
            y = np.concatenate([np.log(d["error"][k] + EPS) for d, k in zip(tr, keep_tr)])
            f = ridge_fit(X, y); Xt = design(te, m)[keep_te]; et = te["error"][keep_te]
            res["auc"][f"M{m}"].append(auc(f(Xt), et >= np.quantile(et, 0.9)))
        # M2 restricted to the M3 evaluation subset for a like-for-like delta
        keep = np.isfinite(te["sensitivity"]); X = np.concatenate([design(d, 2)[np.isfinite(d["sensitivity"])] for d in tr])
        y = np.concatenate([np.log(d["error"][np.isfinite(d["sensitivity"])] + EPS) for d in tr])
        f = ridge_fit(X, y); et = te["error"][keep]
        res["auc"].setdefault("M2_on_M3_subset", []).append(auc(f(design(te, 2)[keep]), et >= np.quantile(et, 0.9)))
    A = {k: np.array(v) for k, v in res["auc"].items()}
    rng = np.random.default_rng(0)

    def ci(delta):
        boots = [rng.choice(delta, len(delta)).mean() for _ in range(4000)]
        return {"mean": float(delta.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], "per_seed": delta.tolist()}

    res["delta_auc"] = {"M1-M0": ci(A["M1"] - A["M0"]), "M2-M1": ci(A["M2"] - A["M1"]), "M3-M2(subset)": ci(A["M3"] - A["M2_on_M3_subset"])}
    res["auc_mean"] = {k: float(v.mean()) for k, v in A.items()}
    res["spearman_mean"] = float(np.mean(res["spearman_residual_error"]))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--tag", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); data = load(a.runs, a.tag)
    r = analyse(data); Path(a.out).write_text(json.dumps(r, indent=1)); print(json.dumps({k: r[k] for k in ("n_seeds", "spearman_mean", "auc_mean")}, indent=1))
    print(json.dumps({k: (v["mean"], v["ci95"]) for k, v in r["delta_auc"].items()}))
