"""EXP-0027 (spec.md). Solo lectura de EXP-0026. Uso: python analyze_seed_predictor.py --dir RUTA/results/exp0026 [--out analysis.json]"""
import argparse, glob, json, re
from pathlib import Path
import numpy as np

SEEDS_A = list(range(3001, 3011)); SEEDS_B = [1, 2, 3, 4, 5, 101, 102, 103, 104, 105]

def load(d):
    R = {}
    for f in glob.glob(str(Path(d) / "burgers-*-seed*.json")):
        r = json.load(open(f)); R[(r["arm"], r["seed"])] = r
    return R

def loss_at(r, step):
    for h in r["hist"]:
        if h["step"] == step: return h["loss"]
    return None

def ranks(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r

def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0: return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])

def perm_p(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed); obs = abs(spearman(a, b)); b = np.asarray(b)
    cnt = sum(abs(spearman(a, rng.permutation(b))) >= obs - 1e-12 for _ in range(n))
    return (cnt + 1) / (n + 1)

def analyze(R, arm, step):
    seeds = sorted(SEEDS_A + SEEDS_B); miss = [s for s in seeds if ("B", s) not in R or (arm, s) not in R]
    if miss: return dict(arm=arm, error=f"semillas ausentes: {miss}")
    c = np.array([np.log10(loss_at(R[("B", s)], step)) for s in seeds])
    y = np.array([np.log(R[(arm, s)]["primary_final"] / R[("B", s)]["primary_final"]) for s in seeds])
    # verificacion: trayectoria compartida hasta la primera intervencion
    dev = [abs(loss_at(R[(arm, s)], step) / loss_at(R[("B", s)], step) - 1) for s in seeds]
    rho = spearman(c, y); p = perm_p(c, y)
    ia = [i for i, s in enumerate(seeds) if s in SEEDS_A]; ib = [i for i, s in enumerate(seeds) if s in SEEDS_B]
    rho_a, rho_b = spearman(c[ia], y[ia]), spearman(c[ib], y[ib])
    keep = np.argsort(y)[:-2]; rho_l2 = spearman(c[keep], y[keep])
    sg = lambda v: np.sign(v)
    cond = dict(c1_abs_rho_ge_0_50=abs(rho) >= 0.50, c2_p_le_0_05=p <= 0.05,
                c3_same_sign_both_sets=bool(sg(rho_a) == sg(rho_b) == sg(rho)),
                c4_leave2out=bool(abs(rho_l2) >= 0.40 and sg(rho_l2) == sg(rho)))
    return dict(arm=arm, step=step, n=len(seeds), rho=rho, p_perm=p, rho_3001_3010=rho_a, rho_1_5_101_105=rho_b, rho_drop_top2_y=rho_l2,
                max_rel_dev_shared_trajectory=float(max(dev)), conditions=cond, supported=bool(all(cond.values())),
                y_per_seed={str(s): round(float(v), 4) for s, v in zip(seeds, y)})

def exploratory(R, arm):
    seeds = sorted(SEEDS_A + SEEDS_B)
    y = np.array([np.log(R[(arm, s)]["primary_final"] / R[("B", s)]["primary_final"]) for s in seeds])
    pa = np.array([np.log10(R[("B", s)]["primary_adam"]) for s in seeds])
    sl = np.array([np.log10(loss_at(R[("B", s)], 500)) - np.log10(loss_at(R[("B", s)], 300)) for s in seeds])
    return dict(EXPLORATORIO=True, rho_y_vs_log_primary_adam_B=spearman(pa, y), rho_y_vs_slope_log_loss_300_500_B=spearman(sl, y))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", default=None); a = ap.parse_args()
    R = load(a.dir)
    p1 = analyze(R, "E500", 500); p2 = analyze(R, "E1000L", 1000)
    if "error" not in p2 and "error" not in p1:
        p2["consistent_with_P1"] = bool(np.sign(p2["rho"]) == np.sign(p1["rho"]) and abs(p2["rho"]) >= 0.40)
    res = dict(P1_E500=p1, P2_E1000L=p2, exploratorio=exploratory(R, "E500") if "error" not in p1 else None)
    s = json.dumps(res, indent=1); print(s)
    if a.out: Path(a.out).write_text(s)

if __name__ == "__main__": main()
