"""EXP-0026: analisis preregistrado (spec.md). Contrastes pareados por semilla; sin ajuste de umbrales. Uso: python analyze_diag.py --dir DIR [--out analisis.json]"""
import argparse, glob, json
from pathlib import Path
import numpy as np

ARMS = ("B", "E1000L", "E1000", "E500L", "E500")


def load(d, problem):
    R = {}
    for f in glob.glob(str(Path(d) / f"{problem}-*-seed*.json")):
        r = json.load(open(f))
        if r.get("primary_final") is not None and r["seed"] >= 3001:     # factorial: semillas 3001-3010 (la replica cruzada se analiza aparte)
            R[(r["arm"], r["seed"])] = r["primary_final"]
    return R


def paired(R, a, b):
    s = sorted({k[1] for k in R if k[0] == a} & {k[1] for k in R if k[0] == b})
    return s, [R[(a, x)] / R[(b, x)] for x in s]


def contrast(R, a, b, name, rule=True):
    s, q = paired(R, a, b)
    if not q:
        return dict(contrast=name, n=0)
    out = dict(contrast=name, n=len(q), median=float(np.median(q)), geomean=float(np.exp(np.mean(np.log(q)))), a_better=int(sum(x < 1 for x in q)),
               per_seed={str(x): round(v, 4) for x, v in zip(s, q)})
    if rule:
        out["supported"] = bool(out["median"] <= 0.90 and out["a_better"] >= 7 and len(q) == 10)
    return out


def factorial(R):
    res = {"H-DX-1": contrast(R, "E1000", "E1000L", "e(E1000)/e(E1000L)"), "H-DX-2": contrast(R, "E500", "E1000", "e(E500)/e(E1000)"),
           "desc_cadence_with_refresh": contrast(R, "E500L", "E1000L", "e(E500L)/e(E1000L)", rule=False),
           "desc_refresh_at_500": contrast(R, "E500L", "E500", "e(E500L)/e(E500)", rule=False), "vs_B": {}}
    for arm in ARMS[1:]:
        c = contrast(R, arm, "B", f"e({arm})/e(B)", rule=False)
        if c["n"]:
            q = list(c["per_seed"].values())
            c["tail_over_2x"] = int(sum(x > 2 for x in q))
            c["max_ratio"] = float(max(q))
            c["stable"] = bool(c["median"] <= 0.90 and c["a_better"] >= 8 and c["tail_over_2x"] == 0 and c["n"] == 10)
        res["vs_B"][arm] = c
    return res


def replay(d):
    """Replica cruzada (v1.1-A), Burgers: semillas 1-5 (1-B T1) y 101-105 (1-B V); brazos B, E1000L, E500."""
    R = {}
    for f in glob.glob(str(Path(d) / "burgers-*-seed*.json")):
        r = json.load(open(f))
        if r.get("primary_final") is not None and r["seed"] < 3001:
            R[(r["arm"], r["seed"])] = r["primary_final"]
    out = {}
    for tag, seeds in (("T1_seeds_1_5", range(1, 6)), ("V_seeds_101_105", range(101, 106))):
        sub = {k: v for k, v in R.items() if k[1] in seeds}
        blk = {}
        for arm in ("E1000L", "E500"):
            c = contrast(sub, arm, "B", f"e({arm})/e(B)", rule=False)
            blk[arm] = c
        out[tag] = blk
    v, t = out["V_seeds_101_105"]["E1000L"], out["T1_seeds_1_5"]["E1000L"]
    out["reproduces_V"] = bool(v.get("n") == 5 and v["median"] >= 2.0 and sum(x > 1 for x in v["per_seed"].values()) >= 3)
    out["reproduces_T1"] = bool(t.get("n") == 5 and t["median"] <= 0.90)
    e5 = out["V_seeds_101_105"]["E500"]
    out["E500_improves_on_V_seeds"] = bool(e5.get("n") == 5 and e5["median"] <= 0.90 and e5["a_better"] >= 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {p: factorial(load(a.dir, p)) for p in ("burgers", "ac")}
    res["replay_burgers"] = replay(a.dir)
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8")


if __name__ == "__main__":
    main()
