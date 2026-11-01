"""EXP-0025: analisis preregistrado del bloque. Aplica LITERALMENTE las hipotesis y guardas del spec (spec.md); no ajusta umbrales.
r_X = mediana, sobre semillas, de error_final(X)/error_final(B) pareado por semilla (grader de evals/ via `primary_final`).
H-OBS-1: por problema r_O <= min(1, r_R, r_E) + 0.10 y ninguna semilla de O > 2x B; ademas r_O <= 0.90 en al menos un problema.
H-OBS-2: r_O <= 0.90 * r_comparador en al menos un problema (O- uniforme; si O elige a0 en mas de la mitad de las decisiones del problema, el comparador es O-m).
H-OBS-3: razon O/B en la region de interfaz (5 % de mayor |u_x| de la referencia) <= razon en el resto - 0.10 en >= 4 de 5 semillas.
Guardas: error al final de Adam, media geometrica de O/B <= 1.25; reloj O <= 1.5x B; 5/5 con perdida finita; region critica: ninguna semilla con O/B > 1.25 en la interfaz;
cola: ninguna semilla de O > 2x B. Uso: python analyze_block.py --dir DIR [--out analisis.json]"""
import argparse, glob, json
from pathlib import Path
import numpy as np

EPS, MARGIN, DELTA_WIN, GAMMA, TAIL, CLOCK = 0.10, 0.10, 0.90, 0.25, 2.0, 1.5


def load(d):
    R = {}
    for f in glob.glob(str(Path(d) / "*-seed*.json")):
        name = Path(f).name
        if name in ("ledger.json", "progress.json"):
            continue
        r = json.load(open(f))
        R[(r["problem"], r["arm"], r["seed"])] = r
    return R


def ratios(R, p, arm, key="primary_final"):
    out = {}
    for (pp, a, s), r in R.items():
        if pp == p and a == arm and (p, "B", s) in R and r.get(key) is not None and R[(p, "B", s)].get(key) is not None:
            out[s] = r[key] / R[(p, "B", s)][key]
    return out


def med(x):
    return float(np.median(list(x.values()))) if x else None


def gm(x):
    return float(np.exp(np.mean(np.log(list(x.values()))))) if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = load(a.dir)
    probs = sorted({k[0] for k in R})
    res = dict(problems={})
    h1, h2, h3, guards = {}, {}, {}, {}
    for p in probs:
        seeds = sorted({k[2] for k in R if k[0] == p})
        rr = {arm: ratios(R, p, arm) for arm in ("R", "E", "O", "Orand", "Om")}
        a0_frac = None
        ch = [c for (pp, arm, s), r in R.items() if pp == p and arm == "O" for c in r["observer"]["chosen"]]
        if ch:
            a0_frac = sum(c == "a0" for c in ch) / len(ch)
        comp_name = "Om" if (a0_frac is not None and a0_frac > 0.5 and rr["Om"]) else "Orand"
        rO, rR, rE, rC = med(rr["O"]), med(rr["R"]), med(rr["E"]), med(rr[comp_name])
        best_fixed = min(v for v in (1.0, rR, rE) if v is not None)
        h1[p] = dict(r_O=rO, r_R=rR, r_E=rE, bound=best_fixed + EPS, passes=(rO is not None and rO <= best_fixed + EPS),
                     max_seed_ratio_O=max(rr["O"].values()) if rr["O"] else None, tail_ok=(max(rr["O"].values()) <= TAIL) if rr["O"] else None)
        h2[p] = dict(comparator=comp_name, r_comp=rC, r_Orand=med(rr["Orand"]), r_Om=med(rr["Om"]), passes=(rO is not None and rC is not None and rO <= DELTA_WIN * rC))
        reg = {}
        for s in seeds:
            O, B = R.get((p, "O", s)), R.get((p, "B", s))
            if O and B and "region_final" in O and "region_final" in B:
                reg[s] = (O["region_final"]["l2re_top"] / B["region_final"]["l2re_top"], O["region_final"]["l2re_rest"] / B["region_final"]["l2re_rest"])
        h3[p] = dict(per_seed_top_rest=reg, n_pass=sum(t <= r - MARGIN for t, r in reg.values()), n=len(reg))
        adam = ratios(R, p, "O", "primary_adam")
        clock = {s: R[(p, "O", s)]["total_seconds"] / R[(p, "B", s)]["total_seconds"] for s in seeds if (p, "O", s) in R and "total_seconds" in R[(p, "O", s)] and "total_seconds" in R[(p, "B", s)]}
        adam_clock = {s: R[(p, "O", s)]["adam_seconds"] / R[(p, "B", s)]["adam_seconds"] for s in seeds if (p, "O", s) in R}
        guards[p] = dict(adam_end_geomean_O_over_B=gm(adam), adam_end_ok=(gm(adam) is not None and gm(adam) <= 1 + GAMMA), clock_total_median=med(clock), clock_adam_median=med(adam_clock),
                         clock_ok=(med(clock) is not None and med(clock) <= CLOCK), finite_all=all(R[(p, "O", s)].get("finite_params", False) for s in seeds if (p, "O", s) in R),
                         region_top_worst_O_over_B=max((t for t, _ in reg.values()), default=None), region_ok=all(t <= 1 + GAMMA for t, _ in reg.values()),
                         inert_O_a0_fraction=a0_frac, O_chose=dict(zip(*np.unique(ch, return_counts=True))) if ch else None)
        guards[p]["O_chose"] = {k: int(v) for k, v in guards[p]["O_chose"].items()} if guards[p]["O_chose"] else None
        res["problems"][p] = dict(seeds=seeds, medians={k: med(v) for k, v in rr.items()}, per_seed={k: {str(s): round(x, 4) for s, x in v.items()} for k, v in rr.items()})
    any_win = any(h1[p]["r_O"] is not None and h1[p]["r_O"] <= DELTA_WIN for p in probs)
    res["H-OBS-1"] = dict(per_problem=h1, win_in_some_problem=any_win,
                          verdict="NO RECHAZADA" if all(h1[p]["passes"] and h1[p]["tail_ok"] for p in probs) and any_win and all(g["adam_end_ok"] and g["clock_ok"] and g["finite_all"] and g["region_ok"] for g in guards.values()) else "RECHAZADA")
    res["H-OBS-2"] = dict(per_problem=h2, verdict="NO RECHAZADA" if any(h2[p]["passes"] for p in probs) else "RECHAZADA")
    res["H-OBS-3"] = dict(per_problem=h3, verdict="RESPALDADA" if all(v["n"] and v["n_pass"] >= 4 for v in h3.values()) else "NO RESPALDADA")
    res["guards"] = guards
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8")


if __name__ == "__main__":
    main()
