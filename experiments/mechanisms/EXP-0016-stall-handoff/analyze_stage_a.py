"""EXP-0016, etapa A: criterios H-HANDOFF-1 (techo de error) y H-HANDOFF-T (techo de costo) sobre los JSON del barrido.

Definiciones del spec v2 (fijadas antes de correr):
  E(k)     = L2 relativo contra la referencia tras L-BFGS desde el traspaso k (el tope de L-BFGS iguala el reloj total del control)
  E_fijo   = E(1.00);  k* = argmin_k E(k) (oraculo, incluye 1.00)
  H-HANDOFF-1 (Burgers): media geometrica de E(k*)/E_fijo <= 0.80  Y  >= 4/5 semillas con razon <= 0.80
  H-HANDOFF-T (Burgers): existe k < 1.00 con E(k) <= 1.05*E_fijo y reloj total <= 0.60*T_control en >= 4/5 semillas
  Allen-Cahn: mismos criterios, replica (n = 3: >= 2/3 semillas), no decide.
Uso: python analyze_stage_a.py [carpeta con los JSON]
"""
import json
import math
import sys
from pathlib import Path

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results/stage-a/stageA")
SEEDS = {"burgers": [301, 302, 303, 304, 305], "ac": [601, 602, 603]}


def err(score):
    return next(iter(score.values()))


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def load(problem, seed):
    return json.loads((DIR / f"{problem}-seed{seed}.json").read_text())


def seed_table(res):
    runs = res["runs"]
    rows = []
    for f in sorted(runs, key=float):
        r = runs[f]
        rows.append(dict(f=float(f), k=r["k"], E=err(r["final_score"]), E_adam=err(r["adam_score_at_k"]), total_s=r["total_seconds"],
                         lbfgs_s=r["lbfgs_seconds"], it=r["lbfgs_iterations"], status=r["lbfgs_status"], cap=r["cap"]))
    return rows


def main():
    out = {}
    for problem, seeds in SEEDS.items():
        print(f"\n=== {problem} ===")
        ratios, t_ok, per_seed = [], [], []
        for s in seeds:
            res = load(problem, s)
            rows = seed_table(res)
            fixed = next(r for r in rows if r["f"] == 1.0)
            best = min(rows, key=lambda r: r["E"])
            ratio = best["E"] / fixed["E"]
            cheap = [r for r in rows if r["f"] < 1.0 and r["E"] <= 1.05 * fixed["E"] and r["total_s"] <= 0.60 * fixed["total_s"]]
            ratios.append(ratio)
            t_ok.append(bool(cheap))
            per_seed.append(dict(seed=s, E_fijo=fixed["E"], k_star=best["k"], E_star=best["E"], ratio=ratio, total_fixed=fixed["total_s"],
                                 cheap_ks=[r["k"] for r in cheap], rows=rows))
            line = " ".join(f"k={r['k']}:{r['E']:.2e}({r['status'][6:10]},{r['lbfgs_s']:.0f}s)" for r in rows)
            print(f" seed {s}: E_fijo {fixed['E']:.3e} | k* {best['k']} ratio {ratio:.3f} | T_control {fixed['total_s']:.0f}s | cheap {per_seed[-1]['cheap_ks']}")
            print("    ", line)
        n_ok = sum(r <= 0.80 for r in ratios)
        need = 4 if problem == "burgers" else 2
        h1 = gm(ratios) <= 0.80 and n_ok >= need
        ht = sum(t_ok) >= need
        print(f" H-HANDOFF-1: GM {gm(ratios):.3f}, {n_ok}/{len(seeds)} seeds <= 0.80 -> {'SE SOSTIENE' if h1 else 'NO se sostiene'}")
        print(f" H-HANDOFF-T: {sum(t_ok)}/{len(seeds)} seeds with a cheap handoff -> {'SE SOSTIENE' if ht else 'NO se sostiene'}")
        out[problem] = dict(ratios=ratios, gm=gm(ratios), n_le_080=n_ok, h1=h1, h_t=ht, n_cheap=sum(t_ok), per_seed=per_seed)
    (DIR.parent / "stage_a_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
