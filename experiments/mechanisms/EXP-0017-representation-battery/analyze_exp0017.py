"""EXP-0017: criterios del spec firmado sobre los 40 estados calculados por run_exp0017.py.

Parte 1 (H-LOC-1). Por problema, senal S en {S1, S2, S3, S5} (S4 no computable: no se registro la trayectoria de la capa 1) y metrica en {Spearman, solape 5 %}:
  diferencia pareada por estado d = metrica(S) - metrica(R); estadistico = mediana de d; p unilateral por permutacion con signo (10 000, signo invertido por
  estado: el spec habla de permutacion "con signo dentro de semilla", y 10 000 permutaciones aleatorias solo tienen sentido con inversion por estado; los estados de
  una misma semilla comparten la trayectoria de Adam, asi que el p es optimista; se reporta tambien la prueba exacta con inversion por SEMILLA, que es la conservadora).
  Holm sobre 5 senales x 2 metricas x 2 problemas = 20 pruebas (las 4 de S4 entran con p = 1). H-LOC-1 se sostiene para S si en AMBOS problemas y en AMBAS metricas
  la mediana de S supera a la del residuo y el p ajustado < 0.05.
Parte 2. H-ATTR-1: por problema, mediana (sobre pares) de Spearman(A, Delta) >= 0.30 y > la del control de ocupacion O, con la diferencia pareada significativa
  (permutacion con signo, p < 0.05), en ambos problemas. H-ATTR-2: mediana del reparto del 20 % superior de A >= 0.50, en ambos problemas.
Verificacion previa: el residuo debe reproducir los valores del EXP-0016 en cada estado.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("exp0017_stats", str(HERE / "stats.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

STATES = HERE / "results/states"
B16 = json.loads((HERE.parent / "EXP-0016-stall-handoff/results/stage-a/stage_b_analysis.json").read_text())
SIGNALS = ["S1", "S2", "S3", "S4", "S5"]
METRICS = ["sp", "ov"]
PROBLEMS = {"burgers": [301, 302, 303, 304, 305], "ac": [601, 602, 603]}


def load(problem):
    rows = []
    for seed in PROBLEMS[problem]:
        for f in ["0.25", "0.4", "0.6", "0.8", "1.0"]:
            rows.append(json.loads((STATES / f"{problem}-{seed}-{f}.json").read_text()))
    return rows


def exact_seed_flip_p(d, seeds):
    """Prueba exacta unilateral invirtiendo el signo de TODAS las diferencias de una semilla a la vez (la unidad es la semilla)."""
    d, seeds = np.asarray(d), np.asarray(seeds)
    uniq = sorted(set(seeds.tolist()))
    obs = np.median(d)
    cnt = 0
    for flips in itertools.product([-1.0, 1.0], repeat=len(uniq)):
        sign = np.array([flips[uniq.index(s)] for s in seeds])
        cnt += np.median(sign * d) >= obs - 1e-15
    return cnt / 2 ** len(uniq)


def main():
    data = {p: load(p) for p in PROBLEMS}
    out = {"repro": {}, "part1": {}, "part2": {}}

    # --- verificacion previa ---
    worst = 0.0
    for p, rows in data.items():
        prev = {(r["seed"], r["f"]): r for r in B16[p]["rows"]}
        for r in rows:
            q = prev[(r["seed"], r["f"])]
            worst = max(worst, abs(r["repro_residual"]["sp"] - q["sp_res"]), abs(r["repro_residual"]["ov"] - q["ov_res"]))
    out["repro"]["max_abs_diff_vs_exp0016"] = worst
    print(f"Reproduccion del residuo del EXP-0016 en los 40 estados: diferencia maxima {worst:.2e} -> {'OK' if worst < 1e-6 else 'NO COINCIDE'}")

    # --- parte 1 ---
    tests = []
    for p, rows in data.items():
        seeds = [r["seed"] for r in rows]
        for s in SIGNALS:
            for m in METRICS:
                if s == "S4":
                    tests.append((p, s, m, None, None, None, None, 1.0, 1.0))
                    continue
                vs = np.array([r["metrics"][s][m] for r in rows])
                vr = np.array([r["metrics"]["R"][m] for r in rows])
                d = vs - vr
                tests.append((p, s, m, float(np.median(vs)), float(np.median(vr)), float(np.median(d)), float(np.mean(d > 0)),
                              st.sign_flip_pvalue(d), exact_seed_flip_p(d, seeds)))
    holm_p = st.holm([t[7] for t in tests])
    seed_holm_p = st.holm([t[8] for t in tests])
    table = {}
    for t, hp, hp2 in zip(tests, holm_p, seed_holm_p):
        p, s, m = t[:3]
        table.setdefault(p, {}).setdefault(s, {})[m] = dict(median_signal=t[3], median_residual=t[4], median_diff=t[5], frac_states_better=t[6],
                                                          p=t[7], p_holm=float(hp), p_seed_exact=t[8], p_seed_exact_holm=float(hp2))
    out["part1"]["table"] = table
    print("\nPARTE 1 (mediana de la señal | mediana del residuo | p Holm por estado | p Holm exacto por semilla)")
    holds = {}
    for s in SIGNALS:
        ok = True
        for p in PROBLEMS:
            for m in METRICS:
                c = table[p][s][m]
                if s == "S4":
                    ok = False
                    continue
                better = c["median_signal"] > c["median_residual"] and c["p_holm"] < 0.05
                ok = ok and better
        holds[s] = ok
    for p in PROBLEMS:
        print(f" {p}:")
        for s in SIGNALS:
            if s == "S4":
                print("   S4: no computable (no se registró la trayectoria de la capa 1)")
                continue
            line = []
            for m in METRICS:
                c = table[p][s][m]
                line.append(f"{m} {c['median_signal']:+.3f} vs R {c['median_residual']:+.3f} (p_Holm {c['p_holm']:.3f}; semilla {c['p_seed_exact_holm']:.3f})")
            print(f"   {s}: " + " | ".join(line))
    out["part1"]["H_LOC_1_holds"] = holds
    print("\n H-LOC-1 por señal:", {k: ("SE SOSTIENE" if v else "no") for k, v in holds.items()})
    # mejor señal aunque ninguna pase: la de mayor mediana de Spearman media entre problemas
    best = max((s for s in SIGNALS if s != "S4"), key=lambda s: np.mean([table[p][s]["sp"]["median_signal"] for p in PROBLEMS]))
    out["part1"]["best_signal_by_spearman"] = best
    print(f" Mejor señal por Spearman medio entre problemas (descriptivo): {best}")
    abs_line = {s: {p: float(np.median([abs(r["metrics"][s]["sp"]) for r in data[p]])) for p in PROBLEMS} for s in SIGNALS if s != "S4"}
    out["part1"]["median_abs_spearman"] = abs_line
    print(" |Spearman| mediano (descriptivo, ignora el signo):", {s: {p: round(v, 3) for p, v in d.items()} for s, d in abs_line.items()})

    # --- parte 2 ---
    print("\nPARTE 2")
    h1_all, h2_all = True, True
    for p, rows in data.items():
        spA = np.array([r["attr"]["sp_A"] for r in rows])
        spO = np.array([r["attr"]["sp_O"] for r in rows])
        spAs = np.array([r["attr"]["sp_Astar"] for r in rows])
        share = np.array([r["attr"]["top20_share_A"] for r in rows])
        sharestar = np.array([r["attr"]["top20_share_Astar"] for r in rows])
        d = spA - spO
        pv = st.sign_flip_pvalue(d)
        pv_seed = exact_seed_flip_p(d, [r["seed"] for r in rows])
        h1 = bool(np.median(spA) >= 0.30 and np.median(spA) > np.median(spO) and pv < 0.05)
        h2 = bool(np.median(share) >= 0.50)
        h1_all, h2_all = h1_all and h1, h2_all and h2
        eff = [r["effect"]["sp_A_vs_drop"] for r in rows if r["effect"]["error_dropped"] and r["effect"]["sp_A_vs_drop"] is not None]
        by_layer = np.median(np.array([r["attr"]["sp_A_by_layer"] for r in rows]), axis=0)
        out["part2"][p] = dict(median_sp_A=float(np.median(spA)), median_sp_O=float(np.median(spO)), median_sp_Astar=float(np.median(spAs)),
                               p_A_vs_O=pv, p_A_vs_O_seed_exact=pv_seed, H_ATTR_1=h1, median_top20_share_A=float(np.median(share)),
                               median_top20_share_Astar=float(np.median(sharestar)), H_ATTR_2=h2, effect_median_sp_A_vs_drop=float(np.median(eff)) if eff else None,
                               effect_n_pairs=len(eff), median_sp_A_by_layer=by_layer.tolist(), n_pairs=len(rows))
        o = out["part2"][p]
        print(f" {p} (n={len(rows)} pares): Spearman(A, Delta) mediano {o['median_sp_A']:+.3f} (umbral 0.30) | ocupacion O {o['median_sp_O']:+.3f} | techo A* {o['median_sp_Astar']:+.3f}")
        print(f"    A vs O: p por estado {pv:.4f}, p exacto por semilla {pv_seed:.4f} -> H-ATTR-1 {'SE SOSTIENE' if h1 else 'NO'}; por capa (mediana Spearman A–Delta) {np.round(by_layer, 2).tolist()}")
        print(f"    reparto del 20 % superior de A: mediana {o['median_top20_share_A']:.3f} (umbral 0.50; difuso = 0.20); con error real A*: {o['median_top20_share_Astar']:.3f} -> H-ATTR-2 {'SE SOSTIENE' if h2 else 'NO'}")
        print(f"    efecto (descriptivo): Spearman(A de la capa 0, caida del error por franja) mediano {o['effect_median_sp_A_vs_drop']} en {o['effect_n_pairs']} pares donde el error bajo")
    out["part2"]["H_ATTR_1"], out["part2"]["H_ATTR_2"] = h1_all, h2_all
    print(f"\n H-ATTR-1 (ambos problemas): {'SE SOSTIENE' if h1_all else 'NO se sostiene'} | H-ATTR-2 (ambos problemas): {'SE SOSTIENE' if h2_all else 'NO se sostiene'}")
    (HERE / "results/analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
