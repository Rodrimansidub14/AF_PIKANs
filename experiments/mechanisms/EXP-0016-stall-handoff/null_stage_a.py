"""EXP-0016, etapa A: ¿cuanto del techo del oraculo se explica solo por tomar el mejor de 5 valores ruidosos?

POST-HOC, no preregistrado (se declara asi). El criterio H-HANDOFF-1 usa k* = argmin_k E(k) y E(1.00) esta entre los 5 candidatos; si E(k) fuera
ruido intercambiable entre los k de una semilla (sin efecto de k), la razon E(k*)/E_fijo seria < 1 igualmente. Prueba de permutacion:
por semilla se reasigna al azar cual de los 5 valores hace de "control" y se recalcula el estadistico (media geometrica de min/control entre semillas;
numero de semillas con razon <= 0.80). El p es la fraccion de permutaciones con un estadistico tan extremo como el observado.
Tambien: razon media E(k)/E(1.00) por k (media geometrica entre semillas) con su prueba de signos exacta, para ver si hay un efecto sistematico de k.
"""
import json
import math
import random
from math import comb
from pathlib import Path

D = Path(__file__).parent / "results/stage-a/stageA"
SEEDS = {"burgers": [301, 302, 303, 304, 305], "ac": [601, 602, 603]}


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def sign_p(n_neg, n):
    """p bilateral exacta de la prueba de signos."""
    k = min(n_neg, n - n_neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


random.seed(20261019)
out = {}
for problem, seeds in SEEDS.items():
    E = []
    for s in seeds:
        r = json.loads((D / f"{problem}-seed{s}.json").read_text())["runs"]
        E.append([next(iter(r[f]["final_score"].values())) for f in sorted(r, key=float)])  # k crecientes, ultimo = 1.00
    obs = [min(e) / e[-1] for e in E]
    obs_gm, obs_n = gm(obs), sum(x <= 0.8 for x in obs)
    n_perm, hit_gm, hit_n = 100000, 0, 0
    for _ in range(n_perm):
        rat = []
        for e in E:
            c = random.randrange(len(e))
            rat.append(min(e) / e[c])
        hit_gm += gm(rat) <= obs_gm
        hit_n += sum(x <= 0.8 for x in rat) >= obs_n
    per_k = {}
    for j, f in enumerate(["0.25", "0.4", "0.6", "0.8"][: len(E[0]) - 1]):
        lr = [e[j] / e[-1] for e in E]
        per_k[f] = dict(gm=gm(lr), n_below_1=sum(x < 1 for x in lr), sign_p=sign_p(sum(x < 1 for x in lr), len(lr)))
    # dispersion de log E dentro de la semilla (ruido entre traspasos) frente a entre semillas
    within = [max(math.log(x) for x in e) - min(math.log(x) for x in e) for e in E]
    out[problem] = dict(observed_gm=obs_gm, observed_n_le_080=obs_n, p_gm=hit_gm / n_perm, p_count=hit_n / n_perm,
                        null_median_gm=None, per_k=per_k, within_seed_log_range_mean=sum(within) / len(within))
    print(f"\n== {problem}: observado GM {obs_gm:.3f}, {obs_n}/{len(seeds)} semillas <= 0.80")
    print(f"   permutacion (100000): p(GM <= observado) = {hit_gm / n_perm:.4f}; p(#<=0.80 >= observado) = {hit_n / n_perm:.4f}")
    for f, v in per_k.items():
        print(f"   E(k={f})/E(1.00): GM {v['gm']:.3f}, {v['n_below_1']}/{len(seeds)} semillas < 1, p signos {v['sign_p']:.3f}")
    print(f"   rango de log E dentro de la semilla (max-min) medio: {sum(within) / len(within):.2f} (factor {math.exp(sum(within) / len(within)):.1f}x)")
(D.parent / "null_stage_a.json").write_text(json.dumps(out, indent=1))
