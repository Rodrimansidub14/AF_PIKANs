"""EXP-0016, etapa B (CPU, sin corridas nuevas): H-AWARE-1 (B1) y H-HANDOFF-2 (B2), con los criterios del spec v2.

B1: en cada checkpoint de Adam de la malla (5 por semilla) se calculan, SIN referencia, el mapa interno m(x) (de los coeficientes de la capa 0 registrados cada 500 pasos)
y el residuo |r(x)| (de la red en ese checkpoint) sobre la malla de la referencia; la referencia solo da la ETIQUETA |u - u_ref|(x). Metricas por checkpoint:
Spearman(senal, error) y solape del 5 % superior de la senal con el 5 % superior del error (azar = 0.05).
  H-AWARE-1 (por problema): mediana, sobre todos los checkpoints, de Spearman(m, error) >= 0.30 y del solape >= 0.25, y ambos mayores que los del residuo
  en >= 2/3 de las semillas (mediana por semilla sobre sus 5 checkpoints).
B2: k_t por semilla de cada disparo (interno: asentamiento q*=0.8, rho*=0.10; linea base: estancamiento de la perdida rho*=0.10), sin referencia. E(k_t) se lee del barrido
interpolando log E linealmente en k entre puntos de la malla (clip en los extremos). EI = [log E_fijo - log E(k_t)] / [log E_fijo - log E(k*)], definido donde
log E_fijo - log E(k*) > log 1.05.
  H-HANDOFF-2 (por problema): mediana de EI del disparo interno >= 0.50 y EI interno >= EI de la perdida en >= 2/3 de las semillas con techo medible.
Subconjunto de la malla de la referencia (Burgers: cada 4.o x y 2.o t; Allen-Cahn: completa) para acotar el costo de las derivadas segundas en CPU.
Uso: python analyze_stage_b.py [carpeta con los JSON y .pt de la etapa A]
"""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[2]))
sys.path.insert(0, str(ROOT))
from src.mechanisms.settling_focus import settling_rates, unsettled_map  # noqa: E402

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results/stage-a/stageA"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
spec = importlib.util.spec_from_file_location("exp0016_sweep", str(HERE / "sweep.py"))
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)
pc = sw.load_campaign(BL / "burgers-close-20261003/pikan_campaign.py")
torch.set_num_threads(4)

PROBLEMS = {
    "burgers": dict(seeds=[301, 302, 303, 304, 305], ref=BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz", width=48, sub=(2, 4), need=4 / 5),
    "ac": dict(seeds=[601, 602, 603], ref=BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat", width=16, sub=(1, 1), need=2 / 3),
}
GRID_F = ["0.25", "0.4", "0.6", "0.8", "1.0"]


def top_overlap(a, b, frac=0.05):
    n = max(1, int(round(frac * a.size)))
    ta, tb = np.argsort(a)[-n:], np.argsort(b)[-n:]
    return len(set(ta.tolist()) & set(tb.tolist())) / n


def interp_logE(ks, E, k_t):
    ks, logE = np.asarray(ks, float), np.log(np.asarray(E, float))
    return float(np.interp(k_t, ks, logE))      # clip en los extremos


def gm_median(xs):
    return float(np.median(xs))


def main():
    summary = {}
    for problem, cfg in PROBLEMS.items():
        prob = pc.make_problem(problem, str(cfg["ref"]))
        st, sx = cfg["sub"]
        xx, tt = np.meshgrid(prob.x[::sx], prob.t[::st], indexing="xy")
        u_ref = prob.u[::st, ::sx]
        pts = np.stack((xx.reshape(-1), tt.reshape(-1)), 1)
        P = {"residual": torch.as_tensor(pts, dtype=torch.float32)}
        rows, b2 = [], []
        print(f"\n=== {problem}: malla {u_ref.shape} = {pts.shape[0]} puntos ===", flush=True)
        for seed in cfg["seeds"]:
            res = json.loads((DIR / f"{problem}-seed{seed}.json").read_text())
            tr = np.load(DIR / f"{problem}-seed{seed}-adam-trace.npz")
            ends, _, rho = settling_rates(tr["steps"], tr["coeffs"])
            lo, hi = tr["lo"].astype(float), tr["hi"].astype(float)
            xn = 2.0 * (pts - lo) / (hi - lo) - 1.0
            for f in GRID_F:
                k = res["grid"][f]
                ck = torch.load(DIR / f"{problem}-seed{seed}-adam-k{k}.pt", map_location="cpu", weights_only=False)["state_dict"]
                model = prob.build("rbf", cfg["width"], 11, seed).to(dtype=torch.float32)
                model.load_state_dict(ck)
                with torch.no_grad():
                    pred = prob.transform(P["residual"], model(P["residual"])).numpy().reshape(u_ref.shape)
                err = np.abs(pred - u_ref).reshape(-1)
                r = sw.residual_per_point(pc, prob, model, P).abs().numpy()
                m = unsettled_map(xn, rho[list(ends).index(k)], tr["centers"], float(tr["width"]))
                rows.append(dict(seed=seed, f=f, k=k, sp_map=float(spearmanr(m, err)[0]), ov_map=top_overlap(m, err),
                                 sp_res=float(spearmanr(r, err)[0]), ov_res=top_overlap(r, err)))
                print(f" seed {seed} k={k:>5}: map  rho_s={rows[-1]['sp_map']:+.2f} overlap={rows[-1]['ov_map']:.2f} | residual rho_s={rows[-1]['sp_res']:+.2f} overlap={rows[-1]['ov_res']:.2f}", flush=True)
            # B2
            runs = res["runs"]
            ks = [runs[f]["k"] for f in GRID_F]
            E = [next(iter(runs[f]["final_score"].values())) for f in GRID_F]
            logEf, logEs = math.log(E[-1]), math.log(min(E))
            kt_int = res["trigger_settling"]["q0.8_rho0.10"]["k_t"]
            kt_loss = res["trigger"]["0.1"]["k_t"]
            measurable = (logEf - logEs) > math.log(1.05)
            ei = lambda kt: (logEf - interp_logE(ks, E, kt)) / (logEf - logEs) if measurable else None  # noqa: E731
            b2.append(dict(seed=seed, k_star=ks[int(np.argmin(E))], measurable=measurable, kt_internal=kt_int, kt_loss=kt_loss,
                           fired_internal=res["trigger_settling"]["q0.8_rho0.10"]["fired"], fired_loss=res["trigger"]["0.1"]["fired"],
                           EI_internal=ei(kt_int), EI_loss=ei(kt_loss)))
        # criterios
        sp_map = float(np.median([r["sp_map"] for r in rows]))
        ov_map = float(np.median([r["ov_map"] for r in rows]))
        sp_res = float(np.median([r["sp_res"] for r in rows]))
        ov_res = float(np.median([r["ov_res"] for r in rows]))
        better = []
        for seed in cfg["seeds"]:
            rs = [r for r in rows if r["seed"] == seed]
            better.append(np.median([r["sp_map"] for r in rs]) > np.median([r["sp_res"] for r in rs])
                          and np.median([r["ov_map"] for r in rs]) > np.median([r["ov_res"] for r in rs]))
        h_aware = sp_map >= 0.30 and ov_map >= 0.25 and (sum(better) / len(better)) >= 2 / 3
        meas = [b for b in b2 if b["measurable"]]
        ei_int = [b["EI_internal"] for b in meas]
        ei_loss = [b["EI_loss"] for b in meas]
        h2 = bool(meas) and float(np.median(ei_int)) >= 0.5 and (sum(i >= l for i, l in zip(ei_int, ei_loss)) / len(meas)) >= 2 / 3
        print(f"\n B1 {problem}: mediana Spearman mapa {sp_map:+.3f} (umbral 0.30) | solape mapa {ov_map:.3f} (0.25) | residuo: {sp_res:+.3f} / {ov_res:.3f}")
        print(f"    semillas donde el mapa supera al residuo en ambas metricas: {sum(better)}/{len(better)} -> H-AWARE-1 {'SE SOSTIENE' if h_aware else 'NO se sostiene'}")
        print(f" B2 {problem}: semillas con techo medible {len(meas)}/{len(b2)}; EI interno {[round(x, 2) for x in ei_int]}, EI perdida {[round(x, 2) for x in ei_loss]}")
        for b in b2:
            print(f"    seed {b['seed']}: k*={b['k_star']} k_t interno={b['kt_internal']}{'' if b['fired_internal'] else ' (no dispara)'} k_t perdida={b['kt_loss']}{'' if b['fired_loss'] else ' (no dispara)'}")
        print(f"    mediana EI interno {float(np.median(ei_int)) if ei_int else float('nan'):.2f} -> H-HANDOFF-2 {'SE SOSTIENE' if h2 else 'NO se sostiene'}")
        summary[problem] = dict(rows=rows, b2=b2, sp_map=sp_map, ov_map=ov_map, sp_res=sp_res, ov_res=ov_res, seeds_map_better=sum(better),
                                h_aware_1=bool(h_aware), h_handoff_2=bool(h2))
    (DIR.parent / "stage_b_analysis.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
