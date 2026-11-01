"""EXP-0017 (CPU): para cada par (estado de Adam k, estado de L-BFGS iniciado en k) de la etapa A del EXP-0016 calcula las señales de la
representacion (S1, S2, S3, S5), el residuo (R), la responsabilidad por base (A, O, A*) y el movimiento (Delta), y las metricas contra el error.
Las señales y la atribucion NO leen la referencia; la referencia solo da la etiqueta |u - u_ref| (y A*, etiquetado `oracle_ceiling`).

S4 (asentamiento de la capa 1) NO se puede calcular con la definicion firmada: la etapa A registro la trayectoria de la capa 0, no la de la capa 1.
Se reporta como no computable (ausencia de medicion, no cero). Una version aproximada con los 5 puntos de la malla se calcula aparte, solo descriptiva.

Un JSON por estado en `results/states/` (se reanuda). Uso: python run_exp0017.py [--problem burgers|ac] [--limit N]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[2]))
EXP16 = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff"
DIR = EXP16 / "results/stage-a/stageA"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
OUT = HERE / "results/states"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sw = _load(EXP16 / "sweep.py", "exp0016_sweep")
sig = _load(HERE / "signals.py", "exp0017_signals")
st = _load(HERE / "stats.py", "exp0017_stats")
pc = sw.load_campaign(BL / "burgers-close-20261003/pikan_campaign.py")

PROBLEMS = {
    "burgers": dict(seeds=[301, 302, 303, 304, 305], ref=BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz", width=48, sub=(2, 4)),
    "ac": dict(seeds=[601, 602, 603], ref=BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat", width=16, sub=(1, 1)),
}
GRID_F = ["0.25", "0.4", "0.6", "0.8", "1.0"]
N_SUB = 8192
SUB_SEED = 20261020


def build(prob, cfg, seed, path):
    model = prob.build("rbf", cfg["width"], 11, seed).to(dtype=torch.float32)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    return model


def process(problem, cfg, prob, grid, seed, f):
    pts, u_ref = grid
    res = json.loads((DIR / f"{problem}-seed{seed}.json").read_text())
    k = res["grid"][f]
    m_a = build(prob, cfg, seed, DIR / f"{problem}-seed{seed}-adam-k{k}.pt")
    m_l = build(prob, cfg, seed, DIR / f"{problem}-seed{seed}-lbfgs-k{k}.pt")
    P = {"residual": pts}
    T = prob.transform
    with torch.no_grad():
        err_a = (T(pts, m_a(pts)).numpy().reshape(u_ref.shape) - u_ref).reshape(-1)
        err_l = (T(pts, m_l(pts)).numpy().reshape(u_ref.shape) - u_ref).reshape(-1)
    err_a, err_l = np.abs(err_a), np.abs(err_l)
    r_full = sw.residual_per_point(pc, prob, m_a, P).abs().numpy()
    out = dict(problem=problem, seed=seed, f=f, k=k,
               repro_residual=dict(sp=st.spearman(r_full, err_a), ov=st.top_overlap(r_full, err_a)))
    idx = np.sort(np.random.default_rng(SUB_SEED).choice(pts.shape[0], size=min(N_SUB, pts.shape[0]), replace=False))
    xs, es, rs, es_l = pts[idx], err_a[idx], r_full[idx], err_l[idx]

    t0 = time.time()
    signals = {"S1": sig.s1_jacobian_norm(m_a, xs, T, chunk=256).numpy(), "S2": sig.s2_active_bases(m_a, xs).numpy(),
               "S3": sig.s3_curvature(m_a, xs).numpy(), "S5": sig.s5_resolution_mismatch(m_a, xs, T)[0].numpy(), "R": rs}
    out["metrics"] = {name: dict(sp=st.spearman(v, es), ov=st.top_overlap(v, es)) for name, v in signals.items()}
    out["signal_seconds"] = time.time() - t0

    # --- parte 2: atribucion ---
    w_res, w_err = torch.as_tensor(rs, dtype=torch.float32), torch.as_tensor(es, dtype=torch.float32)
    A, O = sig.responsibility(m_a, xs, w_res)
    A_star, _ = sig.responsibility(m_a, xs, w_err)
    delta = sig.flat(sig.movement(m_a.state_dict(), m_l.state_dict()))
    Af, Of, Asf = sig.flat(A), sig.flat(O), sig.flat(A_star)
    out["attr"] = dict(sp_A=st.spearman(Af, delta), sp_O=st.spearman(Of, delta), sp_Astar=st.spearman(Asf, delta),
                       top20_share_A=sig.top_share(Af, 0.2), top20_share_Astar=sig.top_share(Asf, 0.2), n_bases=int(Af.size),
                       sp_A_by_layer=[st.spearman(sig.flat([a]), sig.flat([d])) for a, d in zip(A, sig.movement(m_a.state_dict(), m_l.state_dict()))])
    # --- comprobacion de efecto (descriptiva): error por franja de soporte de cada base de la capa 0, antes y despues ---
    h0 = sig.layer_inputs(m_a, xs)[0]
    ph = sig.phi(h0, m_a.layers[0]).detach().numpy()                           # (n, 2, G)
    drops, a0 = [], []
    for d in range(ph.shape[1]):
        for g in range(ph.shape[2]):
            mask = ph[:, d, g] > 0.5
            if mask.sum() >= 20 and es[mask].mean() > 0:
                drops.append(float((es[mask].mean() - es_l[mask].mean()) / es[mask].mean()))
                a0.append(float(A[0][d, g]))
    out["effect"] = dict(global_before=float(np.linalg.norm(es)), global_after=float(np.linalg.norm(es_l)),
                         error_dropped=bool(np.linalg.norm(es_l) < np.linalg.norm(es)), n_strips=len(drops),
                         sp_A_vs_drop=st.spearman(a0, drops) if len(drops) >= 5 else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=list(PROBLEMS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    done = 0
    for problem, cfg in PROBLEMS.items():
        if a.problem and problem != a.problem:
            continue
        prob = pc.make_problem(problem, str(cfg["ref"]))
        st_, sx = cfg["sub"]
        xx, tt = np.meshgrid(prob.x[::sx], prob.t[::st_], indexing="xy")
        u_ref = prob.u[::st_, ::sx]
        grid = (torch.as_tensor(np.stack((xx.reshape(-1), tt.reshape(-1)), 1), dtype=torch.float32), u_ref)
        for seed in cfg["seeds"]:
            for f in GRID_F:
                path = OUT / f"{problem}-{seed}-{f}.json"
                if path.exists():
                    continue
                t0 = time.time()
                path.write_text(json.dumps(process(problem, cfg, prob, grid, seed, f)))
                done += 1
                print(f"{problem} {seed} k={f}: {time.time() - t0:.0f}s", flush=True)
                if a.limit and done >= a.limit:
                    return


if __name__ == "__main__":
    main()
