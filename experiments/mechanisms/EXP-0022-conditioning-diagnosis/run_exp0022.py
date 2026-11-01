"""EXP-0022 (CPU): para cada par (estado de Adam k, estado L-BFGS iniciado en k) de la etapa A del EXP-0016, espectro de Gauss-Newton en el estado de Adam,
kappa_eff con cuatro precondicionadores (crudo, Jacobi, por arista [primario], por entrada [descriptivo]), cociente de Rayleigh del desplazamiento,
Gram de la base por arista y localizacion de la energia del desplazamiento. No lee la referencia. Un JSON por estado en `results/states/` (se reanuda).
Uso: python run_exp0022.py [--problem burgers|ac] [--limit N] [--n-pts 2048] [--seeds ...]
"""
import argparse
import importlib.util
import json
import os
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
GRID_F = ["0.25", "0.4", "0.6", "0.8", "1.0"]
SUB_SEED = 20261020
M_LIST = (10, 100, 500)
EPS_REL = 1e-3


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gn = _load(HERE / "gn.py", "exp0022_gn")
sw = _load(EXP16 / "sweep.py", "exp0016_sweep")
pc = sw.load_campaign(BL / "burgers-close-20261003/pikan_campaign.py")
PROBLEMS = {
    "burgers": dict(seeds=[301, 302, 303, 304, 305], ref=BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz", width=48),
    "ac": dict(seeds=[601, 602, 603], ref=BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat", width=16),
}


def build(prob, cfg, seed, path):
    model = prob.build("rbf", cfg["width"], 11, seed).to(dtype=torch.float32)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    return model.double()


def kappas(eigs):
    return {str(m): gn.kappa(eigs, m) for m in M_LIST if m <= len(eigs)} | {"lam1": float(eigs[0])}


def energy_localization(d, edges, model):
    """Fraccion de la energia de d en el 20 % de aristas con mas energia, por capa."""
    out = {}
    for l in range(3):
        e = np.array([float(np.square(d[c]).sum()) for (ll, o, i), c in edges.items() if ll == l])
        e = np.sort(e)[::-1]
        out[str(l)] = float(e[:max(1, int(round(0.2 * len(e))))].sum() / e.sum()) if e.sum() > 0 else None
    return out


@torch.no_grad()
def gram_conditioning(model, pts):
    """Numero de condicion y rango efectivo de la Gram de la base por arista: datos reales frente a entradas uniformes en [-1, 1]."""
    from math import pi
    h = 2.0 * (pts - model.lo) / (model.hi - model.lo) - 1.0
    res = {}
    for l, layer in enumerate(model.layers):
        z = h
        phi = lambda z_: torch.exp(-((z_.unsqueeze(-1) - layer.centers) / layer.width).square())
        P = phi(z)                                                        # (N, I, G)
        G = torch.einsum("nig,nih->igh", P, P) / z.shape[0]
        ev = torch.linalg.eigvalsh(G).clamp_min(1e-300)
        cond = (ev[:, -1] / ev[:, 0]).detach().cpu().numpy()
        eff = ((ev.sum(-1)) ** 2 / (ev ** 2).sum(-1)).detach().cpu().numpy()
        U = torch.rand(20000, z.shape[1], dtype=z.dtype, generator=torch.Generator().manual_seed(1)) * 2 - 1
        Pu = phi(U)
        Gu = torch.einsum("nig,nih->igh", Pu, Pu) / U.shape[0]
        evu = torch.linalg.eigvalsh(Gu).clamp_min(1e-300)
        condu = (evu[:, -1] / evu[:, 0]).detach().cpu().numpy()
        res[str(l)] = dict(cond_median=float(np.median(cond)), cond_uniform_median=float(np.median(condu)), eff_rank_median=float(np.median(eff)))
        z_next = torch.tanh(layer(z))
        h = z_next
    return res


def process(problem, cfg, prob, seed, f, n_pts, subset=0, jdtype=torch.float32):
    res = json.loads((DIR / f"{problem}-seed{seed}.json").read_text())
    k = res["grid"][f]
    m_a = build(prob, cfg, seed, DIR / f"{problem}-seed{seed}-adam-k{k}.pt")
    m_l = build(prob, cfg, seed, DIR / f"{problem}-seed{seed}-lbfgs-k{k}.pt")
    pool = prob.pools(seed, "cpu", torch.float64)["residual"]
    perm = np.random.default_rng([SUB_SEED, seed]).permutation(pool.shape[0])
    idx = np.sort(perm[subset * n_pts:(subset + 1) * n_pts])
    X = pool[idx]
    t0 = time.time()
    J = gn.jacobian(prob, m_a, X, chunk=16, dtype=jdtype)
    t_jac = time.time() - t0
    edges, rest, P = gn.layout(m_a)
    out = dict(problem=problem, seed=seed, f=f, k=k, n_pts=int(X.shape[0]), n_params=int(P), seconds_jacobian=t_jac)
    spectra = {}
    spectra["raw"] = gn.gram_spectrum(J)
    spectra["jacobi"] = gn.gram_spectrum(J, scale=gn.jacobi_scale(J))
    JS = gn.apply_edge_precond(J, edges, rest, EPS_REL)
    spectra["edge"] = gn.gram_spectrum(JS)
    del JS
    JS = gn.apply_input_precond(J, edges, rest, EPS_REL)
    spectra["input"] = gn.gram_spectrum(JS)
    del JS
    out["kappa"] = {name: kappas(e) for name, e in spectra.items()}
    lam1 = float(spectra["raw"][0])
    d = (gn.flat_params(m_l) - gn.flat_params(m_a))
    out["rho_d"] = gn.rayleigh_rel(J, d, lam1)
    g = torch.Generator().manual_seed(SUB_SEED + seed)
    sg = [gn.rayleigh_rel(J, d * (torch.randint(0, 2, d.shape, generator=g) * 2 - 1).to(d.dtype), lam1) for _ in range(20)]
    out["rho_signrand"] = float(np.mean(sg))
    out["rho_ratio"] = out["rho_d"] / out["rho_signrand"]
    out["spectrum_raw"] = [float(v) for v in spectra["raw"][:600]]
    out["disp_norm_rel"] = float(d.norm() / gn.flat_params(m_a).norm())
    out["energy_top20_by_layer"] = energy_localization(d.numpy(), edges, m_a)
    out["gram_basis"] = gram_conditioning(m_a, X)
    out["spectrum_head"] = [float(v) for v in spectra["raw"][:20]]
    out["seconds_total"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=list(PROBLEMS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-pts", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    done = 0
    for problem, cfg in PROBLEMS.items():
        if a.problem and problem != a.problem:
            continue
        prob = pc.make_problem(problem, str(cfg["ref"]))
        for seed in cfg["seeds"]:
            for f in GRID_F:
                path = outdir / f"{problem}-{seed}-{f}.json"
                if path.exists():
                    continue
                path.write_text(json.dumps(process(problem, cfg, prob, seed, f, a.n_pts)))
                done += 1
                print(f"{problem} {seed} f={f}: listo", flush=True)
                if a.limit and done >= a.limit:
                    return


if __name__ == "__main__":
    main()
