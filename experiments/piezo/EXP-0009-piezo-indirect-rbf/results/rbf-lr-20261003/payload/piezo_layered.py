#!/usr/bin/env python3
"""DRAFT_UNVERIFIED: layered formulation of EXP-0009 (indirect PVDF bimorph).

One network per layer (bottom: eta in [0, .5], opposite poling on top: eta in [.5, 1]); constant coefficients inside each layer, so no
coefficient jump is smeared inside a network. Interface (eta = .5): continuity of u, v, phi (displacements, potential) and of the
normal traction/displacement components sigma_z, tau_xz, D_y; sigma_x and D_x may jump. Same physics, scales, reference and metrics as
`piezo_pinn.py` (imported for constants, models and the residual). Hard BCs: u = v = 0 at x = 0; phi = 0 at y = 0 (bottom layer) and
phi = V at y = H (top layer). Group-weight balancing as in the thesis PINN (optional).
Modes: smoke | run.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc
from torch import nn

import piezo_pinn as pz

GROUPS = ("mech_const", "elec_const", "equil", "bc", "iface")
DT, DEV = torch.float64, torch.device("cpu")
VS = 0.0


class Layer(nn.Module):
    def __init__(self, net, layer):
        super().__init__()
        self.net, self.layer = net, layer  # layer 0 bottom, 1 top

    def forward(self, xi):
        x, eta = xi[:, 0:1], xi[:, 1:2]
        zeta = (eta - 0.5 * self.layer) / 0.5
        o = self.net(torch.cat([x, zeta], dim=1))
        v = o[:, 1:2]
        if VS > 0:
            mid = self.net(torch.cat([x, torch.full_like(zeta, 0.5)], dim=1))[:, 1:2]
            v = mid + VS * (v - mid)
        bubble = zeta if self.layer == 0 else (1.0 - zeta)
        return torch.cat([x * o[:, 0:1], x * v, eta + bubble * o[:, 2:3], o[:, 3:]], dim=1)


class Bimorph(nn.Module):
    def __init__(self, arm, width):
        super().__init__()
        mk = (lambda: pz.MLP([2, 100, 250, 8])) if arm == "mlp" else (lambda: pz.RBFKAN(2, width, 8))
        self.layers = nn.ModuleList([Layer(mk(), 0), Layer(mk(), 1)])


def build(arm, width, seed):
    torch.manual_seed(seed)
    return Bimorph(arm, width).to(device=DEV, dtype=DT)


def make_pools(seed, n_col=2048, n_bd=256):
    ss = [int(x.generate_state(1)[0]) for x in np.random.SeedSequence(seed).spawn(12)]

    def sob(d, n, k):
        return qmc.Sobol(d, scramble=True, seed=ss[k]).random_base2(int(math.log2(n)))

    t = lambda a: torch.tensor(a, dtype=DT, device=DEV)
    P = {}
    for li in (0, 1):
        c = sob(2, n_col, li)
        P[f"col{li}"] = t(np.stack([c[:, 0], 0.5 * li + 0.5 * c[:, 1]], 1))
        e = lambda k: 0.5 * li + 0.5 * sob(1, n_bd, k)[:, 0]
        P[f"right{li}"] = t(np.stack([np.ones(n_bd), e(2 + 4 * li)], 1))
        P[f"left{li}"] = t(np.stack([np.zeros(n_bd), e(3 + 4 * li)], 1))
    P["bot"] = t(np.stack([sob(1, n_bd, 10)[:, 0], np.zeros(n_bd)], 1))
    P["top"] = t(np.stack([sob(1, n_bd, 11)[:, 0], np.ones(n_bd)], 1))
    P["iface"] = t(np.stack([sob(1, n_bd, 9)[:, 0], 0.5 * np.ones(n_bd)], 1))
    return P


def loss_groups(model, P):
    sgn = (1.0, -1.0)
    g = {k: 0.0 for k in GROUPS}
    for li in (0, 1):
        r = pz.residual_terms(model.layers[li], P[f"col{li}"], sign=sgn[li])
        sq = {k: v.square().mean() for k, v in r.items()}
        g["mech_const"] = g["mech_const"] + sq["sx"] + sq["sz"] + sq["tau"]
        g["elec_const"] = g["elec_const"] + sq["Dx"] + sq["Dy"]
        g["equil"] = g["equil"] + sq["div1"] + sq["div2"] + sq["divD"]
        o = model.layers[li](P[f"right{li}"])
        g["bc"] = g["bc"] + o[:, 3].square().mean() + o[:, 5].square().mean() + o[:, 6].square().mean()
        o = model.layers[li](P[f"left{li}"])
        g["bc"] = g["bc"] + o[:, 6].square().mean()
    o = model.layers[0](P["bot"])
    g["bc"] = g["bc"] + o[:, 4].square().mean() + o[:, 5].square().mean()
    o = model.layers[1](P["top"])
    g["bc"] = g["bc"] + o[:, 4].square().mean() + o[:, 5].square().mean()
    ob, ot = model.layers[0](P["iface"]), model.layers[1](P["iface"])
    for c in (0, 1, 2, 4, 5, 7):  # u, v, phi, sigma_z, tau, D_y continuous
        g["iface"] = g["iface"] + (ob[:, c] - ot[:, c]).square().mean()
    return g


W = {k: 1.0 for k in GROUPS}


def loss_fn(model, P):
    g = loss_groups(model, P)
    return sum(W[k] * g[k] for k in GROUPS), {k: float(v.detach()) for k, v in g.items()}


def rebalance(model, P, ema=0.9):
    g = loss_groups(model, P)
    params = list(model.parameters())
    norms = {}
    for k in GROUPS:
        gr = torch.autograd.grad(g[k], params, retain_graph=True, allow_unused=True)
        norms[k] = float(torch.sqrt(sum((x ** 2).sum() for x in gr if x is not None))) + 1e-30
    ref = max(norms.values())
    for k in GROUPS:
        W[k] = ema * W[k] + (1 - ema) * ref / norms[k]


def score(model, fem):
    xy = fem["xy"]
    xi = torch.tensor(np.stack([xy[:, 0] / pz.L, xy[:, 1] / pz.H], 1), dtype=DT, device=DEV)
    with torch.no_grad():
        ob, ot = model.layers[0](xi), model.layers[1](xi)
        o = torch.where((xi[:, 1:2] <= 0.5), ob, ot).double().cpu().numpy()
        tip = float(model.layers[0](torch.tensor([[1.0, 0.0]], dtype=DT, device=DEV))[0, 1]) * pz.SC_V
    u, v, p = o[:, 0] * pz.SC_U, o[:, 1] * pz.SC_V, o[:, 2] * pz.V

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    return {"u_relL2": rel(u, fem["u"]), "v_relL2": rel(v, fem["v"]), "phi_relL2": rel(p, fem["phi"]), "tip_v": tip,
            "tip_v_fem": float(fem["v"][np.argmin(np.abs(xy[:, 0] - pz.L) + xy[:, 1])])}


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.fem)
    fem = {k: z[k] for k in ("xy", "u", "v", "phi")}
    P = make_pools(a.seed, a.n_col, a.n_bd)
    model = build(a.arm, a.width, a.seed)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    hist, t0 = [], time.perf_counter()
    snaps = []
    for ep in range(1, a.adam_steps + 1):
        if a.balance and ep % a.balance == 1:
            rebalance(model, P)
        opt.zero_grad(set_to_none=True)
        l, parts = loss_fn(model, P)
        l.backward()
        opt.step()
        if ep % 100 == 0 or ep == a.adam_steps:
            hist.append({"stage": "adam", "step": ep, "loss": float(l.detach()), **parts})
        if ep % 2000 == 0:
            sc = score(model, fem)
            snaps.append({"step": ep, "u": sc["u_relL2"], "v": sc["v_relL2"], "phi": sc["phi_relL2"], "tip_v": sc["tip_v"]})
    res = {"status": pz.STATUS, "formulation": "layered", "arm": a.arm, "seed": a.seed, "params": params, "adam_steps": a.adam_steps,
           "adam_seconds": time.perf_counter() - t0, "dtype": str(DT), "vsplit": VS, "balance": a.balance, "weights_final": dict(W),
           "adam_snapshots": snaps, "score_adam_final": score(model, fem), "scientific_accepted": False, "reference_accepted": False}
    torch.save({"state_dict": model.state_dict()}, out / "checkpoint-adam-final.pt")
    print(json.dumps({"arm": a.arm, "seed": a.seed, "params": params, "adam": res["score_adam_final"], "snaps": snaps[-3:]}), flush=True)
    if a.lbfgs_weights == "unit":
        for k in GROUPS:
            W[k] = 1.0
    lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=25, max_eval=125, tolerance_grad=1e-12, tolerance_change=0.0,
                           history_size=100, line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad(set_to_none=True)
        l, _ = loss_fn(model, P)
        l.backward()
        return l

    last = float(loss_fn(model, P)[0].detach())
    acc = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best = {"err": res["score_adam_final"]["v_relL2"], "state": acc, "it": 0}
    t1, it, plateau, status = time.perf_counter(), 0, 0, "lbfgs_time_cap"
    while time.perf_counter() - t1 < a.lbfgs_seconds and it < a.lbfgs_max_iter:
        b = lb.state[lb._params[0]].get("n_iter", 0)
        try:
            lb.step(closure)
        except Exception as e:  # noqa: BLE001
            status = f"lbfgs_exception:{type(e).__name__}"
            break
        cur = float(loss_fn(model, P)[0].detach())
        if not math.isfinite(cur) or cur > last * 1.5:
            status = "lbfgs_rejected"
            break
        dn = lb.state[lb._params[0]].get("n_iter", 0) - b
        imp = (last - cur) / max(abs(last), 1e-30)
        acc = {k: v.detach().clone() for k, v in model.state_dict().items()}
        last, it = cur, it + dn
        hist.append({"stage": "lbfgs", "step": it, "loss": cur})
        if dn == 0:
            status = "lbfgs_zero_progress"
            break
        plateau = plateau + 1 if imp < 1e-10 else 0
        if plateau >= 5:
            status = "lbfgs_plateau"
            break
    else:
        status = "lbfgs_iter_cap" if it >= a.lbfgs_max_iter else "lbfgs_time_cap"
    model.load_state_dict(acc)
    res.update({"lbfgs_status": status, "lbfgs_iterations": it, "lbfgs_seconds": time.perf_counter() - t1, "score_lbfgs_final": score(model, fem)})
    torch.save({"state_dict": model.state_dict()}, out / "checkpoint-lbfgs-final.pt")
    (out / "history.json").write_text(json.dumps(hist))
    (out / "run.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"arm": a.arm, "seed": a.seed, "lbfgs": res["score_lbfgs_final"], "status": status, "iters": it}), flush=True)


def smoke(a):
    P = make_pools(1, 256, 32)
    for arm in ("mlp", "rbf"):
        m = build(arm, a.width, 1)
        xi = torch.tensor([[0.0, 0.3], [0.5, 0.0], [0.5, 1.0]], dtype=DT, device=DEV)
        ob, ot = m.layers[0](xi), m.layers[1](xi)
        assert torch.allclose(ob[0, :2], torch.zeros(2, dtype=DT)), "u,v vanish at x=0"
        assert abs(float(ob[1, 2])) < 1e-12 and abs(float(ot[2, 2]) - 1.0) < 1e-12, "phi=0 bottom, V top"
        l0, _ = loss_fn(m, P)
        l0.backward()
        assert torch.isfinite(l0)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(30):
            opt.zero_grad()
            l, _ = loss_fn(m, P)
            l.backward()
            opt.step()
        assert float(l) < float(l0)
        print(arm, "OK", float(l0), "->", float(l), "params", sum(p.numel() for p in m.parameters()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("smoke", "run"))
    ap.add_argument("--arm", choices=("mlp", "rbf"), default="rbf")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fem")
    ap.add_argument("--out")
    ap.add_argument("--n-col", type=int, default=2048)
    ap.add_argument("--n-bd", type=int, default=256)
    ap.add_argument("--adam-steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--vsplit", type=float, default=0.0)
    ap.add_argument("--balance", type=int, default=0)
    ap.add_argument("--lbfgs-weights", choices=("frozen", "unit"), default="unit")
    ap.add_argument("--lbfgs-seconds", type=float, default=300.0)
    ap.add_argument("--lbfgs-max-iter", type=int, default=15000)
    args = ap.parse_args()
    DT = torch.float32 if args.dtype == "float32" else torch.float64
    DEV = torch.device(args.device)
    VS = args.vsplit
    (smoke if args.mode == "smoke" else run)(args)
