#!/usr/bin/env python3
"""DRAFT_UNVERIFIED: indirect (voltage-driven) PVDF bimorph, 2D plane stress, mixed 8-output PINN (u,v,phi,sx,sz,txz,Dx,Dy).

Own implementation of the physics (not Daniel Gonzalez's code: licence unconfirmed). Problem as in the thesis P5 / Daniel's indirect formulation:
beam L=0.1 m, H=1 mm, clamped at x=0 (u=v=0), 100 V on top (y=H), 0 V on bottom (y=0), opposite poling in the two layers (interface y=H/2).
Coefficient convention fixed by `sign_check.py` (the FEM field satisfies div(sigma)=0, div(D)=0 only with opposite-sign e31/e33 and the
constant-strain permittivity): bottom layer e31=+0.02904, e33=-0.05158; top layer flipped.
Reference field: Daniel's FEM with poling_sign=-1 (same convention), 201x21 grid.
Nondimensional inputs xi=x/L, eta=y/H; outputs scaled by fixed characteristic scales; hard BCs on u,v,phi.
Arms: mlp (2-100-250-8 pyramid, tanh, Glorot) vs rbf (RBF-PIKAN width W, 11 bases/edge). Stages: Adam then strong-Wolfe L-BFGS; both scored.
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

STATUS = "DRAFT_UNVERIFIED"
DT = torch.float64  # set from --dtype
DEV = torch.device("cpu")  # set from --device
VSPLIT = 0.0  # >0: v-hat = xi*[N1(xi,.5) + VSPLIT*(N1(xi,eta)-N1(xi,.5))] (separates the O(1) bending deflection from the ~1e-5 thickness strain)
L, H, V = 0.1, 1e-3, 100.0
C11 = 2.0e9 / (1 - 0.29 ** 2)
C12 = 0.29 * C11
C22 = C11
G = 0.775e9
D31, D33 = 2.2e-11, -3.0e-11
E31 = C11 * D31 + C12 * D33
E33 = C12 * D31 + C11 * D33
EPS = 12 * 8.854187817620e-12 - (D31 * (C11 * D31 + C12 * D33) + D33 * (C12 * D31 + C11 * D33))  # constant-strain permittivity
SC_U, SC_V, SC_S, SC_D = 3e-7, 3.3e-5, 6e3, 1e-5  # characteristic scales for u, v, stress, D


# ----------------------------------------------------------------------------- models
class RBFEdge(nn.Module):
    def __init__(self, inputs, outputs, bases):
        super().__init__()
        spacing = 2.0 / (bases - 1)
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, bases))
        self.register_buffer("width", torch.tensor(1.75 * spacing))
        self.coefficients = nn.Parameter(torch.randn(outputs, inputs, bases) * (0.1 / math.sqrt(inputs)))
        self.base = nn.Parameter(torch.randn(outputs, inputs) * (0.1 / math.sqrt(inputs)))

    def forward(self, x):
        basis = torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width).square())
        return torch.einsum("big,oig->bo", basis, self.coefficients) + torch.nn.functional.linear(torch.nn.functional.silu(x), self.base)


class RBFKAN(nn.Module):
    def __init__(self, n_in, width, n_out, bases=11):
        super().__init__()
        self.layers = nn.ModuleList([RBFEdge(n_in, width, bases), RBFEdge(width, width, bases), RBFEdge(width, n_out, bases)])

    def forward(self, xi):
        h = 2.0 * xi - 1.0  # inputs already in [0,1]
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)


class MLP(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, xi):
        x = 2.0 * xi - 1.0
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)


class Piezo(nn.Module):
    """Wraps a raw 8-output net with hard BCs; returns scaled outputs (uh, vh, ph, sx, sz, tau, Dx, Dy)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, xi):
        o = self.net(xi)
        x, y = xi[:, 0:1], xi[:, 1:2]
        v = o[:, 1:2]
        if VSPLIT > 0:
            mid = self.net(torch.cat([x, torch.full_like(y, 0.5)], dim=1))[:, 1:2]
            v = mid + VSPLIT * (v - mid)
        return torch.cat([x * o[:, 0:1], x * v, y + y * (1 - y) * o[:, 2:3], o[:, 3:]], dim=1)


# ----------------------------------------------------------------------------- problem
def d(f, x):
    return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True, retain_graph=True)[0]


def coeffs(eta):
    """per-point e31, e33: bottom (eta<0.5) +E31,+E33 ; top flipped."""
    sg = torch.where(eta < 0.5, 1.0, -1.0)
    return sg * E31, sg * E33


def make_pools(seed, n_col=4096, n_bd=256):
    seeds = [int(x.generate_state(1)[0]) for x in np.random.SeedSequence(seed).spawn(6)]
    col = qmc.Sobol(2, scramble=True, seed=seeds[0]).random_base2(int(math.log2(n_col)))

    def side(k):
        return qmc.Sobol(1, scramble=True, seed=seeds[k]).random_base2(int(math.log2(n_bd)))[:, 0]

    zeros, ones = np.zeros(n_bd), np.ones(n_bd)
    pools = {"col": col,
             "left": np.stack([zeros, side(1)], 1), "right": np.stack([ones, side(2)], 1),
             "bot": np.stack([side(3), zeros], 1), "top": np.stack([side(4), ones], 1)}
    return {k: torch.tensor(v, dtype=DT, device=DEV) for k, v in pools.items()}


def residual_terms(model, xi):
    xi = xi.clone().requires_grad_(True)
    o = model(xi)
    uh, vh, ph, sx, sz, tau, Dx, Dy = [o[:, i:i + 1] for i in range(8)]
    gu, gv, gp = d(uh, xi), d(vh, xi), d(ph, xi)
    exx = SC_U / L * gu[:, 0:1]
    eyy = SC_V / H * gv[:, 1:2]
    exy = 0.5 * (SC_U / H * gu[:, 1:2] + SC_V / L * gv[:, 0:1])
    Ex = -V / L * gp[:, 0:1]
    Ey = -V / H * gp[:, 1:2]
    e31, e33 = coeffs(xi[:, 1:2])
    r = {"sx": (SC_S * sx - (C11 * exx + C12 * eyy - e31 * Ey)) / SC_S,
         "sz": (SC_S * sz - (C12 * exx + C22 * eyy - e33 * Ey)) / SC_S,
         "tau": (SC_S * tau - G * exy) / SC_S,
         "Dx": (SC_D * Dx - EPS * Ex) / SC_D,
         "Dy": (SC_D * Dy - (e31 * exx + e33 * eyy + EPS * Ey)) / SC_D}
    gsx, gsz, gt, gDx, gDy = d(sx, xi), d(sz, xi), d(tau, xi), d(Dx, xi), d(Dy, xi)
    r["div1"] = (gsx[:, 0:1] / L + gt[:, 1:2] / H) * L  # equilibrium, scaled by S/L
    r["div2"] = (gt[:, 0:1] / L + gsz[:, 1:2] / H) * L
    r["divD"] = (gDx[:, 0:1] / L + gDy[:, 1:2] / H) * L
    return r


GROUPS = ("mech_const", "elec_const", "equil", "bc")
WEIGHTS = {g: 1.0 for g in GROUPS}
TERM_W = {"tau": 1.0, "sz": 1.0}  # extra per-term weights inside mech_const (set from --w-tau/--w-sz)


def loss_groups(model, P):
    r = residual_terms(model, P["col"])
    sq = {k: v.square().mean() for k, v in r.items()}
    g = {"mech_const": sq["sx"] + TERM_W["sz"] * sq["sz"] + TERM_W["tau"] * sq["tau"], "elec_const": sq["Dx"] + sq["Dy"],
         "equil": sq["div1"] + sq["div2"] + sq["divD"]}
    o = model(P["right"])
    bc = o[:, 3].square().mean() + o[:, 5].square().mean() + o[:, 6].square().mean()  # free end: sx=tau=0, Dx=0
    o = model(P["left"])
    bc = bc + o[:, 6].square().mean()  # Dx=0 at clamped end
    for k in ("bot", "top"):
        o = model(P[k])
        bc = bc + o[:, 4].square().mean() + o[:, 5].square().mean()  # free surfaces: sz=tau=0
    g["bc"] = bc
    return g


def loss_fn(model, P):
    g = loss_groups(model, P)
    total = sum(WEIGHTS[k] * g[k] for k in GROUPS)
    return total, {k: float(v.detach()) for k, v in g.items()}


def rebalance(model, P, ema=0.9):
    """Gradient-norm balancing between groups (Wang et al. style, as in the thesis PINN's adaptive weights)."""
    g = loss_groups(model, P)
    params = [p for p in model.parameters()]
    norms = {}
    for k in GROUPS:
        gr = torch.autograd.grad(g[k], params, retain_graph=True, allow_unused=True)
        norms[k] = float(torch.sqrt(sum((x ** 2).sum() for x in gr if x is not None))) + 1e-30
    ref = max(norms.values())
    for k in GROUPS:
        WEIGHTS[k] = ema * WEIGHTS[k] + (1 - ema) * ref / norms[k]


# ----------------------------------------------------------------------------- scoring
def score(model, fem):
    xi = torch.tensor(np.stack([fem["xy"][:, 0] / L, fem["xy"][:, 1] / H], 1), dtype=DT, device=DEV)
    with torch.no_grad():
        o = model(xi).double().cpu().numpy()
        tip = float(model(torch.tensor([[1.0, 0.0]], dtype=DT, device=DEV))[0, 1]) * SC_V
    u, v, p = o[:, 0] * SC_U, o[:, 1] * SC_V, o[:, 2] * V

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    return {"u_relL2": rel(u, fem["u"]), "v_relL2": rel(v, fem["v"]), "phi_relL2": rel(p, fem["phi"]),
            "tip_v": tip, "tip_v_fem": float(fem["v"][np.argmin(np.abs(fem["xy"][:, 0] - L) + fem["xy"][:, 1])])}


def build(arm, width, seed):
    torch.manual_seed(seed)
    net = MLP([2, 100, 250, 8]) if arm == "mlp" else RBFKAN(2, width, 8)
    return Piezo(net.to(device=DEV, dtype=DT))


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.fem)
    fem = {k: z[k] for k in ("xy", "u", "v", "phi")}
    P = make_pools(a.seed, a.n_col, a.n_bd)
    model = build(a.arm, a.width, a.seed)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    hist = []
    t0 = time.perf_counter()
    for ep in range(1, a.adam_steps + 1):
        if a.balance and ep % a.balance == 1:
            rebalance(model, P)
        opt.zero_grad(set_to_none=True)
        l, parts = loss_fn(model, P)
        l.backward()
        opt.step()
        if ep % 100 == 0 or ep == a.adam_steps:
            hist.append({"stage": "adam", "step": ep, "loss": float(l.detach()), **parts})
    res = {"status": STATUS, "arm": a.arm, "seed": a.seed, "params": params, "adam_steps": a.adam_steps,
           "adam_seconds": time.perf_counter() - t0, "dtype": str(DT), "vsplit": VSPLIT, "balance": a.balance, "lbfgs_weights": a.lbfgs_weights, "weights_final": dict(WEIGHTS), "term_w": dict(TERM_W), "score_adam_final": score(model, fem),
           "scientific_accepted": False, "reference_accepted": False}
    print(json.dumps({"arm": a.arm, "seed": a.seed, "params": params, "adam": res["score_adam_final"]}), flush=True)
    if a.lbfgs_weights == "unit":
        for k in GROUPS:
            WEIGHTS[k] = 1.0
    lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=25, max_eval=125, tolerance_grad=1e-12, tolerance_change=0.0,
                           history_size=100, line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad(set_to_none=True)
        l, _ = loss_fn(model, P)
        l.backward()
        return l

    last = float(loss_fn(model, P)[0].detach())
    acc = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t1 = time.perf_counter()
    it, plateau, status = 0, 0, "lbfgs_time_cap"
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
        last = cur
        it += dn
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
    res.update({"lbfgs_status": status, "lbfgs_iterations": it, "lbfgs_seconds": time.perf_counter() - t1,
                "score_lbfgs_final": score(model, fem)})
    torch.save({"state_dict": model.state_dict()}, out / "checkpoint-lbfgs-final.pt")
    (out / "history.json").write_text(json.dumps(hist))
    (out / "run.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"arm": a.arm, "seed": a.seed, "lbfgs": res["score_lbfgs_final"], "status": status, "iters": it}), flush=True)


def smoke(a):
    """Tier 0: pools, hard BCs, finite loss/gradients, short descent."""
    P = make_pools(1, 256, 32)
    for arm in ("mlp", "rbf"):
        m = build(arm, a.width, 1)
        xi = torch.tensor([[0.0, 0.3], [0.5, 0.0], [0.5, 1.0]], dtype=DT, device=DEV)
        o = m(xi)
        assert torch.allclose(o[0, :2], torch.zeros(2, dtype=DT, device=DEV)), "u,v must vanish at x=0"
        assert abs(float(o[1, 2])) < 1e-12 and abs(float(o[2, 2]) - 1.0) < 1e-12, "phi must be 0 at bottom and V at top"
        l0, _ = loss_fn(m, P)
        l0.backward()
        g = torch.cat([p.grad.flatten() for p in m.parameters()])
        assert torch.isfinite(l0) and torch.isfinite(g).all()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(50):
            opt.zero_grad()
            l, _ = loss_fn(m, P)
            l.backward()
            opt.step()
        assert float(l) < float(l0), "loss must decrease"
        print(arm, "OK", float(l0), "->", float(l), "params", sum(p.numel() for p in m.parameters()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("smoke", "run"))
    ap.add_argument("--arm", choices=("mlp", "rbf"), default="rbf")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vsplit", type=float, default=0.0)
    ap.add_argument("--balance", type=int, default=0, help="rebalance group weights every N Adam steps (0=off)")
    ap.add_argument("--lbfgs-weights", choices=("frozen", "unit"), default="frozen", help="group weights during L-BFGS")
    ap.add_argument("--w-tau", type=float, default=1.0)
    ap.add_argument("--w-sz", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--fem")
    ap.add_argument("--out")
    ap.add_argument("--n-col", type=int, default=4096)
    ap.add_argument("--n-bd", type=int, default=256)
    ap.add_argument("--adam-steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs-seconds", type=float, default=300.0)
    ap.add_argument("--lbfgs-max-iter", type=int, default=15000)
    args = ap.parse_args()
    DT = torch.float32 if args.dtype == "float32" else torch.float64
    DEV = torch.device(args.device)
    VSPLIT = args.vsplit
    TERM_W["tau"], TERM_W["sz"] = args.w_tau, args.w_sz
    (smoke if args.mode == "smoke" else run)(args)
