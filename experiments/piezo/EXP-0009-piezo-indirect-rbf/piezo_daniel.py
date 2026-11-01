#!/usr/bin/env python3
"""DRAFT_UNVERIFIED: faithful re-implementation (equations and recipe, not code) of the indirect PINN of Gonzalez Carrillo / PINN_pz_v3.ipynb.

Recipe read from `PINN_pz_v3.ipynb`, `geom_creation.ipynb`, `scripts/train_indirect.py` and the README of PINNs_piezoelectricity (commit b90b19c):
raw SI coordinates (metres), 8 raw outputs (u, v, phi, sx, sz, txz, Dx, Dy), hard BC u=v=0 at x=0 and phi = V/H*y + y(y-H)*N,
collocation = meshgrid of two sorted Latin-hypercube vectors (150 x 150 = 22,500 points), 400 LHS points on the left/right faces,
per-point coefficients with his dataset convention (bottom layer e31=+0.02904, e33=+0.05158 after the loader's e33 sign flip; top layer opposite;
eps1 = eps2 = -1.0406e-10), loss = MSE(constitutive) + MSE(div) + MSE(BC) with FIXED unit weights (the adaptive update is never switched on in the notebook),
BC = free end (sx = tau = 0) and D.n = 0 on both ends (NO free-surface condition on top/bottom), Adam lr 1e-3 for 1000 epochs, then 200 L-BFGS
'epochs' (lr 0.01, PyTorch defaults max_iter=20, no line search, grad-norm clip 0.5), float64. Arms: mlp (2-100-250-8 pyramid, tanh, raw inputs),
mlpn (same on normalised inputs), rbf (RBF-PIKAN width W, normalised inputs). Reference: Daniel's FEM, poling_sign=-1, 201x21 grid.
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

STATUS = "DRAFT_UNVERIFIED"
W_, H_, V_ = 0.1, 1e-3, 100.0
C11, C12, C22, G = pz.C11, pz.C12, pz.C22, pz.G
EPS = -pz.EPS  # his epsilon_1 = epsilon_2 = -D_const_strain
E31B, E33B = abs(pz.E31), abs(pz.E33)  # bottom layer after the loader flip: e31=+0.02904, e33=+0.05158
DT, DEV = torch.float64, torch.device("cpu")


class Net(nn.Module):
    def __init__(self, arm, width):
        super().__init__()
        self.arm = arm
        self.core = pz.RBFKAN(2, width, 8) if arm == "rbf" else pz.MLP([2, 100, 250, 8])
        self.register_buffer("scale", torch.tensor([W_, H_]))

    def forward(self, xy):
        if self.arm == "mlp":  # raw metres, faithful (undo MLP's 2*xi-1 map: it expects xi in [0,1])
            o = self._raw(xy)
        else:
            o = self.core(xy / self.scale)
        x, y = xy[:, 0:1], xy[:, 1:2]
        return torch.cat([x * o[:, 0:1], x * o[:, 1:2], y * (y - H_) * o[:, 2:3] + V_ / H_ * y, o[:, 3:]], dim=1)

    def _raw(self, xy):
        h = xy
        for layer in self.core.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.core.layers[-1](h)


def lhs_sorted(n, seed):
    return qmc.LatinHypercube(d=1, seed=seed).random(n)[:, 0]


def make_pools(seed, n_grid=150, n_bc=400):
    x = np.sort(lhs_sorted(n_grid, seed) * W_)
    y = np.sort(lhs_sorted(n_grid, seed + 1) * H_)
    X, Y = np.meshgrid(x, y)
    col = np.stack([X.ravel(), Y.ravel()], 1)
    rng = np.random.default_rng(seed + 2)
    col = col[rng.permutation(len(col))]
    yl, yr = lhs_sorted(n_bc, seed + 3) * H_, lhs_sorted(n_bc, seed + 4) * H_
    t = lambda a: torch.tensor(a, dtype=DT, device=DEV)
    return {"col": t(col), "left": t(np.stack([np.zeros(n_bc), yl], 1)), "right": t(np.stack([np.full(n_bc, W_), yr], 1))}


def g(f, x):
    return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True, retain_graph=True)[0]


def loss_fn(model, P):
    xy = P["col"].clone().requires_grad_(True)
    o = model(xy)
    u, v, phi, sx, sz, tau, Dx, Dy = [o[:, i:i + 1] for i in range(8)]
    gu, gv, gp = g(u, xy), g(v, xy), g(phi, xy)
    exx, eyy, exy = gu[:, 0:1], gv[:, 1:2], 0.5 * (gu[:, 1:2] + gv[:, 0:1])
    Ex, Ey = -gp[:, 0:1], -gp[:, 1:2]
    sg = torch.where(xy[:, 1:2] < 0.5 * H_, 1.0, -1.0)
    e31, e33 = sg * E31B, sg * E33B
    sx_c = C11 * exx + C12 * eyy - e31 * Ey
    sz_c = C12 * exx + C22 * eyy - e33 * Ey
    tau_c = G * exy
    Dx_c = EPS * Ex
    Dy_c = e31 * exx + e33 * eyy + EPS * Ey
    gsx, gsz, gt, gDx, gDy = g(sx, xy), g(sz, xy), g(tau, xy), g(Dx, xy), g(Dy, xy)
    mech = (sx - sx_c).square().mean() + (sz - sz_c).square().mean() + (tau - tau_c).square().mean()
    elec = (Dx - Dx_c).square().mean() + (Dy - Dy_c).square().mean()
    div = (gsx[:, 0:1] + gt[:, 1:2]).square().mean() + (gt[:, 0:1] + gsz[:, 1:2]).square().mean() + (gDx[:, 0:1] + gDy[:, 1:2]).square().mean()
    orr, orl = model(P["right"]), model(P["left"])
    bc = orr[:, 3].square().mean() + orr[:, 5].square().mean() + orr[:, 6].square().mean() + orl[:, 6].square().mean()
    total = mech + elec + div + bc
    return total, {"mech": float(mech.detach()), "elec": float(elec.detach()), "div": float(div.detach()), "bc": float(bc.detach())}


def score(model, fem):
    xy = torch.tensor(fem["xy"], dtype=DT, device=DEV)
    with torch.no_grad():
        o = model(xy).double().cpu().numpy()
        tip = float(model(torch.tensor([[W_, 0.0]], dtype=DT, device=DEV))[0, 1])

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    return {"u_relL2": rel(o[:, 0], fem["u"]), "v_relL2": rel(o[:, 1], fem["v"]), "phi_relL2": rel(o[:, 2], fem["phi"]), "tip_v": tip,
            "tip_v_fem": float(fem["v"][np.argmin(np.abs(fem["xy"][:, 0] - W_) + fem["xy"][:, 1])])}


def build(arm, width, seed):
    torch.manual_seed(seed)
    return Net(arm, width).to(device=DEV, dtype=DT)


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.fem)
    fem = {k: z[k] for k in ("xy", "u", "v", "phi")}
    P = make_pools(a.seed, a.n_grid, a.n_bc)
    model = build(a.arm, a.width, a.seed)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    hist, snaps, t0 = [], [], time.perf_counter()
    for ep in range(1, a.adam_steps + 1):
        opt.zero_grad(set_to_none=True)
        l, parts = loss_fn(model, P)
        l.backward()
        opt.step()
        if ep % 100 == 0 or ep == a.adam_steps:
            hist.append({"stage": "adam", "step": ep, "loss": float(l.detach()), **parts})
            sc = score(model, fem)
            snaps.append({"step": ep, "u": sc["u_relL2"], "v": sc["v_relL2"], "phi": sc["phi_relL2"], "tip_v": sc["tip_v"]})
    res = {"status": STATUS, "formulation": "daniel_faithful", "arm": a.arm, "seed": a.seed, "params": params, "adam_steps": a.adam_steps,
           "adam_seconds": time.perf_counter() - t0, "n_collocation": int(len(P["col"])), "adam_snapshots": snaps,
           "score_adam_final": score(model, fem), "scientific_accepted": False, "reference_accepted": False}
    torch.save({"state_dict": model.state_dict()}, out / "checkpoint-adam-final.pt")
    print(json.dumps({"arm": a.arm, "seed": a.seed, "params": params, "adam": res["score_adam_final"], "snaps": snaps[-2:]}), flush=True)
    lb = torch.optim.LBFGS(model.parameters(), lr=a.lbfgs_lr)  # PyTorch defaults, as in the notebook
    t1, status, best = time.perf_counter(), "lbfgs_complete", None
    for ep in range(a.lbfgs_epochs):
        def closure():
            lb.zero_grad()
            l, _ = loss_fn(model, P)
            l.backward()
            return l

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        try:
            lb.step(closure)
        except Exception as e:  # noqa: BLE001
            status = f"lbfgs_exception:{type(e).__name__}"
            break
        if ep % 20 == 0 or ep == a.lbfgs_epochs - 1:
            lv = float(closure().detach())
            hist.append({"stage": "lbfgs", "step": ep, "loss": lv})
            sc = score(model, fem)
            snaps.append({"step": a.adam_steps + ep, "u": sc["u_relL2"], "v": sc["v_relL2"], "phi": sc["phi_relL2"], "tip_v": sc["tip_v"]})
            if not math.isfinite(lv):
                status = "lbfgs_nonfinite"
                break
        if time.perf_counter() - t1 > a.lbfgs_seconds:
            status = "lbfgs_time_cap"
            break
    res.update({"lbfgs_status": status, "lbfgs_seconds": time.perf_counter() - t1, "adam_snapshots": snaps, "score_lbfgs_final": score(model, fem)})
    torch.save({"state_dict": model.state_dict()}, out / "checkpoint-lbfgs-final.pt")
    (out / "history.json").write_text(json.dumps(hist))
    (out / "run.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"arm": a.arm, "seed": a.seed, "lbfgs": res["score_lbfgs_final"], "status": status}), flush=True)


def smoke(a):
    P = make_pools(1, 30, 40)
    for arm in ("mlp", "mlpn", "rbf"):
        m = build(arm, a.width, 1)
        xy = torch.tensor([[0.0, 3e-4], [0.05, 0.0], [0.05, 1e-3]], dtype=DT)
        o = m(xy)
        assert torch.allclose(o[0, :2], torch.zeros(2, dtype=DT)) and abs(float(o[1, 2])) < 1e-12 and abs(float(o[2, 2]) - V_) < 1e-9
        l0, _ = loss_fn(m, P)
        assert torch.isfinite(l0)
        print(arm, "OK", float(l0), "params", sum(p.numel() for p in m.parameters()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("smoke", "run"))
    ap.add_argument("--arm", choices=("mlp", "mlpn", "rbf"), default="mlp")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fem")
    ap.add_argument("--out")
    ap.add_argument("--n-grid", type=int, default=150)
    ap.add_argument("--n-bc", type=int, default=400)
    ap.add_argument("--adam-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs-epochs", type=int, default=200)
    ap.add_argument("--lbfgs-lr", type=float, default=0.01)
    ap.add_argument("--lbfgs-seconds", type=float, default=1800.0)
    args = ap.parse_args()
    DT = torch.float32 if args.dtype == "float32" else torch.float64
    DEV = torch.device(args.device)
    (smoke if args.mode == "smoke" else run)(args)
