"""Step 4 (local, CPU): can the RBF-KAN architecture even REPRESENT the DFG Re=20 FEM field?

Supervised regression (no physics loss, no PDE) of RBF-KAN variants onto the archived Taylor-Hood DFG-L3 nodal
values: velocity at P2 dof nodes, pressure at P1 nodes. Held-out = random 20 % of nodes never used in fitting.
If the current architecture (2-16-16-3, 11 Gaussians per edge) cannot reach a few-% relative L2 here, no
physics-informed training of it can. Variants change capacity only (width, bases per edge, geometry feature);
they are NOT baselines and nothing here is selected against the scoring grid.

Pressure uses a fixed gauge: the mean pressure over the training nodes is subtracted from both target and fit
(p is defined up to a constant; the outlet gauge is handled in the official scorer, not here).
Run: py -3.13 capacity_fit_dfg.py [--steps 6000]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2] / "src"))
from research_pipeline.sciml import dfg  # noqa: E402

L3 = ROOT / "kaggle-outputs/dfg-fem-field-followup-retry5-20260929/dfg-fem-field-followup-retry5/dfg/DFG-L3"
OUT = Path(__file__).parent
torch.set_num_threads(6)
torch.manual_seed(0)
np.random.seed(0)

co = np.load(L3 / "fem_coefficients.npz")
vxy = co["velocity_dof_coordinates"][:, :2]
vel = co["velocity_coefficients"].reshape(-1, 2)
pxy = co["pressure_dof_coordinates"][:, :2]
pval = co["pressure_coefficients"]
U, PS = dfg.U_MEAN, dfg.PRESSURE_SCALE
pmean = pval.mean()


class Edge(nn.Module):
    def __init__(self, i, o, nb):
        super().__init__()
        self.register_buffer("c", torch.linspace(-1.0, 1.0, nb))
        self.register_buffer("w", torch.tensor(2.0 / (nb - 1) * 1.75))  # width ~ 1.75x spacing (as 0.35 for nb=11)
        self.coef = nn.Parameter(torch.randn(o, i, nb) * 0.1 / math.sqrt(i))
        self.base = nn.Parameter(torch.randn(o, i) * 0.1 / math.sqrt(i))

    def forward(self, x):
        b = torch.exp(-((x.unsqueeze(-1) - self.c) / self.w).square())
        return torch.einsum("big,oig->bo", b, self.coef) + torch.nn.functional.linear(torch.nn.functional.silu(x), self.base)


class Net(nn.Module):
    def __init__(self, width, nb, geom):
        super().__init__()
        self.geom = geom
        nin = 3 if geom else 2
        self.layers = nn.ModuleList([Edge(nin, width, nb), Edge(width, width, nb), Edge(width, 3, nb)])

    def forward(self, xy):
        x = 2.0 * xy[:, 0:1] / dfg.DOMAIN_X[1] - 1.0
        y = 2.0 * xy[:, 1:2] / dfg.DOMAIN_Y[1] - 1.0
        feats = [x, y]
        if self.geom:  # distance to cylinder centre, saturating at 0.3 -> known geometry, not the solution
            d = torch.sqrt((xy[:, 0:1] - 0.2) ** 2 + (xy[:, 1:2] - 0.2) ** 2)
            feats.append(2.0 * torch.clamp(d / 0.3, 0, 1) - 1.0)
        h = torch.cat(feats, 1)
        for l in self.layers[:-1]:
            h = torch.tanh(l(h))
        return self.layers[-1](h)


def split(n, frac=0.2):
    idx = np.random.permutation(n)
    k = int(n * frac)
    return idx[k:], idx[:k]


vtr, vte = split(len(vxy))
ptr, pte = split(len(pxy))
Xv = torch.tensor(vxy, dtype=torch.float32)
Yv = torch.tensor(vel / U, dtype=torch.float32)
Xp = torch.tensor(pxy, dtype=torch.float32)
Yp = torch.tensor((pval - pmean) / PS, dtype=torch.float32)


def rel(pred, true):
    return float(torch.linalg.norm(pred - true) / torch.linalg.norm(true))


def fit(name, width, nb, geom, steps, batch=4096):
    net = Net(width, nb, geom)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    t0 = time.perf_counter()
    for s in range(steps):
        iv = torch.tensor(np.random.choice(vtr, batch))
        ip = torch.tensor(np.random.choice(ptr, batch))
        out_v = net(Xv[iv])[:, :2]
        out_p = net(Xp[ip])[:, 2:3]
        loss = (out_v - Yv[iv]).square().mean() + (out_p - Yp[ip].unsqueeze(1)).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        pv = net(Xv[vte])[:, :2]
        pp = net(Xp[pte])[:, 2:3]
        trv = net(Xv[vtr[:60000]])[:, :2]
        trp = net(Xp[ptr[:60000]])[:, 2:3]
        res = {
            "name": name, "width": width, "bases": nb, "geom_feature": geom, "params": npar, "steps": steps,
            "heldout_u_relL2": rel(pv[:, 0], Yv[vte][:, 0]), "heldout_v_relL2": rel(pv[:, 1], Yv[vte][:, 1]),
            "heldout_p_relL2_gauge_mean": rel(pp[:, 0], Yp[pte]),
            "train_u_relL2": rel(trv[:, 0], Yv[vtr[:60000]][:, 0]),
            "train_p_relL2_gauge_mean": rel(trp[:, 0], Yp[ptr[:60000]]),
            "seconds": time.perf_counter() - t0,
        }
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    variants = [("current_2-16-16-3_b11", 16, 11, False), ("w16_b11_geom", 16, 11, True),
                ("w32_b11", 32, 11, False), ("w32_b21_geom", 32, 21, True), ("w64_b21_geom", 64, 21, True)]
    results = []
    for v in variants:
        results.append(fit(*v, steps=a.steps))
        json.dump(results, open(OUT / "capacity-fit-dfg.json", "w"), indent=2)
