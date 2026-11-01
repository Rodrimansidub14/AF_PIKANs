#!/usr/bin/env python3
"""DRAFT_UNVERIFIED: DFG 2D-1 (Re=20, steady) under the configuration of Gonzalez Carrillo's 2024 UVG thesis (section 5.1.2):
channel 2.2 x 0.41, cylinder r=0.05 at (0.2,0.2), nu=0.001, inlet u=4*0.3*y*(0.41-y)/0.41^2, LHS sampling (318 boundary points,
22,345 collocation points), MLP 4x50 tanh with 3 raw outputs (u,v,p), loss = MSE(NSx)+MSE(NSy)+MSE(cont) + MSE(BC),
Adam 2000 epochs lr 1e-3 then L-BFGS 1000 iterations lr 0.1 (interpreted as 1000 single-iteration steps, no line search).
Variants:
  literal   : boundary points on inlet, walls and cylinder ONLY, pressure target 0 at ALL of them (as drawn in the thesis, Fig. 19);
              no outlet condition.  Kept to show what the published setup yields; it is not the DFG benchmark.
  corrected : velocity BC on inlet/walls/cylinder, FEATFLOW do-nothing outlet (nu*u_x - p = 0, nu*v_x = 0) on the outlet, no p pins.
Arms: mlp (4x50, raw coordinates) and rbf (RBF-PIKAN, width --width, bases 11, same raw outputs).  Pools shared per seed.
Scored post hoc against the FEM DFG-L3 working reference (221x42 grid, outlet-gauge pressure) and with the FEATFLOW sigma.n QoIs.
Modes: worker | campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pikan_campaign import MLP, RBFKAN, grad  # noqa: E402

STATUS = "DRAFT_UNVERIFIED"
L, H, CX, CY, R = 2.2, 0.41, 0.2, 0.2, 0.05
NU, UMAX, UMEAN, DIAM = 0.001, 0.3, 0.2, 0.1
FEAT = {"Cd": 5.57953523384, "Cl": 0.010618948146, "dp": 0.11752016697}
N_BC, N_COL = 318, 22345


def lhs(n, d, seed):
    return qmc.LatinHypercube(d=d, seed=seed).random(n)


def make_pools(seed, variant, dtype, device):
    pts = lhs(32000, 2, 10 * seed + 1) * np.array([L, H])
    pts = pts[np.hypot(pts[:, 0] - CX, pts[:, 1] - CY) > R][:N_COL]
    assert len(pts) == N_COL
    lengths = {"inlet": H, "top": L, "bottom": L, "cyl": 2 * math.pi * R}
    if variant == "corrected":
        lengths["outlet"] = H
    base = {k: v for k, v in lengths.items() if k != "outlet"}
    tot = sum(base.values())
    counts = {k: int(round(N_BC * v / tot)) for k, v in base.items()}
    counts["top"] += N_BC - sum(counts.values())
    if variant == "corrected":
        counts["outlet"] = max(8, int(round(N_BC * H / tot)))
    bc = {}
    for i, (k, n) in enumerate(counts.items()):
        s = lhs(n, 1, 10 * seed + 3 + 100 * i)[:, 0]
        if k == "inlet": xy = np.stack((np.zeros(n), s * H), 1)
        elif k == "outlet": xy = np.stack((np.full(n, L), s * H), 1)
        elif k == "top": xy = np.stack((s * L, np.full(n, H)), 1)
        elif k == "bottom": xy = np.stack((s * L, np.zeros(n)), 1)
        else:
            th = 2 * math.pi * s
            xy = np.stack((CX + R * np.cos(th), CY + R * np.sin(th)), 1)
        bc[k] = xy
    t = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
    return {"col": t(pts), **{k: t(v) for k, v in bc.items()}}, {k: len(v) for k, v in bc.items()}


def residuals(model, xy):
    xy = xy.detach().requires_grad_(True)
    o = model(xy)
    u, v, p = o[:, 0:1], o[:, 1:2], o[:, 2:3]
    gu, gv, gp = grad(u, xy), grad(v, xy), grad(p, xy)
    uxx = grad(gu[:, 0:1], xy)[:, 0:1]; uyy = grad(gu[:, 1:2], xy)[:, 1:2]
    vxx = grad(gv[:, 0:1], xy)[:, 0:1]; vyy = grad(gv[:, 1:2], xy)[:, 1:2]
    nsx = u * gu[:, 0:1] + v * gu[:, 1:2] + gp[:, 0:1] - NU * (uxx + uyy)
    nsy = u * gv[:, 0:1] + v * gv[:, 1:2] + gp[:, 1:2] - NU * (vxx + vyy)
    return nsx, nsy, gu[:, 0:1] + gv[:, 1:2]


def loss_fn(model, P, variant):
    nsx, nsy, cont = residuals(model, P["col"])
    phys = nsx.square().mean() + nsy.square().mean() + cont.square().mean()
    sets = ["inlet", "top", "bottom", "cyl"]
    bxy = torch.cat([P[k] for k in sets])
    bo = model(bxy)
    tu = torch.zeros(len(bxy), 1, dtype=bxy.dtype, device=bxy.device)
    n_in = len(P["inlet"])
    y = P["inlet"][:, 1:2]
    tu[:n_in] = 4.0 * UMAX * y * (H - y) / H ** 2
    bc = (bo[:, 0:1] - tu).square().mean() + bo[:, 1:2].square().mean()
    if variant == "literal":
        bc = bc + bo[:, 2:3].square().mean()  # p target 0 at every boundary point, as in the thesis figure
    else:
        xo = P["outlet"].detach().requires_grad_(True)
        oo = model(xo)
        gu = grad(oo[:, 0:1], xo); gv = grad(oo[:, 1:2], xo)
        bc = bc + (NU * gu[:, 0:1] - oo[:, 2:3]).square().mean() + (NU * gv[:, 0:1]).square().mean()
    return phys + bc, {"phys": float(phys.detach()), "bc": float(bc.detach())}


def qoi(model, dev):
    n = 2048
    th = 2 * math.pi * torch.arange(n, dtype=torch.float64, device=dev) / n
    xy = torch.stack((CX + R * torch.cos(th), CY + R * torch.sin(th)), 1).requires_grad_(True)
    o = model(xy)
    gu, gv = grad(o[:, 0:1], xy), grad(o[:, 1:2], xy)
    nx, ny = torch.cos(th), torch.sin(th)
    ds = 2 * math.pi * R / n
    p = o[:, 2]
    fd = torch.sum(NU * (gu[:, 0] * nx + gu[:, 1] * ny) - p * nx) * ds
    fl = torch.sum(NU * (gv[:, 0] * nx + gv[:, 1] * ny) - p * ny) * ds
    sc = 2.0 / (UMEAN ** 2 * DIAM)
    with torch.no_grad():
        pf = model(torch.tensor([[0.15, 0.2]], dtype=torch.float64, device=dev))[0, 2]
        pr = model(torch.tensor([[0.25, 0.2]], dtype=torch.float64, device=dev))[0, 2]
    return {"Cd": float(sc * fd), "Cl": float(sc * fl), "dp": float(pf - pr)}


def score(model, fem, dev):
    m = model.double()
    xy = torch.as_tensor(fem["xy"], dtype=torch.float64, device=dev)
    with torch.no_grad():
        o = m(xy).cpu().numpy()
        anchor = m(torch.tensor([[L, 0.2]], dtype=torch.float64, device=dev))[0, 2].item()
    u, v, p = o[:, 0], o[:, 1], o[:, 2] - anchor
    ru = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    q = qoi(m, dev)
    return {"u_relL2": ru(u, fem["velocity"][:, 0]), "v_relL2": ru(v, fem["velocity"][:, 1]),
            "p_relL2_gauge": ru(p, fem["pressure_gauge_fixed"]), **q,
            "Cd_relerr": abs(q["Cd"] - FEAT["Cd"]) / FEAT["Cd"], "Cl_abserr": abs(q["Cl"] - FEAT["Cl"]),
            "dp_relerr": abs(q["dp"] - FEAT["dp"]) / FEAT["dp"]}


def worker(a):
    dev = torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    z = np.load(a.fem)
    fem = {k: z[k] for k in ("xy", "velocity", "pressure_gauge_fixed")}
    dt = torch.float32
    P, counts = make_pools(a.seed, a.variant, dt, dev)
    torch.manual_seed(a.seed)
    if a.arm == "mlp":
        model = MLP([2, 50, 50, 50, 50, 3])
    else:
        model = RBFKAN(2, a.width, 3, 11, [0.0, 0.0], [L, H])
    model = model.to(device=dev, dtype=dt)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.perf_counter(); hist = []
    for ep in range(1, a.adam_steps + 1):
        opt.zero_grad(set_to_none=True)
        l, parts = loss_fn(model, P, a.variant); l.backward(); opt.step()
        if ep % 100 == 0 or ep == a.adam_steps:
            hist.append({"stage": "adam", "step": ep, "loss": float(l.detach()), **parts})
    if dev.type == "cuda": torch.cuda.synchronize()
    t_adam = time.perf_counter() - t0
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "stage": "adam_final"}, out / "checkpoint-adam-final.pt")
    res = {"status": STATUS, "variant": a.variant, "arm": a.arm, "seed": a.seed, "width": a.width if a.arm == "rbf" else None,
           "params": params, "bc_counts": counts, "n_collocation": N_COL, "adam_steps": a.adam_steps, "adam_seconds": t_adam,
           "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu", "dtype": "float32",
           "fem_reference_sha256": hashlib.sha256(Path(a.fem).read_bytes()).hexdigest(),
           "score_adam_final": score(model, fem, dev), "scientific_accepted": False, "reference_accepted": False}
    model.float()
    print(json.dumps({"seed": a.seed, "arm": a.arm, "variant": a.variant, "adam": res["score_adam_final"]}), flush=True)
    model.to(dt)
    lb = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=1)
    t1 = time.perf_counter(); status = "lbfgs_complete"
    for it in range(1, a.lbfgs_steps + 1):
        def closure():
            lb.zero_grad(set_to_none=True)
            l, _ = loss_fn(model, P, a.variant); l.backward(); return l
        try:
            lv = lb.step(closure)
        except Exception as e:  # noqa: BLE001
            status = f"lbfgs_exception:{type(e).__name__}"; break
        if not math.isfinite(float(lv)):
            status = "lbfgs_nonfinite"; break
        if it % 100 == 0 or it == a.lbfgs_steps:
            hist.append({"stage": "lbfgs", "step": it, "loss": float(lv)})
    if dev.type == "cuda": torch.cuda.synchronize()
    res.update({"lbfgs_status": status, "lbfgs_seconds": time.perf_counter() - t1, "lbfgs_steps": a.lbfgs_steps})
    torch.save({"state_dict": model.state_dict(), "stage": "lbfgs_final"}, out / "checkpoint-lbfgs-final.pt")
    res["score_lbfgs_final"] = score(model, fem, dev)
    (out / "history.json").write_text(json.dumps(hist))
    (out / "run.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"seed": a.seed, "arm": a.arm, "variant": a.variant, "lbfgs": res["score_lbfgs_final"], "status": status}), flush=True)


def campaign(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_gpu = max(1, torch.cuda.device_count())
    queue = [(v, arm, s) for s in a.seeds for v in a.variants for arm in a.arms]
    running, ledger, t0 = {}, [], time.monotonic()
    while queue or running:
        for g in list(running):
            pr, job, st = running[g]
            if pr.poll() is not None:
                ledger.append({"job": job, "rc": pr.returncode, "wall": time.monotonic() - st}); del running[g]
        if time.monotonic() - t0 > a.wall_cap: queue = []
        for g in range(min(n_gpu, 2)):
            if g not in running and queue:
                v, arm, s = queue.pop(0)
                tag = f"{v}-{arm}-seed-{s}"
                cmd = [sys.executable, __file__, "worker", "--variant", v, "--arm", arm, "--seed", str(s), "--width", str(a.width),
                       "--device", f"cuda:{g}", "--fem", a.fem, "--out", str(out / tag), "--adam-steps", str(a.adam_steps),
                       "--lbfgs-steps", str(a.lbfgs_steps)]
                running[g] = (subprocess.Popen(cmd, stdout=open(out / f"{tag}.log", "w"), stderr=subprocess.STDOUT), tag, time.monotonic())
        (out / "ledger.json").write_text(json.dumps({"status": STATUS, "runs": ledger, "running": [j for _, j, _ in running.values()]}))
        time.sleep(2)
    (out / "ledger.json").write_text(json.dumps({"status": STATUS, "runs": ledger, "complete": True, "wall": time.monotonic() - t0}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("worker", "campaign"))
    ap.add_argument("--variant", choices=("literal", "corrected"), default="corrected")
    ap.add_argument("--variants", nargs="+", default=["literal", "corrected"])
    ap.add_argument("--arm", choices=("mlp", "rbf"), default="mlp")
    ap.add_argument("--arms", nargs="+", default=["mlp", "rbf"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[801, 802, 803, 804, 805])
    ap.add_argument("--seed", type=int, default=801)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adam-steps", type=int, default=2000)
    ap.add_argument("--lbfgs-steps", type=int, default=1000)
    ap.add_argument("--wall-cap", type=float, default=1e9)
    a = ap.parse_args()
    {"worker": worker, "campaign": campaign}[a.mode](a)
