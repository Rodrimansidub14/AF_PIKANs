#!/usr/bin/env python3
"""DRAFT_UNVERIFIED single-file GPU campaign: PINNacle NS2d-CG (backward step, Re=100) and the DeepXDE
Allen-Cahn contract, RBF-KAN vs tanh-MLP, plus a speed bench.  Exploratory; nothing here is accepted.

Modes
  bench     time one training step per architecture (choose network size before spending GPU-hours)
  campaign  run a paired queue (arm x seed) over the visible GPUs, one worker process per GPU
  worker    one (problem, arm, seed) run; Adam stage then optional L-BFGS stage; scores both endpoints

Contracts (see README.md next to this file for sources)
  ns   : PINNacle NS2d-CG: domain [0,4]x[0,2] minus [0,2]x[1,2]; nu=0.01; inlet u=4y(1-y), v=0 at x=0;
         p=0 at x=4; no-slip elsewhere; 8192 domain + 2048 boundary points; uniform loss weights; Adam 1e-3,
         20000 full-batch iterations; MLP = tanh 5x100 Glorot-normal (vanilla).  Scored on the COMSOL file
         ref/ns_0_obstacle.dat (columns x y u v p): L2RE per component, their mean, and flattened.
  ac   : DeepXDE Allen-Cahn demo: u_t = 1e-3 u_xx + 5(u-u^3), x in [-1,1], t in [0,1], IC x^2 cos(pi x),
         u(+-1,t) = -1 imposed by the hard output transform; 8000 domain + 400 boundary + 800 initial residual
         points; MLP tanh 3x20 Glorot-normal; Adam 1e-3 40000 full-batch iterations, then L-BFGS.  Scored on the
         third-party DeepXDE Allen_Cahn.mat (201x101) with DeepXDE's unweighted relative L2.
Both problems use scrambled Sobol pools shared between arms for the same seed.  Deviation from PINNacle /
DeepXDE pipelines (own PyTorch implementation, Sobol pools) is declared, not hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc
from torch import nn

STATUS = "DRAFT_UNVERIFIED"


# ------------------------------------------------------------------ models
class RBFEdge(nn.Module):
    """Fixed Gaussian basis per edge + SiLU residual path (same family as the DFG/Heat/AC RBF-PIKAN)."""

    def __init__(self, inputs: int, outputs: int, bases: int):
        super().__init__()
        spacing = 2.0 / (bases - 1)
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, bases))
        self.register_buffer("width", torch.tensor(1.75 * spacing))  # 0.35 for 11 bases
        self.coefficients = nn.Parameter(torch.randn(outputs, inputs, bases) * (0.1 / math.sqrt(inputs)))
        self.base = nn.Parameter(torch.randn(outputs, inputs) * (0.1 / math.sqrt(inputs)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width).square())
        return torch.einsum("big,oig->bo", basis, self.coefficients) + torch.nn.functional.linear(
            torch.nn.functional.silu(x), self.base)


class RBFKAN(nn.Module):
    def __init__(self, n_in, width, n_out, bases, lo, hi):
        super().__init__()
        self.layers = nn.ModuleList([RBFEdge(n_in, width, bases), RBFEdge(width, width, bases),
                                     RBFEdge(width, n_out, bases)])
        self.register_buffer("lo", torch.tensor(lo))
        self.register_buffer("hi", torch.tensor(hi))

    def forward(self, xy):
        h = 2.0 * (xy - self.lo) / (self.hi - self.lo) - 1.0
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)


class MLP(nn.Module):
    """DeepXDE FNN: tanh, Glorot normal weights, zero biases, raw inputs."""

    def __init__(self, widths):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)


def sobol(count, dim, seed):
    return qmc.Sobol(d=dim, scramble=True, seed=seed).random_base2(int(round(math.log2(count))))


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]


# ------------------------------------------------------------------ NS2d-CG
NU = 0.01


class NSProblem:
    name = "ns2d_cg"
    lo, hi, n_in, n_out = [0.0, 0.0], [4.0, 2.0], 2, 3
    adam_steps_default = 20000

    def __init__(self, ref_path):
        self.ref = np.loadtxt(ref_path, comments="%", encoding="utf-8").astype(np.float64)  # x y u v p
        assert self.ref.shape[1] == 5

    def build(self, arm, width, bases, seed):
        torch.manual_seed(seed)
        if arm == "mlp":
            return MLP([2] + [100] * 5 + [3])
        return RBFKAN(2, width, 3, bases, self.lo, self.hi)

    def pools(self, seed, device, dtype):
        raw = sobol(16384, 2, 10 * seed + 1) * np.array([4.0, 2.0])
        keep = ~((raw[:, 0] < 2.0) & (raw[:, 1] > 1.0))
        dom = raw[keep][:8192]
        assert len(dom) == 8192
        s = sobol(2048, 1, 10 * seed + 3)[:, 0] * 12.0
        # perimeter order: inlet (1) | bottom (4) | outlet (2) | top-right (2) | step face x=2 (1) | step top y=1 (2)
        pts, tag = np.zeros((2048, 2)), np.zeros(2048, dtype=int)  # tag 0 inlet, 1 outlet, 2 other
        for i, a in enumerate(s):
            if a < 1: pts[i], tag[i] = (0.0, a), 0
            elif a < 5: pts[i], tag[i] = (a - 1.0, 0.0), 2
            elif a < 7: pts[i], tag[i] = (4.0, a - 5.0), 1
            elif a < 9: pts[i], tag[i] = (4.0 - (a - 7.0), 2.0), 2
            elif a < 10: pts[i], tag[i] = (2.0, 2.0 - (a - 9.0)), 2
            else: pts[i], tag[i] = (2.0 - (a - 10.0), 1.0), 2
        t = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
        return {"domain": t(dom), "inlet": t(pts[tag == 0]), "outlet": t(pts[tag == 1]), "other": t(pts[tag == 2])}

    def loss(self, model, P):
        xy = P["domain"].detach().requires_grad_(True)
        out = model(xy)
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        gu, gv, gp = grad(u, xy), grad(v, xy), grad(p, xy)
        uxx = grad(gu[:, 0:1], xy)[:, 0:1]
        uyy = grad(gu[:, 1:2], xy)[:, 1:2]
        vxx = grad(gv[:, 0:1], xy)[:, 0:1]
        vyy = grad(gv[:, 1:2], xy)[:, 1:2]
        mx = u * gu[:, 0:1] + v * gu[:, 1:2] + gp[:, 0:1] - NU * (uxx + uyy)
        my = u * gv[:, 0:1] + v * gv[:, 1:2] + gp[:, 1:2] - NU * (vxx + vyy)
        ct = gu[:, 0:1] + gv[:, 1:2]
        yi = P["inlet"][:, 1:2]
        oi, oo, ot = model(P["inlet"]), model(P["outlet"]), model(P["other"])
        terms = {"momentum_x": mx.square().mean(), "momentum_y": my.square().mean(), "continuity": ct.square().mean(),
                 "inlet_u": (oi[:, 0:1] - 4.0 * yi * (1.0 - yi)).square().mean(), "inlet_v": oi[:, 1:2].square().mean(),
                 "outlet_p": oo[:, 2:3].square().mean(), "other_u": ot[:, 0:1].square().mean(),
                 "other_v": ot[:, 1:2].square().mean()}
        return sum(terms.values()), terms

    @torch.no_grad()
    def score(self, model, device, dtype):
        xy = torch.as_tensor(self.ref[:, :2], dtype=dtype, device=device)
        pred = model(xy).double().cpu().numpy()
        true = self.ref[:, 2:5]
        comp = [float(np.linalg.norm(pred[:, i] - true[:, i]) / np.linalg.norm(true[:, i])) for i in range(3)]
        flat = float(np.linalg.norm(pred - true) / np.linalg.norm(true))
        return {"l2re_u": comp[0], "l2re_v": comp[1], "l2re_p": comp[2], "l2re_mean_components": sum(comp) / 3,
                "l2re_flat": flat}


# ------------------------------------------------------------------ Allen-Cahn (DeepXDE contract)
class ACProblem:
    name = "allen_cahn"
    lo, hi, n_in, n_out = [-1.0, 0.0], [1.0, 1.0], 2, 1
    adam_steps_default = 40000

    def __init__(self, ref_path):
        import scipy.io as sio
        m = sio.loadmat(ref_path)
        self.x, self.t, self.u = m["x"].reshape(-1), m["t"].reshape(-1), m["u"]  # u (nt, nx)

    def build(self, arm, width, bases, seed):
        torch.manual_seed(seed)
        if arm == "mlp":
            return MLP([2, 20, 20, 20, 1])
        return RBFKAN(2, width, 1, bases, self.lo, self.hi)

    @staticmethod
    def transform(xt, raw):
        x, t = xt[:, 0:1], xt[:, 1:2]
        return x.square() * torch.cos(math.pi * x) + (1.0 - x.square()) * t * raw

    def pools(self, seed, device, dtype):
        d = sobol(8192, 2, 10 * seed + 1)[:8000]
        d[:, 0] = 2 * d[:, 0] - 1
        b = sobol(512, 1, 10 * seed + 3)[:400, 0]
        bnd = np.stack((np.where(np.arange(400) % 2 == 0, -1.0, 1.0), b), 1)
        i = sobol(1024, 1, 10 * seed + 4)[:800, 0]
        ini = np.stack((2 * i - 1, np.zeros(800)), 1)
        allp = np.concatenate((d, bnd, ini))
        return {"residual": torch.as_tensor(allp, dtype=dtype, device=device)}

    def loss(self, model, P):
        xt = P["residual"].detach().requires_grad_(True)
        u = self.transform(xt, model(xt))
        g = grad(u, xt)
        uxx = grad(g[:, 0:1], xt)[:, 0:1]
        res = g[:, 1:2] - 0.001 * uxx - 5.0 * (u - u.pow(3))
        term = res.square().mean()
        return term, {"pde": term}

    @torch.no_grad()
    def score(self, model, device, dtype):
        xx, tt = np.meshgrid(self.x, self.t, indexing="xy")
        xt = torch.as_tensor(np.stack((xx.reshape(-1), tt.reshape(-1)), 1), dtype=dtype, device=device)
        pred = self.transform(xt, model(xt)).double().cpu().numpy().reshape(self.u.shape)
        err = pred - self.u
        return {"l2re_mat": float(np.linalg.norm(err) / np.linalg.norm(self.u)),
                "max_abs_mat": float(np.abs(err).max())}


# ------------------------------------------------------------------ Burgers 1D (PINNacle contract, Cole-Hopf reference)
class BurgersProblem:
    name = "burgers_pinnacle_1d"
    lo, hi, n_in, n_out = [-1.0, 0.0], [1.0, 1.0], 2, 1
    adam_steps_default = 20000
    nu = 0.01 / math.pi
    n_col = 8192

    def __init__(self, ref_path):
        z = np.load(ref_path)
        self.x, self.t, self.u = z["x"], z["t"], z["u"]  # u (nt, nx)

    def build(self, arm, width, bases, seed):
        torch.manual_seed(seed)
        if arm == "mlp":
            return MLP([2] + [100] * 5 + [1])  # PINNacle Vanilla architecture
        return RBFKAN(2, width, 1, bases, self.lo, self.hi)

    @staticmethod
    def transform(xt, raw):  # hard IC -sin(pi x) and u(+-1,t)=0
        x, t = xt[:, 0:1], xt[:, 1:2]
        return -torch.sin(math.pi * x) + (1.0 - x.square()) * t * raw

    def pools(self, seed, device, dtype):
        d = sobol(self.n_col, 2, 10 * seed + 1)
        d[:, 0] = 2 * d[:, 0] - 1
        return {"residual": torch.as_tensor(d, dtype=dtype, device=device)}

    def loss(self, model, P):
        xt = P["residual"].detach().requires_grad_(True)
        u = self.transform(xt, model(xt))
        g = grad(u, xt)
        uxx = grad(g[:, 0:1], xt)[:, 0:1]
        res = g[:, 1:2] + u * g[:, 0:1] - self.nu * uxx
        term = res.square().mean()
        return term, {"pde": term}

    @torch.no_grad()
    def score(self, model, device, dtype):
        xx, tt = np.meshgrid(self.x, self.t, indexing="xy")
        xt = torch.as_tensor(np.stack((xx.reshape(-1), tt.reshape(-1)), 1), dtype=dtype, device=device)
        pred = self.transform(xt, model(xt)).double().cpu().numpy().reshape(self.u.shape)
        err = pred - self.u
        return {"l2re_coleHopf": float(np.linalg.norm(err) / np.linalg.norm(self.u)), "max_abs": float(np.abs(err).max())}


class WrappedForAC(nn.Module):
    """Score/loss operate on the hard-constrained field; the raw net is what checkpoints store."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, xt):
        return self.net(xt)


def make_problem(name, ref):
    return {"ns": NSProblem, "ac": ACProblem, "burgers": BurgersProblem}[name](ref)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ------------------------------------------------------------------ bench
def bench(args):
    device = torch.device(args.device)
    rows = []
    configs = [("ns", "mlp", 0, 0), ("ns", "rbf", 16, 11), ("ns", "rbf", 32, 11), ("ns", "rbf", 48, 11),
               ("ns", "rbf", 64, 11), ("ns", "rbf", 32, 21), ("ac", "mlp", 0, 0), ("ac", "rbf", 16, 11)]
    dtypes = [torch.float32, torch.float64] if args.both_dtypes else [torch.float32]
    for pname, arm, width, bases in configs:
        prob = make_problem(pname, args.ns_ref if pname == "ns" else args.ac_ref)
        for dt in dtypes:
            if pname == "ns" and dt == torch.float64 and arm == "rbf" and width > 32:
                continue
            model = prob.build(arm, width, bases, 0)
            if pname == "ac":
                base_net = model
                model = WrappedForAC(base_net)
            model = model.to(device=device, dtype=dt)
            P = prob.pools(0, device, dt)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            def one():
                opt.zero_grad(set_to_none=True)
                l, _ = prob.loss(model, P)
                l.backward()
                opt.step()

            for _ in range(5):
                one()
            sync(device)
            t0 = time.perf_counter()
            n = 0
            while time.perf_counter() - t0 < args.seconds:
                one()
                n += 1
            sync(device)
            dtm = time.perf_counter() - t0
            rows.append({"problem": pname, "arm": arm, "width": width, "bases": bases, "dtype": str(dt),
                         "params": sum(p.numel() for p in model.parameters()), "steps_timed": n,
                         "seconds_per_step": dtm / n, "steps_per_second": n / dtm,
                         "peak_mem_mb": (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None})
            print(json.dumps(rows[-1]), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "bench.json").write_text(json.dumps(rows, indent=2))


# ------------------------------------------------------------------ worker
def run_worker(args):
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    prob = make_problem(args.problem, args.ref)
    if args.n_col and hasattr(prob, "n_col"):
        prob.n_col = args.n_col
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    net = prob.build(args.arm, args.width, args.bases, args.seed)
    model = net
    model = model.to(device=device, dtype=dtype)
    P = prob.pools(args.seed, device, dtype)
    steps = args.adam_steps or prob.adam_steps_default
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist, t0 = [], time.perf_counter()
    status = "adam_complete"
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        total, terms = prob.loss(model, P)
        total.backward()
        opt.step()
        if step % 100 == 0:
            val = float(total.detach())
            if not math.isfinite(val):
                status = "nonfinite_loss"
                break
            hist.append({"step": step, "loss": val, "seconds": time.perf_counter() - t0})
        if args.adam_seconds_cap and time.perf_counter() - t0 > args.adam_seconds_cap:
            status = "adam_time_cap"
            break
    sync(device)
    adam_seconds = time.perf_counter() - t0
    adam_steps_done = step
    torch.save({"state_dict": model.state_dict(), "stage": "adam_final", "step": adam_steps_done},
               out / "checkpoint-adam-final.pt")
    result = {"status": STATUS, "problem": args.problem, "arm": args.arm, "seed": args.seed, "width": args.width,
              "bases": args.bases, "dtype": args.dtype, "params": params, "adam_steps": adam_steps_done,
              "adam_target": steps, "adam_status": status, "adam_seconds": adam_seconds,
              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "ref_sha256": sha(args.ref), "scientific_accepted": False, "reference_accepted": False}
    result["score_adam_final"] = prob.score(model, device, dtype)
    result["final_loss_terms_adam"] = {k: float(v.detach()) for k, v in prob.loss(model, P)[1].items()}
    (out / "history.json").write_text(json.dumps(hist))
    print(json.dumps({"seed": args.seed, "arm": args.arm, "adam": result["score_adam_final"], "s": adam_seconds}),
          flush=True)

    if args.lbfgs_dtype == "float64" and dtype != torch.float64:
        model.double()
        P = {k: v.double() for k, v in P.items()}
        dtype = torch.float64
        result["lbfgs_dtype"] = "float64"
    if args.lbfgs_seconds > 0 and status == "adam_complete":
        opt2 = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=args.lbfgs_chunk, max_eval=args.lbfgs_chunk * 5,
                                 history_size=100, tolerance_grad=1e-8, tolerance_change=0.0,
                                 line_search_fn="strong_wolfe")
        accepted = {k: v.detach().clone() for k, v in model.state_dict().items()}
        last = float(prob.loss(model, P)[0].detach())
        t1, iters, lstatus = time.perf_counter(), 0, "lbfgs_time_cap"

        def closure():
            opt2.zero_grad(set_to_none=True)
            l, _ = prob.loss(model, P)
            l.backward()
            return l

        plateau = 0
        while True:
            if time.perf_counter() - t1 >= args.lbfgs_seconds:
                lstatus = "lbfgs_time_cap"
                break
            if iters >= args.lbfgs_max_iter:
                lstatus = "lbfgs_iter_cap"
                break
            before = opt2.state[opt2._params[0]].get("n_iter", 0)
            try:
                opt2.step(closure)
            except Exception as exc:  # noqa: BLE001 - recorded, state restored below
                lstatus = f"lbfgs_exception:{type(exc).__name__}"
                break
            after = opt2.state[opt2._params[0]].get("n_iter", 0)
            cur = float(prob.loss(model, P)[0].detach())
            if not math.isfinite(cur) or cur > last * 1.5:
                model.load_state_dict(accepted)
                lstatus = "lbfgs_rejected_nonfinite_or_worse"
                break
            improved = (last - cur) / max(abs(last), 1e-30)
            accepted = {k: v.detach().clone() for k, v in model.state_dict().items()}
            last = cur
            iters += after - before
            if after == before:
                lstatus = "optimizer_stopped_early"
                break
            plateau = plateau + 1 if improved < 1e-9 else 0
            if plateau >= 3:
                lstatus = "lbfgs_plateau"
                break
        model.load_state_dict(accepted)
        sync(device)
        result.update({"lbfgs_status": lstatus, "lbfgs_iterations": iters,
                       "lbfgs_seconds": time.perf_counter() - t1})
        torch.save({"state_dict": model.state_dict(), "stage": "lbfgs_final"}, out / "checkpoint-lbfgs-final.pt")
        result["score_lbfgs_final"] = prob.score(model, device, dtype)
        print(json.dumps({"seed": args.seed, "arm": args.arm, "lbfgs": result["score_lbfgs_final"],
                          "iters": iters, "status": lstatus}), flush=True)
    result["checkpoint_sha256"] = {p.name: sha(p) for p in out.glob("checkpoint-*.pt")}
    (out / "run.json").write_text(json.dumps(result, indent=2))


# ------------------------------------------------------------------ campaign driver
def run_campaign(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1
    queue = []
    specs = []
    for tok in args.arms:  # token: arm[@width][#n_seeds]
        limit = None
        if "#" in tok:
            tok, n = tok.split("#")
            limit = int(n)
        arm, _, w = tok.partition("@")
        specs.append((arm, int(w) if w else args.width, limit))
    for si, seed in enumerate(args.seeds):
        for arm, width, limit in specs:
            if limit is None or si < limit:
                queue.append((seed, arm, width))
    # long jobs of a seed are already listed first by the caller's arm order
    ledger, running, started = [], {}, time.monotonic()
    while queue or running:
        for gpu in list(running):
            proc, job, t_job = running[gpu]
            if proc.poll() is not None:
                ledger.append({"seed": job[0], "arm": job[1], "width": job[2], "return_code": proc.returncode,
                               "wall_seconds": time.monotonic() - t_job, "gpu": gpu})
                del running[gpu]
        if time.monotonic() - started > args.wall_cap and queue:
            ledger.append({"note": "wall cap reached; remaining jobs not started", "remaining": queue})
            queue = []
        for gpu in range(min(n_gpu, args.max_parallel)):
            if gpu not in running and queue:
                seed, arm, width = queue.pop(0)
                tag = f"{arm}{width}" if arm == "rbf" else arm
                cmd = [sys.executable, __file__, "worker", "--problem", args.problem, "--arm", arm, "--seed", str(seed),
                       "--width", str(width), "--bases", str(args.bases), "--device", f"cuda:{gpu}",
                       "--dtype", args.dtype, "--out", str(out / f"{tag}-seed-{seed}"), "--ref", args.ref,
                       "--adam-steps", str(args.adam_steps), "--adam-seconds-cap", str(args.adam_seconds_cap),
                       "--lbfgs-seconds", str(args.lbfgs_seconds), "--lbfgs-max-iter", str(args.lbfgs_max_iter),
                       "--lbfgs-dtype", args.lbfgs_dtype, "--n-col", str(args.n_col)]
                log = open(out / f"{tag}-seed-{seed}.log", "w")
                running[gpu] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), (seed, arm, width), time.monotonic())
        (out / "ledger.json").write_text(json.dumps({"status": STATUS, "elapsed": time.monotonic() - started,
                                                     "runs": ledger, "running": [j for _, j, _ in running.values()]}, indent=2))
        time.sleep(2.0)
    (out / "ledger.json").write_text(json.dumps({"status": STATUS, "elapsed": time.monotonic() - started,
                                                 "runs": ledger, "complete": True}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("bench", "campaign", "worker"))
    ap.add_argument("--problem", choices=("ns", "ac", "burgers"), default="ns")
    ap.add_argument("--arm", choices=("mlp", "rbf"), default="rbf")
    ap.add_argument("--arms", nargs="+", default=["mlp", "rbf"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[701, 702, 703, 704, 705])
    ap.add_argument("--seed", type=int, default=701)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--bases", type=int, default=11)
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--adam-steps", type=int, default=0)
    ap.add_argument("--adam-seconds-cap", type=float, default=0.0)
    ap.add_argument("--lbfgs-seconds", type=float, default=0.0)
    ap.add_argument("--lbfgs-dtype", default="same", choices=("same", "float64"))
    ap.add_argument("--n-col", type=int, default=0)
    ap.add_argument("--lbfgs-chunk", type=int, default=50)
    ap.add_argument("--lbfgs-max-iter", type=int, default=15000)
    ap.add_argument("--wall-cap", type=float, default=1e9)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--both-dtypes", action="store_true")
    ap.add_argument("--ref", default="")
    ap.add_argument("--ns-ref", default="")
    ap.add_argument("--ac-ref", default="")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    {"bench": bench, "campaign": run_campaign, "worker": run_worker}[a.mode](a)


if __name__ == "__main__":
    main()
