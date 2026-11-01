#!/usr/bin/env python3
"""DRAFT_UNVERIFIED: run the *paper recipe* of Gonzalez Carrillo's indirect PINN (PINNs_piezoelectricity @ e7856f1, 2026-10-02) with either its own
MLP (3x50 SiLU, phase-enriched shared trunk) or an RBF-PIKAN trunk swapped in for `model.net`, everything else unchanged
(layered sampling with 2048 interior points per layer, interface/boundary points, resampling every 200 epochs, per-field scales,
mechanical constitutive stop-gradient routing, hard voltage lifting, 20,000 Adam epochs at lr 1e-3, float64, no L-BFGS).
Scoring is done here against the FEM grid (poling_sign=-1) so MLP and RBF share one evaluator.

usage: py -3 daniel_arm.py --daniel-root <repo clone> --arm mlp|rbf --seed 20260728 --epochs 20000 --out DIR --fem fem.npz [--width 16] [--device cuda:0]
The upstream repo is read/imported, never modified. Its licence file is missing (pyproject declares MIT): this uses it as a documented external reference.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--daniel-root", required=True)
ap.add_argument("--arm", choices=("mlp", "rbf"), default="mlp")
ap.add_argument("--width", type=int, default=16)
ap.add_argument("--seed", type=int, default=20260728)
ap.add_argument("--epochs", type=int, default=20000)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", required=True)
ap.add_argument("--fem", required=True)
ap.add_argument("--interior", type=int, default=2048)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--snapshot-every", type=int, default=2000)
args = ap.parse_args()

root = Path(args.daniel_root).resolve()
sys.path[:0] = [str(root / "src"), str(root)]
stub = types.ModuleType("torchsummary")
stub.summary = lambda *a, **k: None
sys.modules.setdefault("torchsummary", stub)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import pinn_piezo.config as pcfg  # noqa: E402
from pinn_piezo.indirect import model as model_mod  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
pcfg.RUNS_DIR = out / "runs"
pcfg.RUNS_DIR.mkdir(parents=True, exist_ok=True)

z = np.load(args.fem)
FEM = {k: z[k] for k in ("xy", "u", "v", "phi")}
CAPTURE = {}


class RBFNet(nn.Module):
    """Drop-in for `model.net` of the phase-enriched trunk: inputs (xi, eta, |eta-.5|, phase) -> 8 raw outputs."""

    def __init__(self, width):
        super().__init__()
        from math import sqrt  # noqa: F401

        class Edge(nn.Module):
            def __init__(self, i, o, bases=11):
                super().__init__()
                sp = 2.0 / (bases - 1)
                self.register_buffer("centers", torch.linspace(-1.0, 1.0, bases))
                self.register_buffer("width", torch.tensor(1.75 * sp))
                self.coefficients = nn.Parameter(torch.randn(o, i, bases) * (0.1 / math.sqrt(i)))
                self.base = nn.Parameter(torch.randn(o, i) * (0.1 / math.sqrt(i)))

            def forward(self, x):
                b = torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width).square())
                return torch.einsum("big,oig->bo", b, self.coefficients) + torch.nn.functional.linear(torch.nn.functional.silu(x), self.base)

        self.layers = nn.ModuleList([Edge(4, width), Edge(width, width), Edge(width, 8)])

    def forward(self, z):
        h = torch.cat([2 * z[:, 0:1] - 1, 2 * z[:, 1:2] - 1, 4 * z[:, 2:3] - 1, z[:, 3:4]], dim=1)
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)


_orig_build = model_mod.build_default_model


def build_patched(*a, **k):
    m = _orig_build(*a, **k)
    if args.arm == "rbf":
        dev = next(m.parameters()).device
        m.net = RBFNet(args.width).to(device=dev, dtype=next(m.parameters()).dtype)
    CAPTURE["model"] = m
    return m


model_mod.build_default_model = build_patched


def score(m, dev):
    xy = torch.tensor(FEM["xy"], dtype=torch.float64, device=dev)
    with torch.no_grad():
        o = m(xy).double().cpu().numpy()
        tip = float(m(torch.tensor([[0.1, 0.0]], dtype=torch.float64, device=dev))[0, 1])

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    return {"u_relL2": rel(o[:, 0], FEM["u"]), "v_relL2": rel(o[:, 1], FEM["v"]), "phi_relL2": rel(o[:, 2], FEM["phi"]),
            "tip_v": tip, "tip_v_fem": float(FEM["v"][np.argmin(np.abs(FEM["xy"][:, 0] - 0.1) + FEM["xy"][:, 1])])}


# --- snapshot hook: wrap the upstream train() so we can score periodically ---------------------------------------------------------
import pinn_piezo.indirect.train as train_mod  # noqa: E402

SNAPS = []
T0 = time.perf_counter()
_orig_run_adam = train_mod.run_adam


def run_adam_patched(model, tensors, *a, **k):
    dev = next(model.parameters()).device
    ep_total = k.get("epochs", args.epochs)
    chunk = args.snapshot_every
    done = 0
    result = None
    while done < ep_total:
        n = min(chunk, ep_total - done)
        kk = dict(k)
        kk["epochs"] = n
        res = _orig_run_adam(model, tensors, *a, **kk) if result is None else _orig_run_adam(model, tensors, *a, **{**kk, "loss_weights": result[1]})
        result = res if result is None else (result[0] + res[0], res[1], min(result[2], res[2]))
        done += n
        sc = score(model, dev)
        SNAPS.append({"step": done, "u": sc["u_relL2"], "v": sc["v_relL2"], "phi": sc["phi_relL2"], "tip_v": sc["tip_v"], "t": time.perf_counter() - T0})
        print(json.dumps(SNAPS[-1]), flush=True)
    return result


train_mod.run_adam = run_adam_patched

import scripts.train_indirect as ti  # noqa: E402

argv = ["train_indirect", "--epochs-adam", str(args.epochs), "--epochs-lbfgs", "0", "--lr-adam", str(args.lr), "--model-type", "pyramid",
        "--hidden-sizes", "50", "50", "50", "--activation", "silu", "--phase-enriched", "--no-split-trunks", "--no-hard-natural-bcs",
        "--interior-per-layer", str(args.interior), "--boundary-points", "256", "--interface-points", "256", "--resample-every", "200",
        "--interface-weight", "1.0", "--no-slender-transverse-scaling", "--normal-constitutive-weight", "1.0",
        "--constitutive-normalization", "represented", "--no-constitutive-traction-bcs", "--no-constitutive-flux-stopgrad",
        "--mechanical-constitutive-stopgrad", "--stress-warmup-epochs", "0", "--seed", str(args.seed), "--voltage", "100", "--no-zero-output",
        "--no-periodic-checkpoints", "--dtype", "float64", "--device", ("cuda" if args.device.startswith("cuda") else args.device), "--run-name", f"daniel-{args.arm}-seed-{args.seed}"]
sys.argv = argv
ti.main()
m = CAPTURE["model"]
dev = next(m.parameters()).device
res = {"status": "DRAFT_UNVERIFIED", "arm": args.arm, "width": args.width if args.arm == "rbf" else None, "seed": args.seed, "epochs": args.epochs,
       "params": sum(p.numel() for p in m.parameters()), "seconds": time.perf_counter() - T0, "score_final": score(m, dev), "snapshots": SNAPS,
       "upstream_commit": "e7856f1 (2026-10-02)", "scientific_accepted": False, "reference_accepted": False}
torch.save({"state_dict": m.state_dict()}, out / f"checkpoint-{args.arm}-seed-{args.seed}.pt")
(out / "run.json").write_text(json.dumps(res, indent=2))
print(json.dumps({"final": res["score_final"], "params": res["params"], "s": res["seconds"]}), flush=True)
