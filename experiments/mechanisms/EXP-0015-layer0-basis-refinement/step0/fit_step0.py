"""EXP-0015 paso 0 — ajuste supervisado de la referencia (sin PDE).

Pregunta: ¿la arquitectura de control PUEDE representar la solución con el error
que reporta el baseline PINN? Brazos de la capa 0: C (11 uniformes), U41 (41 uniformes),
O11 (11 por equidistribución de un monitor de la referencia), X11 (espaciado de O11 desplazado).

Diagnóstico: usa la referencia como objetivo de regresión. Ningún mecanismo lo usa.
"""
from __future__ import annotations

import argparse
import os
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from scipy.interpolate import griddata
from torch import nn
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)
ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[4]))  # raíz del repo
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
EX = ROOT / "experiments"


# ----------------------------------------------------------------------------- referencias
def load_problem(name: str) -> dict:
    """Devuelve puntos físicos, campos de referencia, normalización a [-1,1] y config de control."""
    if name == "heat":
        xs, ts = np.linspace(0, 1, 257), np.linspace(0, 1, 129)
        X, T = np.meshgrid(xs, ts, indexing="ij")
        pts = np.stack([X.ravel(), T.ravel()], 1)
        u = np.sin(np.pi * pts[:, 0]) * np.exp(-0.1 * np.pi**2 * pts[:, 1])
        return dict(pts=pts, fields=u[:, None], names=["u"], lo=[0, 0], hi=[1, 1], width=16,
                    centers=(-1.05, 1.05), sigma=0.21, constraint="heat", baseline_l2=1.857e-6,
                    baseline_note="RBF-PIKAN tras L-BFGS (v2)")
    if name == "burgers":
        z = np.load(BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz")
        X, T = np.meshgrid(z["x"], z["t"], indexing="ij")
        pts = np.stack([X.ravel(), T.ravel()], 1)
        u = z["u"].T.ravel()
        return dict(pts=pts, fields=u[:, None], names=["u"], lo=[-1, 0], hi=[1, 1], width=48,
                    centers=(-1.0, 1.0), sigma=0.35, constraint="burgers", baseline_l2=7.16e-5,
                    baseline_note="RBF48 tras L-BFGS f64, corrida grande (v2)")
    if name == "allen_cahn":
        m = sio.loadmat(BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat")
        X, T = np.meshgrid(m["x"].ravel(), m["t"].ravel(), indexing="ij")
        pts = np.stack([X.ravel(), T.ravel()], 1)
        u = m["u"].T.ravel()
        return dict(pts=pts, fields=u[:, None], names=["u"], lo=[-1, 0], hi=[1, 1], width=16,
                    centers=(-1.0, 1.0), sigma=0.35, constraint="allen_cahn", baseline_l2=2.686e-3,
                    baseline_note="RBF16 tras L-BFGS (v2)")
    if name == "ns2d_cg":
        r = np.loadtxt(BL / "ns2d-cg-20260930/reference/ns_0_obstacle.dat", comments="%", encoding="utf-8")
        return dict(pts=r[:, :2], fields=r[:, 2:5], names=["u", "v", "p"], lo=[0, 0], hi=[4, 2], width=56,
                    centers=(-1.0, 1.0), sigma=0.35, constraint=None, baseline_l2=0.0655,
                    baseline_note="RBF56 fin de Adam, L2RE plano (v2; sin L-BFGS)")
    if name == "dfg":
        z = np.load(BL / "ns2d-cg-20260930/reference/dfg_L3_fields.npz")
        f = np.column_stack([z["velocity"], z["pressure_gauge_fixed"]])
        return dict(pts=z["xy"], fields=f, names=["u", "v", "p"], lo=[0, 0], hi=[2.2, 0.41], width=16,
                    centers=(-1.0, 1.0), sigma=0.35, constraint=None, baseline_l2=0.070,
                    baseline_note="RBF cohorte B tras L-BFGS: u 0.070, v 0.069, p 0.246")
    if name == "piezo":
        z = np.load(EX / "piezo/EXP-0009-piezo-indirect-rbf/results/reference/fem-grid-indirect-polingminus.npz")
        f = np.column_stack([z["u"], z["v"], z["phi"]])
        return dict(pts=z["xy"], fields=f, names=["u", "v", "phi"], lo=[0, 0], hi=[0.1, 0.001], width=16,
                    centers=(-1.0, 1.0), sigma=0.35, constraint=None, baseline_l2=0.092,
                    baseline_note="RBF16 receta de Daniel, 20k Adam: u 0.092, v 0.104, phi 0.048")
    raise KeyError(name)


# ----------------------------------------------------------------------------- centros
def monitor_marginal(P: dict, dim: int, nbins: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """m(s) en la coordenada normalizada s∈[-1,1] de la dimensión `dim`: media de |∂f/∂s|
    sobre las demás dimensiones, sumando componentes normalizadas por su desviación."""
    s = 2 * (P["pts"] - np.array(P["lo"])) / (np.array(P["hi"]) - np.array(P["lo"])) - 1
    n = 257
    g = np.linspace(-1, 1, n)
    G0, G1 = np.meshgrid(g, g, indexing="ij")
    total = np.zeros((n, n))
    for c in range(P["fields"].shape[1]):
        f = P["fields"][:, c] / (P["fields"][:, c].std() + 1e-300)
        F = griddata(s, f, (G0, G1), method="linear")
        F = np.where(np.isnan(F), griddata(s, f, (G0, G1), method="nearest"), F)
        dF = np.gradient(F, g, axis=dim)
        total += np.abs(dF)
    m = total.mean(axis=1 - dim)
    return g, m


def equidistribute(g: np.ndarray, m: np.ndarray, G: int, floor: float = 0.3) -> np.ndarray:
    M = floor + (1 - floor) * m / m.mean()
    cdf = np.concatenate([[0], np.cumsum(0.5 * (M[1:] + M[:-1]) * np.diff(g))])
    cdf /= cdf[-1]
    c = np.interp(np.linspace(0, 1, G), cdf, g)
    c[0], c[-1] = -1.0, 1.0
    return c


def shifted(c: np.ndarray, shift: float = 0.5) -> np.ndarray:
    """Mismo multiconjunto de espaciados, zona densa desplazada `shift` (circular en [-1,1])."""
    gaps = np.diff(c)
    k = int(round(shift / 2 * len(gaps)))
    gaps = np.roll(gaps, k)
    return np.concatenate([[-1.0], -1.0 + np.cumsum(gaps)])


def local_sigma(c: np.ndarray, factor: float) -> np.ndarray:
    d = np.empty_like(c)
    d[1:-1] = (c[2:] - c[:-2]) / 2
    d[0], d[-1] = c[1] - c[0], c[-1] - c[-2]
    return factor * d


# ----------------------------------------------------------------------------- modelo
class RBFLayer(nn.Module):
    """centers/width de forma [in, G]; con centros uniformes y ancho constante = control."""

    def __init__(self, i: int, o: int, centers: np.ndarray, width: np.ndarray):
        super().__init__()
        self.register_buffer("centers", torch.as_tensor(centers))
        self.register_buffer("width", torch.as_tensor(width))
        G = self.centers.shape[-1]
        self.coefficients = nn.Parameter(torch.empty(o, i, G))
        self.base = nn.Parameter(torch.empty(o, i))
        nn.init.normal_(self.coefficients, std=0.1 / i**0.5)
        nn.init.normal_(self.base, std=0.1 / i**0.5)

    def forward(self, x):
        b = torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width) ** 2)
        return torch.einsum("big,oig->bo", b, self.coefficients) + F.linear(F.silu(x), self.base)


class Net(nn.Module):
    def __init__(self, P: dict, layer0_centers: list[np.ndarray], layer0_widths: list[np.ndarray]):
        super().__init__()
        w, k = P["width"], P["fields"].shape[1]
        G = 11
        uc = np.linspace(*P["centers"], G)
        us = np.full(G, P["sigma"])
        self.l0 = RBFLayer(2, w, np.stack(layer0_centers), np.stack(layer0_widths))
        self.l1 = RBFLayer(w, w, np.tile(uc, (w, 1)), np.tile(us, (w, 1)))
        self.l2 = RBFLayer(w, k, np.tile(uc, (w, 1)), np.tile(us, (w, 1)))
        self.P = P
        self.register_buffer("lo", torch.tensor(P["lo"], dtype=torch.float64))
        self.register_buffer("hi", torch.tensor(P["hi"], dtype=torch.float64))

    def forward(self, xphys, scale):
        s = 2 * (xphys - self.lo) / (self.hi - self.lo) - 1
        h = torch.tanh(self.l0(s))
        h = torch.tanh(self.l1(h))
        raw = self.l2(h) * scale
        c = self.P["constraint"]
        x, t = xphys[:, 0:1], xphys[:, 1:2]
        if c == "heat":
            return torch.sin(math.pi * x) + t * x * (1 - x) * raw
        if c == "allen_cahn":
            return x**2 * torch.cos(math.pi * x) + t * (1 - x**2) * raw
        if c == "burgers":
            return -torch.sin(math.pi * x) + t * (1 - x**2) * raw
        return raw


def arm_layer0(P: dict, arm: str):
    if arm == "U41":
        c = np.linspace(*P["centers"], 41)
        sig = P["sigma"] * (c[1] - c[0]) / ((P["centers"][1] - P["centers"][0]) / 10)
        return [c, c], [np.full(41, sig)] * 2
    uc = np.linspace(*P["centers"], 11)
    if arm == "C":
        return [uc, uc], [np.full(11, P["sigma"])] * 2
    factor = P["sigma"] / (uc[1] - uc[0])
    cs, ws = [], []
    for d in range(2):
        g, m = monitor_marginal(P, d)
        c = equidistribute(g, m, 11)
        if arm == "X11" and d == 0:
            c = shifted(c)
        cs.append(c)
        ws.append(local_sigma(c, factor))
    return cs, ws


# ----------------------------------------------------------------------------- ajuste
def rel_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def fit(P, arm, seed, adam_steps, lbfgs_iters, n_train, log_every=500):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cs, ws = arm_layer0(P, arm)
    net = Net(P, cs, ws)
    X = torch.as_tensor(P["pts"])
    Y = torch.as_tensor(P["fields"])
    scale = Y.std(0, keepdim=True)
    if P["constraint"]:
        scale = torch.ones_like(scale)
    # muestreo de entrenamiento: mitad uniforme, mitad proporcional a |grad f| de la referencia.
    # Es el mismo conjunto para todos los brazos de una semilla (no favorece a ninguno).
    from scipy.spatial import cKDTree
    pts = P["pts"]
    sc = (pts - pts.min(0)) / (pts.max(0) - pts.min(0))
    _, nb = cKDTree(sc).query(sc, k=7)
    g = np.zeros(len(pts))
    for c in range(P["fields"].shape[1]):
        f = P["fields"][:, c] / (P["fields"][:, c].std() + 1e-300)
        d = np.linalg.norm(sc[nb[:, 1:]] - sc[:, None], axis=-1)
        g += (np.abs(f[nb[:, 1:]] - f[:, None]) / np.maximum(d, 1e-12)).max(1)
    prob = 0.5 / len(pts) + 0.5 * g / g.sum()
    idx = np.random.default_rng(1000 + seed).choice(len(pts), size=min(n_train, len(pts)), replace=False, p=prob)
    Xt, Yt = X[idx], Y[idx]
    w = 1.0 / (Yt.std(0, keepdim=True) ** 2)  # pérdida por componente normalizada
    loss_fn = lambda: (((net(Xt, scale) - Yt) ** 2) * w).mean()
    curve = []
    t0 = time.time()

    def evaluate():
        with torch.no_grad():
            pred = torch.cat([net(X[i:i + 20000], scale) for i in range(0, len(X), 20000)]).numpy()
        comps = {n: rel_l2(pred[:, j], P["fields"][:, j]) for j, n in enumerate(P["names"])}
        flat = rel_l2(pred, P["fields"])
        return comps, flat

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(1, adam_steps + 1):
        opt.zero_grad()
        L = loss_fn()
        L.backward()
        opt.step()
        if step % log_every == 0 or step == adam_steps:
            comps, flat = evaluate()
            curve.append(dict(stage="adam", step=step, t=time.time() - t0, loss=float(L.detach()), **{f"l2_{k}": v for k, v in comps.items()}))
    adam_comps, _ = evaluate()
    lb = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=25, history_size=100, line_search_fn="strong_wolfe",
                           tolerance_grad=0.0, tolerance_change=0.0)

    def closure():
        lb.zero_grad()
        L = loss_fn()
        L.backward()
        return L

    for blk in range(lbfgs_iters // 25):
        L = lb.step(closure)
        comps, flat = evaluate()
        curve.append(dict(stage="lbfgs", step=adam_steps + 25 * (blk + 1), t=time.time() - t0, loss=float(L.detach()), **{f"l2_{k}": v for k, v in comps.items()}))
        if not math.isfinite(float(L.detach())):
            break
    comps, flat = evaluate()
    return dict(arm=arm, seed=seed, adam=adam_comps, final=comps, seconds=time.time() - t0,
                params=sum(p.numel() for p in net.parameters()),
                layer0_centers=[c.tolist() for c in cs], curve=curve)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--arms", default="C,U41,O11")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--adam", type=int, default=3000)
    ap.add_argument("--lbfgs", type=int, default=500)
    ap.add_argument("--ntrain", type=int, default=12000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = load_problem(a.problem)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for arm in a.arms.split(","):
        for s in map(int, a.seeds.split(",")):
            f = out / f"{a.problem}-{arm}-seed{s}.json"
            if f.exists():
                continue
            r = fit(P, arm, s, a.adam, a.lbfgs, a.ntrain)
            r.update(problem=a.problem, baseline_l2=P["baseline_l2"], baseline_note=P["baseline_note"],
                     adam_steps=a.adam, lbfgs_iters=a.lbfgs, ntrain=a.ntrain)
            f.write_text(json.dumps(r))
            print(a.problem, arm, s, {k: f"{v:.2e}" for k, v in r["final"].items()}, f"{r['seconds']:.0f}s", flush=True)
