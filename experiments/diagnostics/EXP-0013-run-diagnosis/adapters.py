"""EXP-0013 — adaptadores: cargan un checkpoint y lo exponen con una interfaz común (campos, residuos por punto, capas RBF).

NO importa `evals` ni lee referencias. Cada Bundle expone solo lo que el modelo y la PDE permiten calcular.
Los residuos se calculan con autograd en float64 (el modelo se convierte a float64; los checkpoints float32 se promueven sin cambio de valores).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from common import BL, EX, REPO, load_module

torch.set_default_dtype(torch.float64)

HEAT_ALPHA, HEAT_SCALE = 0.1, 1.1
BURGERS_NU, BURGERS_SCALE = 0.01 / math.pi, 2.0031830988618378
AC_D, AC_R, AC_SCALE = 0.001, 5.0, 6.001
NS_NU = 0.01
DFG_NU = 0.001


@dataclass
class Bundle:
    problem: str
    family: str
    kind: str                       # 'rbf' | 'mlp'
    params: int
    dtype_native: str
    fields: Callable                # coords (N,d) -> (N,k)
    residuals: Callable             # coords (N,d) -> dict name -> (N,) numpy, escalados
    layers: list | None = None      # capas RBF (centers, width, coefficients)
    h0: Callable | None = None      # coords -> entrada normalizada de la capa 0
    phys_per_unit: list | None = None   # longitud física por unidad normalizada, por dimensión de h0
    input_names: list = field(default_factory=list)
    primary: int = 0                # índice del campo usado para escalas de estructura
    model: object = None
    notes: list = field(default_factory=list)
    extra_fn: Callable | None = None   # señales específicas del problema (sin referencia)
    res_tensors: Callable | None = None  # coords -> dict name -> (N,) tensor diferenciable (para gradientes por término)


def _grad(y, x, create=True):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=create, retain_graph=True)[0]


def state_of(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if isinstance(sd, dict) and sd and all(k.startswith("net.") for k in sd):
        sd = {k[4:]: v for k, v in sd.items()}
    return sd


def native_dtype(sd) -> str:
    for v in sd.values():
        if torch.is_tensor(v) and v.is_floating_point():
            return str(v.dtype).replace("torch.", "")
    return "unknown"


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _wrap(res_tensors):
    return lambda pts: {k: v.detach().numpy() for k, v in res_tensors(pts).items()}


def _to_t(points, requires_grad=True):
    return torch.as_tensor(np.asarray(points), dtype=torch.float64).clone().requires_grad_(requires_grad)


# ----------------------------------------------------------------------------- pikan_campaign family (Burgers, AC, NS2d-CG, DFG-C)
def _pc():
    return load_module("pikan_campaign", BL / "burgers-close-20261003/pikan_campaign.py")


def build_pikan_net(sd, lo, hi):
    pc = _pc()
    if "layers.0.coefficients" in sd:
        w, n_in, bases = sd["layers.0.coefficients"].shape
        n_out = sd["layers.2.coefficients"].shape[0]
        net = pc.RBFKAN(n_in, w, n_out, bases, lo, hi)
        kind = "rbf"
    else:
        ws = [sd[f"layers.{i}.weight"].shape for i in range(100) if f"layers.{i}.weight" in sd]
        net = pc.MLP([ws[0][1]] + [s[0] for s in ws])
        kind = "mlp"
    net.load_state_dict(sd, strict=True)
    return net.double().eval(), kind


def _kan_h0(lo, hi):
    lo_t, hi_t = torch.tensor(lo), torch.tensor(hi)
    return lambda xy: 2.0 * (xy - lo_t) / (hi_t - lo_t) - 1.0


def bundle_burgers(path: Path) -> Bundle:
    sd = state_of(path)
    lo, hi = [-1.0, 0.0], [1.0, 1.0]
    net, kind = build_pikan_net(sd, lo, hi)

    def fields(xt):
        x, t = xt[:, 0:1], xt[:, 1:2]
        return -torch.sin(math.pi * x) + (1.0 - x.square()) * t * net(xt)

    def res_tensors(pts):
        xt = _to_t(pts)
        u = fields(xt)
        g = _grad(u, xt)
        uxx = _grad(g[:, 0:1], xt)[:, 0:1]
        r = (g[:, 1:2] + u * g[:, 0:1] - BURGERS_NU * uxx) / BURGERS_SCALE
        return {"pde": r.squeeze(-1)}

    b = Bundle("burgers_1d", "pikan_campaign", kind, n_params(net), native_dtype(sd), fields, _wrap(res_tensors), model=net,
               input_names=["x", "t"], phys_per_unit=[1.0, 0.5], primary=0, extra_fn=burgers_extras, res_tensors=res_tensors)
    if kind == "rbf":
        b.layers, b.h0 = list(net.layers), _kan_h0(lo, hi)
    return b


def bundle_allen_cahn(path: Path) -> Bundle:
    sd = state_of(path)
    lo, hi = [-1.0, 0.0], [1.0, 1.0]
    net, kind = build_pikan_net(sd, lo, hi)

    def fields(xt):
        x, t = xt[:, 0:1], xt[:, 1:2]
        return x.square() * torch.cos(math.pi * x) + (1.0 - x.square()) * t * net(xt)

    def res_tensors(pts):
        xt = _to_t(pts)
        u = fields(xt)
        g = _grad(u, xt)
        uxx = _grad(g[:, 0:1], xt)[:, 0:1]
        r = (g[:, 1:2] - AC_D * uxx - AC_R * (u - u.pow(3))) / AC_SCALE
        return {"pde": r.squeeze(-1)}

    b = Bundle("allen_cahn_1d", "pikan_campaign", kind, n_params(net), native_dtype(sd), fields, _wrap(res_tensors), model=net,
               input_names=["x", "t"], phys_per_unit=[1.0, 0.5], primary=0, extra_fn=ac_extras, res_tensors=res_tensors)
    if kind == "rbf":
        b.layers, b.h0 = list(net.layers), _kan_h0(lo, hi)
    return b


def ns_residuals_factory(fields, nu):
    def res_tensors(pts):
        xy = _to_t(pts)
        o = fields(xy)
        u, v, p = o[:, 0:1], o[:, 1:2], o[:, 2:3]
        gu, gv, gp = _grad(u, xy), _grad(v, xy), _grad(p, xy)
        uxx = _grad(gu[:, 0:1], xy)[:, 0:1]
        uyy = _grad(gu[:, 1:2], xy)[:, 1:2]
        vxx = _grad(gv[:, 0:1], xy)[:, 0:1]
        vyy = _grad(gv[:, 1:2], xy)[:, 1:2]
        mx = u * gu[:, 0:1] + v * gu[:, 1:2] + gp[:, 0:1] - nu * (uxx + uyy)
        my = u * gv[:, 0:1] + v * gv[:, 1:2] + gp[:, 1:2] - nu * (vxx + vyy)
        ct = gu[:, 0:1] + gv[:, 1:2]
        out = {"momentum_x": mx, "momentum_y": my, "continuity": ct}
        return {k: v_.squeeze(-1) for k, v_ in out.items()}

    return res_tensors


def bundle_ns2d(path: Path) -> Bundle:
    sd = state_of(path)
    lo, hi = [0.0, 0.0], [4.0, 2.0]
    net, kind = build_pikan_net(sd, lo, hi)
    fields = lambda xy: net(xy)
    b = Bundle("ns2d_cg", "pikan_campaign", kind, n_params(net), native_dtype(sd), fields, _wrap(_rt := ns_residuals_factory(fields, NS_NU)), model=net,
               input_names=["x", "y"], phys_per_unit=[2.0, 1.0], primary=0, res_tensors=_rt)
    if kind == "rbf":
        b.layers, b.h0 = list(net.layers), _kan_h0(lo, hi)
    return b


def bundle_dfg_c(path: Path) -> Bundle:
    sd = state_of(path)
    lo, hi = [0.0, 0.0], [2.2, 0.41]
    net, kind = build_pikan_net(sd, lo, hi)
    fields = lambda xy: net(xy)
    b = Bundle("dfg_2d", "dfg_daniel_recipe", kind, n_params(net), native_dtype(sd), fields, _wrap(_rt := ns_residuals_factory(fields, DFG_NU)), model=net,
               input_names=["x", "y"], phys_per_unit=[1.1, 0.205], primary=0, extra_fn=dfg_extras, res_tensors=_rt)
    if kind == "rbf":
        b.layers, b.h0 = list(net.layers), _kan_h0(lo, hi)
    return b


# ----------------------------------------------------------------------------- Heat (af_sciml_model / TutorialFNN)
def bundle_heat(path: Path, campaign: str) -> Bundle:
    sd = state_of(path)
    pay = BL / "kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/payload"
    if "layers.3.weight" in sd or any(k.startswith("layers.3") for k in sd):  # FNN 3x20
        mod = load_module("heat_comparison_runner", pay / "heat_comparison_runner.py", pay)
        net = mod.TutorialFNN()
        net.load_state_dict(sd, strict=True)
        net = net.double().eval()
        kind = "mlp"
        fields = lambda xt: net(xt)
        layers = h0 = None
    else:
        mod = load_module("af_sciml_model", BL / "kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2/payload/af_sciml_model.py")
        net = mod.build_model("heat", {"id": "rbf_fixed"})
        net.load_state_dict(sd, strict=True)
        net = net.double().eval()
        kind = "rbf"
        fields = lambda xt: net(xt)
        layers, h0 = list(net.layers), net.normalized

    def res_tensors(pts):
        xt = _to_t(pts)
        u = fields(xt)
        g = _grad(u, xt)
        uxx = _grad(g[:, 0:1], xt)[:, 0:1]
        r = (g[:, 1:2] - HEAT_ALPHA * uxx) / HEAT_SCALE
        return {"pde": r.squeeze(-1)}

    return Bundle("heat_1d", "af_sciml_model" if kind == "rbf" else "tutorial_fnn", kind, n_params(net), native_dtype(sd), fields, _wrap(res_tensors),
                  layers=layers, h0=h0, phys_per_unit=[0.5, 0.5], input_names=["x", "t"], primary=0, model=net, extra_fn=heat_extras, res_tensors=res_tensors)


# ----------------------------------------------------------------------------- DFG cohorts A and B (repo dfg.py / run_campaign_source.py)
def bundle_dfg_ab(path: Path, cohort: str) -> Bundle:
    sd = state_of(path)
    if cohort == "A":
        mod = load_module("repo_dfg", REPO / "src/research_pipeline/sciml/dfg.py", REPO)
        net = mod.DFGRBFPIKAN()
        net.load_state_dict(sd, strict=True)
        kind = "rbf"
        dom = ((0.0, 2.2), (0.0, 0.41))
        h0 = lambda xy: torch.cat((2.0 * (xy[:, 0:1] - dom[0][0]) / (dom[0][1] - dom[0][0]) - 1.0, 2.0 * (xy[:, 1:2] - dom[1][0]) / (dom[1][1] - dom[1][0]) - 1.0), dim=1)
    else:
        src = BL / "kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/run_campaign_source.py"
        mod = load_module("dfg_runner_source", src)
        arm = "DFG-Tanh-MLP" if "layers.0.weight" in sd else "DFG-RBF-PIKAN"
        net = mod.build_model(arm, 0)
        net.load_state_dict(sd, strict=True)
        kind = "mlp" if arm == "DFG-Tanh-MLP" else "rbf"
        h0 = net.normalize if hasattr(net, "normalize") else None
    net = net.double().eval()
    fields = lambda xy: net(xy)
    b = Bundle("dfg_2d", f"dfg_cohort_{cohort}", kind, n_params(net), native_dtype(sd), fields, _wrap(_rt := ns_residuals_factory(fields, DFG_NU)), model=net,
               input_names=["x", "y"], phys_per_unit=[1.1, 0.205], primary=0, extra_fn=dfg_extras, res_tensors=_rt)
    if kind == "rbf":
        b.layers, b.h0 = list(net.layers), h0
    return b


# ----------------------------------------------------------------------------- piezo (upstream read-only, solo para reconstruir el modelo)
class _RBFEdge(torch.nn.Module):
    def __init__(self, i, o, bases=11):
        super().__init__()
        sp = 2.0 / (bases - 1)
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, bases))
        self.register_buffer("width", torch.tensor(1.75 * sp))
        self.coefficients = torch.nn.Parameter(torch.randn(o, i, bases) * (0.1 / math.sqrt(i)))
        self.base = torch.nn.Parameter(torch.randn(o, i) * (0.1 / math.sqrt(i)))

    def forward(self, x):
        b = torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width).square())
        return torch.einsum("big,oig->bo", b, self.coefficients) + torch.nn.functional.linear(torch.nn.functional.silu(x), self.base)


class _RBFNet(torch.nn.Module):
    """Réplica de la clase del driver `daniel_arm.py`: 4 entradas (xi, eta, |eta-.5|, fase) -> 8 salidas crudas."""

    def __init__(self, width):
        super().__init__()
        self.layers = torch.nn.ModuleList([_RBFEdge(4, width), _RBFEdge(width, width), _RBFEdge(width, 8)])

    def forward(self, z):
        h = torch.cat([2 * z[:, 0:1] - 1, 2 * z[:, 1:2] - 1, 4 * z[:, 2:3] - 1, z[:, 3:4]], dim=1)
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)


_UP = REPO / "var/agent_runs/piezo-ref-20260930/daniel-ref-20261003"


def _upstream():
    import types
    if str(_UP / "src") not in sys.path:
        sys.path[:0] = [str(_UP / "src"), str(_UP)]
    stub = types.ModuleType("torchsummary")
    stub.summary = lambda *a, **k: None
    sys.modules.setdefault("torchsummary", stub)
    import pinn_piezo.config as cfg  # noqa: F401
    from pinn_piezo import geometry, materials, scaling
    from pinn_piezo.indirect import model as model_mod
    return cfg, geometry, materials, scaling, model_mod


def bundle_piezo(path: Path) -> Bundle:
    cfg, geometry, materials, scaling, model_mod = _upstream()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    is_rbf = "net.layers.0.coefficients" in sd
    m = model_mod.build_default_model(device=None, model_type="pyramid", hidden_sizes=(50, 50, 50), phase_enriched=True, hard_natural_bcs=False,
                                      activation=torch.nn.SiLU, zero_output=False, split_trunks=False)
    if is_rbf:
        m.net = _RBFNet(sd["net.layers.0.coefficients"].shape[0])
    m.load_state_dict(sd, strict=True)
    m = m.double().eval()
    scales = scaling.indirect_scales(100.0)
    const = scaling.represented_term_scales(scales, strict_transverse=True)

    def fields(xy):
        return m(xy)

    def res_tensors(pts):
        xy = _to_t(pts)
        o = m(xy)
        coeff = torch.as_tensor(geometry.build_coefficients(np.asarray(pts)), dtype=torch.float64)
        u, v, phi, sx, sz, tau, Dx, Dy = [o[:, i:i + 1] for i in range(8)]
        gu, gv, gp = _grad(u, xy), _grad(v, xy), _grad(phi, xy)
        exx, eyy, gam = gu[:, 0:1], gv[:, 1:2], gu[:, 1:2] + gv[:, 0:1]
        Ex, Ey = -gp[:, 0:1], -gp[:, 1:2]
        C11, C12, C22, G, e1, e2, e31, e33 = [coeff[:, i:i + 1] for i in range(8)]
        sx_c = C11 * exx + C12 * eyy - e31 * Ey
        sz_c = C12 * exx + C22 * eyy - e33 * Ey
        tau_c = G * gam
        Dx_c = e1 * Ex
        Dy_c = e31 * exx + e33 * eyy + e2 * Ey
        gsx, gsz, gt = _grad(sx, xy), _grad(sz, xy), _grad(tau, xy)
        gDx, gDy = _grad(Dx, xy), _grad(Dy, xy)
        out = {"constitutive_sigma_xx": (sx - sx_c) / const.sigma_xx, "constitutive_sigma_yy": (sz - sz_c) / const.sigma_yy,
               "constitutive_tau_xy": (tau - tau_c) / const.tau_xy, "constitutive_D_x": (Dx - Dx_c) / const.d_x, "constitutive_D_y": (Dy - Dy_c) / const.d_y,
               "equilibrium_x": (gsx[:, 0:1] + gt[:, 1:2]) / scales.equilibrium_x, "equilibrium_y": (gt[:, 0:1] + gsz[:, 1:2]) / scales.equilibrium_y,
               "gauss": (gDx[:, 0:1] + gDy[:, 1:2]) / scales.gauss}
        return {k: v_.squeeze(-1) for k, v_ in out.items()}

    b = Bundle("piezo_2d", "piezo_daniel_upstream", "rbf" if is_rbf else "mlp", n_params(m), native_dtype(sd), fields, _wrap(res_tensors), model=m,
               input_names=["x", "y"], phys_per_unit=[0.05, 0.0005], primary=1, extra_fn=piezo_extras, res_tensors=res_tensors)
    b.notes.append("modelo reconstruido con el codigo upstream de Daniel (solo lectura); residuos por punto con sus coeficientes y escalas")
    if is_rbf:
        b.layers = list(m.net.layers)
        W, H = 0.1, 1e-3
        b.h0 = lambda xy: torch.cat([2 * (xy[:, 0:1] / W) - 1, 2 * (xy[:, 1:2] / H) - 1, 4 * (xy[:, 1:2] / H - 0.5).abs() - 1,
                                     torch.where(xy[:, 1:2] >= 0.5 * H, torch.ones_like(xy[:, 1:2]), -torch.ones_like(xy[:, 1:2]))], dim=1)
        b.notes.append("capa 0 del RBF recibe 4 entradas (xi, eta, |eta-0.5|, fase); h0 las reconstruye; spacing fisico solo para x e y")
    return b


# ----------------------------------------------------------------------------- señales específicas (sin referencia)
def heat_extras(b: Bundle, pts: np.ndarray, shape) -> dict:
    """Energía y tasa de decaimiento del propio modelo (propiedad de la PDE, no usa la solución exacta)."""
    nx, nt = shape
    xt = _to_t(pts, requires_grad=False)
    with torch.no_grad():
        u = b.fields(xt).reshape(nx, nt)
    x = pts[:, 0].reshape(nx, nt)[:, 0]
    t = pts[:, 1].reshape(nx, nt)[0]
    E = np.trapezoid(u.numpy() ** 2, x, axis=0)
    ok = E > 1e-30
    rate = np.polyfit(t[ok], np.log(E[ok]), 1)[0] if ok.sum() > 3 else float("nan")
    target = -2 * HEAT_ALPHA * math.pi ** 2
    return {"heat/decay_rate": float(rate), "heat/decay_rate_over_expected": float(rate / target), "heat/u_absmax": float(np.abs(u.numpy()).max()),
            "heat/energy_t0": float(E[0]), "heat/energy_t1": float(E[-1])}


def burgers_extras(b: Bundle, pts: np.ndarray, shape) -> dict:
    nx, nt = shape
    xt = _to_t(pts, requires_grad=True)
    u = b.fields(xt)
    g = _grad(u, xt, create=False)
    ux = g[:, 0].detach().numpy().reshape(nx, nt)
    U = u.detach().numpy().reshape(nx, nt)
    x = pts[:, 0].reshape(nx, nt)[:, 0]
    t = pts[:, 1].reshape(nx, nt)[0]
    anti = np.abs(U + U[::-1, :]).mean() / max(np.abs(U).mean(), 1e-300)   # simetría impar u(-x,t) = -u(x,t) (solución exacta)
    late = t >= 1.0 / math.pi
    uxmax = np.abs(ux).max(axis=0)
    width = 2 * np.abs(U).max(axis=0) / np.maximum(uxmax, 1e-300)
    front_pos = []
    for j in np.where(late)[0]:
        s = np.sign(U[:, j])
        idx = np.where(s[:-1] * s[1:] < 0)[0]
        front_pos.append(float(x[idx[np.argmin(np.abs(x[idx]))]]) if len(idx) else float("nan"))
    return {"burgers/antisym_err": float(anti), "burgers/ux_max_late": float(uxmax[late].max()) if late.any() else float("nan"),
            "burgers/front_width_late_min": float(width[late].min()) if late.any() else float("nan"),
            "burgers/front_pos_abs_mean_late": float(np.nanmean(np.abs(front_pos))) if front_pos else float("nan")}


def ac_extras(b: Bundle, pts: np.ndarray, shape) -> dict:
    nx, nt = shape
    xt = _to_t(pts, requires_grad=False)
    with torch.no_grad():
        U = b.fields(xt).numpy().reshape(nx, nt)
    x = pts[:, 0].reshape(nx, nt)[:, 0]
    over = float(np.maximum(np.abs(U) - 1.0, 0).max())
    crossings = [int((np.sign(U[:-1, j]) * np.sign(U[1:, j]) < 0).sum()) for j in range(nt)]
    # energía de Ginzburg-Landau de la forma de la PDE del contrato (d=1e-3): E = ∫ d/2 u_x² + 5/4 (1-u²)²
    ux = np.gradient(U, x, axis=0)
    E = np.trapezoid(AC_D / 2 * ux ** 2 + AC_R / 4 * (1 - U ** 2) ** 2, x, axis=0)
    dE = np.diff(E)
    return {"ac/overshoot_max": over, "ac/interface_count_t_end": crossings[-1], "ac/interface_count_t_mid": crossings[nt // 2],
            "ac/energy_t0": float(E[0]), "ac/energy_t_end": float(E[-1]), "ac/energy_increase_frac": float((dE > 1e-12).mean())}


def dfg_extras(b: Bundle, pts: np.ndarray, shape) -> dict:
    """Balance de caudal entrada→salida (sin referencia) y masa de residuo cerca del cilindro."""
    ys = np.linspace(0, 0.41, 201)
    out = {}
    for name, x0 in (("inlet", 0.0), ("outlet", 2.2), ("x1", 1.1)):
        xy = _to_t(np.stack([np.full_like(ys, x0), ys], 1), requires_grad=False)
        with torch.no_grad():
            uu = b.fields(xy)[:, 0].numpy()
        out[name] = float(np.trapezoid(uu, ys))
    return {"dfg/flux_inlet": out["inlet"], "dfg/flux_outlet": out["outlet"], "dfg/flux_imbalance_rel": float((out["outlet"] - out["inlet"]) / max(abs(out["inlet"]), 1e-300)),
            "dfg/flux_mid": out["x1"]}


def piezo_extras(b: Bundle, pts: np.ndarray, shape) -> dict:
    """Magnitud relativa de los campos y consistencia tensión-predicha vs constitutiva (indicador de estado nulo)."""
    xy = _to_t(pts, requires_grad=True)
    o = b.fields(xy)
    u, v, phi, sx = o[:, 0:1], o[:, 1:2], o[:, 2:3], o[:, 3:4]
    gu, gv = _grad(u, xy, create=False), _grad(v, xy, create=False)
    strain_scale = float(np.sqrt((gu[:, 0].detach().numpy() ** 2 + gv[:, 1].detach().numpy() ** 2).mean()))
    tip = torch.tensor([[0.1, 0.0]])
    with torch.no_grad():
        vt = float(b.fields(tip)[0, 1])
    return {"piezo/strain_rms": strain_scale, "piezo/tip_v_model": vt, "piezo/u_rms": float(u.detach().square().mean().sqrt()),
            "piezo/v_rms": float(v.detach().square().mean().sqrt()), "piezo/sigma_xx_rms": float(sx.detach().square().mean().sqrt())}


# ----------------------------------------------------------------------------- despachador
def load_bundle(row: dict) -> Bundle:
    problem, campaign = row["problem"], row["campaign"]
    path = REPO / row["path"]
    if problem == "heat_1d":
        return bundle_heat(path, campaign)
    if problem == "burgers_1d":
        return bundle_burgers(path)
    if problem == "allen_cahn_1d":
        return bundle_allen_cahn(path)
    if problem == "ns2d_cg":
        return bundle_ns2d(path)
    if problem == "piezo_2d":
        return bundle_piezo(path)
    if problem == "dfg_2d":
        if campaign.startswith("cohort-A"):
            return bundle_dfg_ab(path, "A")
        if campaign.startswith("cohort-B"):
            return bundle_dfg_ab(path, "B")
        return bundle_dfg_c(path)
    raise KeyError(problem)
