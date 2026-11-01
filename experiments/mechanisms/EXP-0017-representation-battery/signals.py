"""EXP-0017: señales de la representación y atribución por base de una RBF-PIKAN (RBFKAN de pikan_campaign.py). Ninguna lee la referencia.

Modelo: h0 = 2 (x - lo) / (hi - lo) - 1;  h1 = tanh(capa0(h0));  h2 = tanh(capa1(h1));  salida cruda = capa2(h2).
Cada arista (capa l, entrada i, salida o) es  f(z) = sum_g c[o,i,g] phi_g(z) + base[o,i] silu(z),  phi_g(z) = exp(-((z - centro_g) / ancho)^2).
`h_l` es la entrada de la capa l; "base (l, i, g)" es la base g de la entrada i de la capa l.

Señales por punto x (spec EXP-0017, parte 1):
  S1  ‖du/dtheta‖ sobre todos los parametros (u = salida con la transformacion dura de la PDE)
  S2  densidad de bases activas de las capas ocultas:  sum_{l>=1} sum_i sum_g 1[phi(h_l,i) > 0.5]
  S3  curvatura de las funciones de arista:  sum_l sum_{i,o} |f''_{l,o,i}(h_{l,i})|
  S5  desajuste de resolucion:  ‖grad_x u‖ / ‖dh1/dx‖_F   (variacion que pide la solucion / estiramiento que ofrece la primera capa oculta)
Atribucion (parte 2):  A_lig = sum_x w(x) phi_lig(h_li(x)) / sum_x phi_lig(h_li(x)),  con w = |residuo| (o el error, solo para el techo A*);
ocupacion O_lig = sum_x phi_lig(h_li(x));  movimiento Delta_lig = ‖c_l[:, i, g](despues) - c_l[:, i, g](antes)‖ / (‖c_l[:, i, g](antes)‖ + eps).

Este modulo recibe pesos, puntos y una transformacion de salida; no recibe ni importa referencias ni `evals` (ver tests/test_exp0017_signals.py).
"""
import math

import numpy as np
import torch
from torch.func import functional_call, grad, jacfwd, vmap

EPS_DELTA = 1e-12


def layer_inputs(model, xt):
    """Entradas de las tres capas: [h0 (N,2), h1 (N,W), h2 (N,W)]."""
    h0 = 2.0 * (xt - model.lo) / (model.hi - model.lo) - 1.0
    h1 = torch.tanh(model.layers[0](h0))
    h2 = torch.tanh(model.layers[1](h1))
    return [h0, h1, h2]


def phi(z, layer):
    """Bases gaussianas de una capa evaluadas en la entrada z (N, I): (N, I, G)."""
    return torch.exp(-((z.unsqueeze(-1) - layer.centers) / layer.width).square())


def _silu_dd(z):
    s = torch.sigmoid(z)
    return s * (1 - s) * (2 + z * (1 - 2 * s))


@torch.no_grad()
def s2_active_bases(model, xt):
    hs = layer_inputs(model, xt)
    return sum((phi(hs[l], model.layers[l]) > 0.5).sum(dim=(1, 2)) for l in (1, 2)).to(torch.float32)


@torch.no_grad()
def s3_curvature(model, xt, chunk=2048):
    out = []
    for lo in range(0, xt.shape[0], chunk):
        hs = layer_inputs(model, xt[lo:lo + chunk])
        total = 0.0
        for l, layer in enumerate(model.layers):
            z = hs[l]
            u = (z.unsqueeze(-1) - layer.centers) / layer.width                     # (n, I, G)
            phi_dd = (4.0 * u.square() - 2.0) / layer.width ** 2 * torch.exp(-u.square())
            f_dd = torch.einsum("nig,oig->noi", phi_dd, layer.coefficients) + layer.base.unsqueeze(0) * _silu_dd(z).unsqueeze(1)
            total = total + f_dd.abs().sum(dim=(1, 2))
        out.append(total)
    return torch.cat(out)


def s1_jacobian_norm(model, xt, transform, chunk=256):
    """‖du/dtheta‖ por punto (u = transform(x, salida cruda)), con torch.func (grad por muestra)."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def f(p, x):
        raw = functional_call(model, (p, buffers), (x.unsqueeze(0),))
        return transform(x.unsqueeze(0), raw).reshape(())

    g = vmap(grad(f), in_dims=(None, 0))
    out = []
    for lo in range(0, xt.shape[0], chunk):
        gs = g(params, xt[lo:lo + chunk].detach())
        out.append(torch.sqrt(sum(v.reshape(v.shape[0], -1).square().sum(dim=1) for v in gs.values())))
    return torch.cat(out)


def s5_resolution_mismatch(model, xt, transform):
    """‖grad_x u‖ / ‖dh1/dx‖_F. Devuelve (s5, grad_norm, stretch)."""
    x = xt.detach().clone().requires_grad_(True)
    u = transform(x, model(x))
    gnorm = torch.autograd.grad(u.sum(), x)[0].norm(dim=1).detach()

    def h1(xi):
        h0 = 2.0 * (xi.unsqueeze(0) - model.lo) / (model.hi - model.lo) - 1.0
        return torch.tanh(model.layers[0](h0))[0]

    with torch.no_grad():
        J = vmap(jacfwd(h1))(xt.detach())                                          # (N, W, 2)
    stretch = J.reshape(J.shape[0], -1).norm(dim=1)
    return gnorm / stretch.clamp_min(1e-30), gnorm, stretch


@torch.no_grad()
def responsibility(model, xt, weights):
    """A_lig y ocupacion O_lig por capa. weights (N,) >= 0. Devuelve (A, O): listas de tensores (I, G) por capa."""
    hs = layer_inputs(model, xt)
    A, O = [], []
    for l, layer in enumerate(model.layers):
        p = phi(hs[l], layer)                                                      # (N, I, G)
        den = p.sum(dim=0)
        A.append(torch.einsum("n,nig->ig", weights, p) / den.clamp_min(1e-30))
        O.append(den)
    return A, O


def movement(state_before, state_after):
    """Delta_lig por capa (listas de arreglos (I, G)) entre dos state_dict del mismo modelo."""
    out = []
    l = 0
    while f"layers.{l}.coefficients" in state_before:
        cb, ca = state_before[f"layers.{l}.coefficients"].double(), state_after[f"layers.{l}.coefficients"].double()
        out.append((torch.linalg.vector_norm(ca - cb, dim=0) / (torch.linalg.vector_norm(cb, dim=0) + EPS_DELTA)).numpy())
        l += 1
    return out


def flat(parts):
    return np.concatenate([np.asarray(p.detach().cpu() if hasattr(p, "detach") else p, dtype=np.float64).reshape(-1) for p in parts])


def top_share(values, frac=0.2):
    """Fraccion del total que concentra el `frac` superior de los valores (>= 0)."""
    v = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))[::-1]
    n = max(1, int(math.ceil(frac * v.size)))
    total = v.sum()
    return float(v[:n].sum() / total) if total > 0 else float("nan")
