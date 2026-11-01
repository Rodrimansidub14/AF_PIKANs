"""EXP-0024 ronda 1, idea A: reubicar los centros de las capas ocultas con los cuantiles de las activaciones.

Mecanismo: para cada entrada i de una capa oculta, los G centros se colocan en cuantiles igualmente espaciados de las activaciones z[:, i] (con un margen en los
extremos), el ancho es `width_factor` veces el espaciado local, y los coeficientes se reajustan por minimos cuadrados con ridge para que la funcion de cada arista
se conserve sobre las muestras. `mix` mezcla cuantiles (1) y rejilla uniforme (0). Solo usa activaciones: ninguna referencia, ningun residuo, ninguna perdida.
"""
import torch


def basis(layer, z):
    return torch.exp(-((z.unsqueeze(-1) - layer.centers.to(z.dtype)) / layer.width.to(z.dtype)) ** 2)


def gram_stats(layer, z):
    """Numero de condicion y rango efectivo (mediana sobre entradas) de la Gram de la base E[phi phi^T]."""
    P = basis(layer, z)
    G = torch.einsum("nig,nih->igh", P, P) / z.shape[0]
    ev = torch.linalg.eigvalsh(G).clamp_min(1e-300)
    return dict(cond=float((ev[:, -1] / ev[:, 0]).median()), eff_rank=float(((ev.sum(-1)) ** 2 / (ev ** 2).sum(-1)).median()))


@torch.no_grad()
def regrid_layer(layer, z, mix=1.0, width_factor=1.75, margin=0.1, min_range=0.05, min_gap=1e-3, ridge=1e-10):
    """Todo el calculo en float64; los buffers y coeficientes se devuelven en el dtype de la capa."""
    dt = layer.coefficients.dtype
    z = z.to(torch.float64)
    N, I = z.shape
    G = layer.centers.shape[-1]
    old = basis(layer, z)                                                   # (N, I, G)
    target = torch.einsum("nig,oig->nio", old, layer.coefficients.to(torch.float64))   # (N, I, O): funcion actual de cada arista
    lo, hi = z.min(0).values, z.max(0).values
    rng = (hi - lo).clamp_min(min_range)
    qs = torch.linspace(0, 1, G, dtype=z.dtype, device=z.device)
    quant = torch.stack([torch.quantile(z[:, i], qs) for i in range(I)])    # (I, G)
    quant[:, 0] = quant[:, 0] - margin * rng
    quant[:, -1] = quant[:, -1] + margin * rng
    mid = (lo + hi) / 2
    uni = mid.unsqueeze(1) + torch.linspace(-1, 1, G, dtype=z.dtype, device=z.device).unsqueeze(0) * (rng / 2 * (1 + 2 * margin)).unsqueeze(1) if mix < 1 else torch.zeros_like(quant)
    c = mix * quant + (1 - mix) * uni
    gaps = torch.diff(c, dim=1).clamp_min(min_gap)
    c = torch.cat([c[:, :1], c[:, :1] + torch.cumsum(gaps, dim=1)], dim=1)
    d = torch.empty_like(c)
    d[:, 1:-1] = (c[:, 2:] - c[:, :-2]) / 2
    d[:, 0], d[:, -1] = c[:, 1] - c[:, 0], c[:, -1] - c[:, -2]
    w = width_factor * d
    layer.centers = c.to(dt)
    layer.width = w.to(dt)
    A = (torch.exp(-((z.unsqueeze(-1) - c) / w) ** 2)).permute(1, 0, 2)     # (I, N, G): con los centros en doble precision
    A = A.to(torch.float64)
    T = target.permute(1, 0, 2)                                             # (I, N, O)
    AtA = A.transpose(1, 2) @ A
    lam = ridge * AtA.diagonal(dim1=1, dim2=2).sum(-1) / G
    X = torch.linalg.solve(AtA + lam[:, None, None] * torch.eye(G, dtype=z.dtype, device=z.device), A.transpose(1, 2) @ T)   # (I, G, O)
    layer.coefficients.copy_(X.permute(2, 0, 1).to(dt))
    fit_err = float(torch.linalg.norm(A @ X - T) / torch.linalg.norm(T).clamp_min(1e-300))
    return dict(fit_rel_error=fit_err)


@torch.no_grad()
def regrid_net(net, s, **kw):
    """Reubica los centros de l1 y l2 (las entradas de las capas ocultas) con las activaciones de los puntos s (ya normalizados a [-1, 1]). Devuelve diagnosticos."""
    out = {}
    z1 = torch.tanh(net.l0(s))
    out["l1_before"] = gram_stats(net.l1, z1)
    out["l1_fit"] = regrid_layer(net.l1, z1, **kw)
    out["l1_after"] = gram_stats(net.l1, z1)
    z2 = torch.tanh(net.l1(z1))
    out["l2_before"] = gram_stats(net.l2, z2)
    out["l2_fit"] = regrid_layer(net.l2, z2, **kw)
    out["l2_after"] = gram_stats(net.l2, z2)
    return out


@torch.no_grad()
def regrid_kan(model, xy, **kw):
    """Igual que regrid_net pero para el RBFKAN de la campana (model.layers[0..2]; entrada normalizada con model.lo/hi)."""
    s = 2.0 * (xy - model.lo) / (model.hi - model.lo) - 1.0
    out = {}
    z1 = torch.tanh(model.layers[0](s))
    out["l1_before"] = gram_stats(model.layers[1], z1.double())
    out["l1_fit"] = regrid_layer(model.layers[1], z1, **kw)
    out["l1_after"] = gram_stats(model.layers[1], z1.double())
    z2 = torch.tanh(model.layers[1](z1))
    out["l2_before"] = gram_stats(model.layers[2], z2.double())
    out["l2_fit"] = regrid_layer(model.layers[2], z2, **kw)
    out["l2_after"] = gram_stats(model.layers[2], z2.double())
    return out
