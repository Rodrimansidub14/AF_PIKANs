"""EXP-0013 — cálculo de señales SIN referencia (rasgos).

Entradas: un Bundle (modelo + PDE) y una malla de coordenadas. NUNCA importa `labels`, `evals` ni lee valores de referencia.
Cada señal se devuelve con clave `grupo/nombre`; lo que no se puede calcular se devuelve en `nulls` con su razón (nunca como cero).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import stats

from adapters import Bundle, _grad, _to_t

EPS = 1e-300


def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)).ravel())
    n = len(x)
    s = x.sum()
    if n == 0 or s <= 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


def top_mass(x2: np.ndarray, frac: float) -> float:
    k = max(1, int(round(frac * len(x2))))
    s = x2.sum()
    return float(np.sort(x2)[-k:].sum() / s) if s > 0 else float("nan")


def entropy_norm(x2: np.ndarray) -> float:
    s = x2.sum()
    if s <= 0:
        return float("nan")
    p = x2 / s
    return float(-(p * np.log(np.clip(p, EPS, None))).sum() / math.log(len(p)))


def residual_stats(prefix: str, r: np.ndarray) -> dict:
    a = np.abs(r)
    r2 = r ** 2
    return {f"{prefix}/rms": float(np.sqrt(r2.mean())), f"{prefix}/p50": float(np.percentile(a, 50)), f"{prefix}/p95": float(np.percentile(a, 95)),
            f"{prefix}/p99": float(np.percentile(a, 99)), f"{prefix}/max": float(a.max()), f"{prefix}/gini": gini(a),
            f"{prefix}/top1_mass": top_mass(r2, 0.01), f"{prefix}/top5_mass": top_mass(r2, 0.05), f"{prefix}/entropy_norm": entropy_norm(r2),
            f"{prefix}/kurtosis": float(stats.kurtosis(a))}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok])[0])


def structure_fields(b: Bundle, pts: np.ndarray, chunk: int = 4096):
    """|∇u| y |Δu| del campo primario (autograd), más las derivadas por coordenada para escalas de estructura."""
    gn, lap, per_dim = [], [], []
    d = pts.shape[1]
    for i in range(0, len(pts), chunk):
        xy = _to_t(pts[i:i + chunk])
        f = b.fields(xy)[:, b.primary:b.primary + 1]
        g = _grad(f, xy)
        L = torch.zeros(len(xy), 1)
        for k in range(d):
            L = L + _grad(g[:, k:k + 1], xy)[:, k:k + 1]
        gn.append(g.norm(dim=1).detach().numpy())
        lap.append(L.abs().squeeze(-1).detach().numpy())
        per_dim.append(g.abs().detach().numpy())
    return np.concatenate(gn), np.concatenate(lap), np.concatenate(per_dim)


def values_field(b: Bundle, pts: np.ndarray, chunk: int = 8192) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(pts), chunk):
            out.append(b.fields(_to_t(pts[i:i + chunk], requires_grad=False))[:, b.primary].numpy())
    return np.concatenate(out)


def layer_inputs(b: Bundle, pts: np.ndarray, limit: int = 8192):
    """Entradas normalizadas de cada capa RBF sobre un subconjunto fijo de la malla."""
    idx = np.linspace(0, len(pts) - 1, min(limit, len(pts))).astype(int)
    h = b.h0(_to_t(pts[idx], requires_grad=False))
    outs = []
    with torch.no_grad():
        for li, layer in enumerate(b.layers):
            outs.append(h)
            h = torch.tanh(layer(h)) if li < len(b.layers) - 1 else layer(h)
    return outs


def basis_signals(b: Bundle, pts: np.ndarray, struct_widths: dict) -> tuple[dict, dict]:
    sig, nulls = {}, {}
    if b.layers is None or b.h0 is None:
        nulls["basis/*"] = "modelo sin bases RBF (MLP)"
        return sig, nulls
    try:
        ins = layer_inputs(b, pts)
    except Exception as e:  # noqa: BLE001
        nulls["basis/*"] = f"no se pudieron obtener las entradas por capa: {type(e).__name__}: {e}"[:200]
        return sig, nulls
    for li, (layer, h) in enumerate(zip(b.layers, ins)):
        if not hasattr(layer, "centers"):
            continue
        centers, width = layer.centers, layer.width
        with torch.no_grad():
            if h.shape[1] != layer.coefficients.shape[1]:   # piezo: la capa 0 recibe 4 entradas; solo se cubren las 2 primeras
                h = h[:, :layer.coefficients.shape[1]] if h.shape[1] > layer.coefficients.shape[1] else h
                if h.shape[1] < layer.coefficients.shape[1]:
                    nulls[f"basis/l{li}/*"] = f"entradas disponibles ({h.shape[1]}) menores que las de la capa ({layer.coefficients.shape[1]})"
                    continue
            basis = torch.exp(-((h.unsqueeze(-1) - centers) / width) ** 2)        # (N, in, G)
            occ = basis.mean(dim=(0, 1))                                          # (G,)
            edge_max = basis.amax(dim=0)                                          # (in, G)
            p = occ / occ.sum().clamp_min(EPS)
            ent = float(-(p * torch.log(p.clamp_min(EPS))).sum())
            coef = layer.coefficients.detach()                                    # (out, in, G)
            edge_norm = coef.norm(dim=2)                                          # (out, in)
            small = float((edge_norm < 1e-3 * edge_norm.max()).double().mean())
            per_basis = coef.norm(dim=(0, 1))
        sig[f"basis/l{li}/occ_mean"] = float(occ.mean())
        sig[f"basis/l{li}/occ_min"] = float(occ.min())
        sig[f"basis/l{li}/dead_fraction"] = float((edge_max < 0.01).double().mean())
        sig[f"basis/l{li}/eff_bases"] = float(math.exp(ent))
        sig[f"basis/l{li}/coef_norm"] = float(coef.norm())
        sig[f"basis/l{li}/small_edge_fraction"] = small
        sig[f"basis/l{li}/coef_basis_gini"] = gini(per_basis.numpy())
        if li < len(b.layers) - 1:
            sig[f"basis/l{li}/tanh_saturation"] = float((torch.tanh(layer(h)).abs() > 0.99).double().mean())
    # relación entre escala de la estructura y espaciado/ancho de bases (capa 0, dimensiones espaciales)
    l0 = b.layers[0]
    if hasattr(l0, "centers") and b.phys_per_unit is not None:
        spacing = float(l0.centers[1] - l0.centers[0])
        width = float(l0.width)
        for i, name in enumerate(b.input_names):
            ppu = b.phys_per_unit[i]
            sig[f"scale/spacing_phys_{name}"] = spacing * ppu
            sig[f"scale/width_phys_{name}"] = width * ppu
            sw = struct_widths.get(name)
            if sw is not None and math.isfinite(sw):
                sig[f"scale/ratio_struct_to_spacing_{name}"] = sw / (spacing * ppu)
                sig[f"scale/ratio_struct_to_width_{name}"] = sw / (width * ppu)
    return sig, nulls


def jacobian_signals(b: Bundle, pts: np.ndarray, count: int = 256) -> dict:
    params = [p for p in b.model.parameters() if p.requires_grad]
    idx = np.linspace(0, len(pts) - 1, min(count, len(pts))).astype(int)
    rows = []
    for k in idx:
        pt = _to_t(pts[k:k + 1], requires_grad=False)
        f = b.fields(pt)[:, b.primary].sum()
        g = torch.autograd.grad(f, params, allow_unused=True)
        rows.append(torch.cat([(v if v is not None else torch.zeros_like(p)).reshape(-1) for v, p in zip(g, params)]))
    sv = torch.linalg.svdvals(torch.stack(rows))
    pv = sv / sv.sum()
    erank = float(torch.exp(-(pv * torch.log(pv.clamp_min(EPS))).sum()))
    return {"jac/effective_rank": erank, "jac/effective_rank_per_param": erank / sum(p.numel() for p in params), "jac/cond": float(sv[0] / sv[-1].clamp_min(EPS)),
            "jac/sigma_max": float(sv[0]), "jac/n_points": len(idx)}


def param_signals(b: Bundle) -> dict:
    flat = torch.cat([p.detach().reshape(-1) for p in b.model.parameters()])
    return {"param/count": int(flat.numel()), "param/norm": float(flat.norm()), "param/absmax": float(flat.abs().max()), "param/small_fraction": float((flat.abs() < 1e-3).double().mean())}


def grad_term_signals(b: Bundle, pts: np.ndarray, count: int = 1024) -> tuple[dict, dict]:
    """Norma de gradiente por término residual y cosenos entre términos (solo si hay >= 2 componentes)."""
    nulls = {}
    idx = np.linspace(0, len(pts) - 1, min(count, len(pts))).astype(int)
    comps = b.res_tensors(pts[idx])
    names = list(comps)
    if len(names) < 2:
        nulls["gradterm/*"] = "un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir"
        return {}, nulls
    params = [p for p in b.model.parameters() if p.requires_grad]
    gvec = {}
    for n in names:
        loss = 0.5 * (comps[n] ** 2).mean()
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        gvec[n] = torch.cat([(v if v is not None else torch.zeros_like(p)).reshape(-1) for v, p in zip(g, params)])
    sig = {f"gradterm/norm_{n}": float(gvec[n].norm()) for n in names}
    norms = np.array([float(gvec[n].norm()) for n in names])
    sig["gradterm/norm_ratio_max_min"] = float(norms.max() / max(norms.min(), EPS))
    cos = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = float(torch.dot(gvec[names[i]], gvec[names[j]]) / (gvec[names[i]].norm() * gvec[names[j]].norm()).clamp_min(EPS))
            sig[f"gradterm/cos_{names[i]}__{names[j]}"] = c
            cos.append(c)
    sig["gradterm/cos_min"] = float(min(cos))
    sig["gradterm/cos_mean"] = float(np.mean(cos))
    return sig, nulls


def compute_signals(b: Bundle, pts: np.ndarray, shape, axes) -> tuple[dict, dict]:
    sig, nulls = {}, {}
    # --- residuo
    res = b.residuals(pts)
    names = list(res)
    allr = np.sqrt(sum(res[n] ** 2 for n in names))
    for n in names:
        if len(names) > 1:
            sig.update(residual_stats(f"res/{n}", res[n]))
    sig.update(residual_stats("res/all", allr))
    sig["res/n_points"] = int(len(allr))
    if shape is not None and len(axes) == 2 and axes[1] == "t":
        R = (allr ** 2).reshape(shape)
        q = np.array_split(np.arange(shape[1]), 4)
        tot = R.sum()
        for k, ix in enumerate(q):
            sig[f"res/t_quartile{k+1}_mass"] = float(R[:, ix].sum() / tot)
        xs = pts[:, 0].reshape(shape)[:, 0]
        ts = pts[:, 1].reshape(shape)[0]
        sig["res/x_centroid"] = float((R.sum(axis=1) * xs).sum() / tot)
        sig["res/t_centroid"] = float((R.sum(axis=0) * ts).sum() / tot)
    # --- estructura de la solución y alineación residuo–gradiente (sin referencia)
    gn, lap, per_dim = structure_fields(b, pts)
    sig["struct/grad_norm_max"] = float(gn.max())
    sig["struct/grad_norm_p95"] = float(np.percentile(gn, 95))
    sig["struct/laplacian_abs_max"] = float(lap.max())
    sig["align/spearman_res_gradnorm"] = spearman(allr, gn)
    sig["align/spearman_res_laplacian"] = spearman(allr, lap)
    sig["align/top5res_in_top5grad"] = float(len(np.intersect1d(np.argsort(allr)[-max(1, len(allr) // 20):], np.argsort(gn)[-max(1, len(gn) // 20):])) / max(1, len(allr) // 20))
    vals = values_field(b, pts)
    vmax = float(np.abs(vals).max())
    struct_widths = {}
    for i, name in enumerate(b.input_names):
        mx = float(per_dim[:, i].max())
        struct_widths[name] = (2 * vmax / mx) if mx > 0 else float("nan")
        sig[f"struct/width_{name}"] = struct_widths[name]
    sig["struct/value_absmax"] = vmax
    # --- bases, jacobiano, parámetros, términos
    s, n = basis_signals(b, pts, struct_widths)
    sig.update(s)
    nulls.update(n)
    try:
        sig.update(jacobian_signals(b, pts))
    except Exception as e:  # noqa: BLE001
        nulls["jac/*"] = f"{type(e).__name__}: {e}"[:200]
    sig.update(param_signals(b))
    try:
        s, n = grad_term_signals(b, pts)
        sig.update(s)
        nulls.update(n)
    except Exception as e:  # noqa: BLE001
        nulls["gradterm/*"] = f"{type(e).__name__}: {e}"[:200]
    # --- específicas del problema
    if b.extra_fn is not None:
        try:
            sig.update(b.extra_fn(b, pts, shape))
        except Exception as e:  # noqa: BLE001
            nulls["problem/*"] = f"{type(e).__name__}: {e}"[:200]
    return sig, nulls
