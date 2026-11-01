"""Responsibility focus (EXP-0018): spread the refinement-phase step across the RBF bases of an RBF-PIKAN according to a per-base responsibility A.

A is an (inputs, bases) array per layer. Where it comes from is NOT this module's business: the oracle ceiling feeds it the true-error responsibility A*
(reference-based, labelled `oracle_ceiling` by the experiment); a practical version would feed it a reference-free estimate. The module only receives
numbers, never a reference solution, and imports no verifier.

Arms (all keep architecture and parameter count; only how L-BFGS moves the coefficients changes):
    P  soft focus   coefficients c_lig = s_lig * c'_lig  (L-BFGS runs on c'), s = clip((A / median A)^(beta/2), 0.1, 10)  == diagonal preconditioning with diag(A^beta)
                    (implemented by replacing the edge forward, not with torch parametrizations, so the parameter order and the L-BFGS sums are untouched)
    H  hard focus   only the top-q fraction of bases by A may move; the rest are frozen (gradient exactly zero)
    N  permuted     P's scales shuffled across the bases of the same layer (same distribution of scales, no information)
The linear residual path `base` of every edge is unrestricted in all arms. The median and the top-q threshold are global (all layers together).
"""
import math

import numpy as np
import torch

CLIP = (0.1, 10.0)


def _arr(a):
    return np.asarray(a.detach().cpu() if hasattr(a, "detach") else a, dtype=np.float64)


def scales_from_responsibility(A_list, beta=1.0, clip=CLIP):
    """s per layer, (inputs, bases) arrays. Global median; a base with A <= 0 gets the lower clip."""
    arrs = [_arr(a) for a in A_list]
    med = float(np.median(np.concatenate([a.reshape(-1) for a in arrs])))
    if not math.isfinite(med) or med <= 0:
        raise ValueError("the median responsibility must be positive")
    return [np.clip((np.maximum(a, 0.0) / med) ** (beta / 2.0), clip[0], clip[1]) for a in arrs]


def permute_within_layer(s_list, seed):
    """Shuffle each layer's scales among its own bases (reproducible): same multiset per layer, no information."""
    out = []
    for l, s in enumerate(s_list):
        rng = np.random.default_rng([seed, l])
        out.append(rng.permutation(np.asarray(s).reshape(-1)).reshape(np.asarray(s).shape))
    return out


def top_fraction_masks(A_list, q=0.5):
    """Boolean masks per layer, True for exactly ceil(q * n) bases with the largest A (global ranking, stable ties)."""
    arrs = [_arr(a) for a in A_list]
    flat = np.concatenate([a.reshape(-1) for a in arrs])
    keep = max(1, int(math.ceil(q * flat.size)))
    order = np.argsort(-flat, kind="stable")
    mask = np.zeros(flat.size, dtype=bool)
    mask[order[:keep]] = True
    out, pos = [], 0
    for a in arrs:
        out.append(mask[pos:pos + a.size].reshape(a.shape))
        pos += a.size
    return out


def _forward_with(layer):
    """forward de la arista de la campana (RBFEdge.forward) con la escala y/o la mascara de congelado aplicadas a los coeficientes."""

    def forward(x):
        c = layer.coefficients
        if layer._rf_scale is not None:
            c = c * layer._rf_scale
        if layer._rf_mask is not None:
            c = torch.where(layer._rf_mask, c, c.detach())
        basis = torch.exp(-((x.unsqueeze(-1) - layer.centers) / layer.width).square())
        return torch.einsum("big,oig->bo", basis, c) + torch.nn.functional.linear(torch.nn.functional.silu(x), layer.base)

    return forward


def _prepare(layer):
    if not hasattr(layer, "_rf_scale"):
        layer._rf_scale, layer._rf_mask = None, None
        layer.forward = _forward_with(layer)


def apply_scales(model, s_list):
    """Reparametrize: the optimizer variable becomes c' = c / s and the layer computes c = s * c'. The parameter list and its order do not change."""
    for layer, s in zip(model.layers, s_list):
        _prepare(layer)
        t = torch.as_tensor(np.asarray(s), dtype=layer.coefficients.dtype, device=layer.coefficients.device).unsqueeze(0)
        layer._rf_scale = t
        with torch.no_grad():
            layer.coefficients.div_(t)
    return model


def apply_freeze(model, masks):
    """Only the bases where mask is True can move (gradient exactly zero elsewhere)."""
    for layer, m in zip(model.layers, masks):
        _prepare(layer)
        layer._rf_mask = torch.as_tensor(np.asarray(m), dtype=torch.bool, device=layer.coefficients.device).unsqueeze(0)
    return model


def release(model):
    """Undo the reparametrization keeping the effective coefficients, so the model is a plain RBF-PIKAN again."""
    for layer in model.layers:
        if hasattr(layer, "_rf_scale"):
            if layer._rf_scale is not None:
                with torch.no_grad():
                    layer.coefficients.mul_(layer._rf_scale)
            del layer._rf_scale, layer._rf_mask
            del layer.forward
    return model
