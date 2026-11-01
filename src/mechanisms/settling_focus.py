"""Settling focus (EXP-0016 v2): decide WHEN to hand off and WHERE to focus from how much each layer-0 RBF basis has settled.

The first layer of an RBF-PIKAN reads the raw coordinates, so every basis (input dimension d, basis g) covers a strip of the domain. The
coefficients c[o, d, g] of that layer are logged every `window` Adam steps. For each basis:

    S_dg(t)   = || c[:, d, g](t) - c[:, d, g](t - window) || / || c[:, d, g](t) ||          (relative change per window)
    rho_dg(t) = S_dg(t) / median of S_dg over the windows that end in (ref_lo, ref_hi]       (rate relative to the early, post-transient one)

A basis is *settled* at step t when rho_dg < rho_star for `persistence` consecutive windows ending at t. Two decisions come out of this:

    when  : k_t = first step >= k_min at which the fraction of settled bases reaches q_star (else k_max: never later than the control);
    where : m(x) = sum_dg phi_dg(x_d) rho_dg / sum_dg phi_dg(x_d), the map of how unsettled each point of the domain is; the refinement-phase
            loss is weighted by w(x) = 1 + alpha * m(x) / mean(m) (normalised to mean 1 here so the loss scale does not change).

Conventions, fixed before running anything: a basis whose coefficients never move (dead, S = 0) has rho = 0 and counts as settled; a basis whose
reference rate is 0 gets rho = 0 for the same reason. The decision at step t uses only snapshots up to t (causal).

This module receives coefficient snapshots and point coordinates only. It reads no reference solution and imports no verifier
(see tests/test_mechanism_isolation.py and tests/test_settling_focus.py).
"""
import math

import numpy as np

DEFAULTS = dict(window=500, k_min=4000, ref_lo=2000, ref_hi=4000, rho_star=0.10, persistence=3, q_star=0.80)
_TINY = 1e-300


def _check(steps, coeffs):
    steps = np.asarray(steps, dtype=np.int64).reshape(-1)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim != 4:
        raise ValueError('coeffs must have shape (T, outputs, inputs, bases)')
    if steps.size != coeffs.shape[0] or steps.size < 2:
        raise ValueError('steps and coeffs must agree on T >= 2')
    if not np.isfinite(coeffs).all():
        raise ValueError('coefficients must be finite')
    if (np.diff(steps) <= 0).any():
        raise ValueError('steps must be strictly increasing')
    return steps, coeffs


def settling_rates(steps, coeffs, window=500, ref_lo=2000, ref_hi=4000):
    """Return (ends, S, rho): S and rho have shape (W, inputs, bases) for the windows ending at `ends`.

    Windows are formed from snapshots exactly `window` steps apart; unequal gaps are an error (the preregistered cadence is fixed).
    """
    steps, coeffs = _check(steps, coeffs)
    if window <= 0:
        raise ValueError('window must be positive')
    if (np.diff(steps) != window).any():
        raise ValueError(f'snapshots must be exactly {window} steps apart')
    norm = np.linalg.norm(coeffs, axis=1)                                   # (T, D, G) over outputs
    diff = np.linalg.norm(coeffs[1:] - coeffs[:-1], axis=1)                 # (T-1, D, G)
    S = diff / np.maximum(norm[1:], _TINY)
    S = np.where((diff == 0) & (norm[1:] == 0), 0.0, S)                     # a dead basis does not move
    ends = steps[1:]
    ref = (ends > ref_lo) & (ends <= ref_hi)
    rho = np.full(S.shape, np.nan)
    if ref.sum() >= 2:
        med = np.median(S[ref], axis=0)                                     # (D, G)
        safe = np.where(med > 0, med, 1.0)
        rho = np.where(med > 0, S / safe, 0.0)
    return ends, S, rho


def settled_mask(rho, rho_star=0.10, persistence=3):
    """settled[w, d, g] is True when rho < rho_star in the `persistence` windows ending at w (nan counts as not settled)."""
    below = np.asarray(rho) < rho_star
    out = np.zeros_like(below)
    for w in range(persistence - 1, below.shape[0]):
        out[w] = below[w - persistence + 1:w + 1].all(axis=0)
    return out


def settling_step(steps, coeffs, k_max, **kw):
    """Decide when to hand off. Result keys: k_t, fired, reason, ends, fraction (settled fraction per window), rho, settled.

    reason: 'settled' | 'not_settled' | 'history_too_short'. k_t = k_max whenever it does not fire.
    """
    p = {**DEFAULTS, **kw}
    if k_max < p['k_min']:
        raise ValueError('k_max must be >= k_min')
    if p['k_min'] < p['ref_hi']:
        raise ValueError('k_min must be >= ref_hi so that a decision never uses a partially built reference rate')
    if not (0 < p['rho_star'] < 1) or not (0 < p['q_star'] <= 1) or p['persistence'] < 1:
        raise ValueError('rho_star in (0, 1), q_star in (0, 1], persistence >= 1')
    steps, coeffs = _check(steps, coeffs)
    ends, S, rho = settling_rates(steps, coeffs, p['window'], p['ref_lo'], p['ref_hi'])
    keep = ends <= k_max
    ends, S, rho = ends[keep], S[keep], rho[keep]
    out = dict(k_t=int(k_max), fired=False, reason='not_settled', ends=ends, fraction=np.full(ends.size, np.nan), rho=rho,
               settled=np.zeros(rho.shape, dtype=bool))
    if not np.isfinite(rho).any():
        out['reason'] = 'history_too_short'
        return out
    settled = settled_mask(rho, p['rho_star'], p['persistence'])
    frac = settled.reshape(settled.shape[0], -1).mean(axis=1)
    out.update(fraction=frac, settled=settled)
    for i in range(ends.size):
        if ends[i] >= p['k_min'] and frac[i] >= p['q_star']:
            out.update(k_t=int(ends[i]), fired=True, reason='settled')
            break
    return out


def rbf_basis(x_norm, centers, width):
    """Gaussian basis phi_g(x) = exp(-((x - c_g) / w)^2) for x of shape (N,), returns (N, G)."""
    x = np.asarray(x_norm, dtype=np.float64).reshape(-1, 1)
    c = np.asarray(centers, dtype=np.float64).reshape(1, -1)
    w = float(width)
    if not math.isfinite(w) or w <= 0:
        raise ValueError('width must be positive')
    return np.exp(-((x - c) / w) ** 2)


def unsettled_map(x_norm, rho_t, centers, width):
    """m(x) = sum_dg phi_dg(x_d) rho_dg / sum_dg phi_dg(x_d) at points x_norm (N, D) in [-1, 1]; rho_t has shape (D, G).

    `centers` and `width` are the layer-0 basis geometry: centers (G,) shared by the dimensions or (D, G); width a scalar or (D,).
    """
    x = np.asarray(x_norm, dtype=np.float64)
    rho = np.asarray(rho_t, dtype=np.float64)
    if x.ndim != 2 or rho.ndim != 2 or x.shape[1] != rho.shape[0]:
        raise ValueError('x_norm must be (N, D) and rho_t (D, G)')
    if not np.isfinite(rho).all():
        raise ValueError('rho_t must be finite (use the last window with a defined rho)')
    centers = np.asarray(centers, dtype=np.float64)
    widths = np.broadcast_to(np.asarray(width, dtype=np.float64), (x.shape[1],))   # scalar, or one width per dimension
    num = np.zeros(x.shape[0])
    den = np.zeros(x.shape[0])
    for d in range(x.shape[1]):
        c = centers if centers.ndim == 1 else centers[d]
        phi = rbf_basis(x[:, d], c, widths[d])
        num += phi @ rho[d]
        den += phi.sum(axis=1)
    return num / np.maximum(den, _TINY)


def focus_weights(values, alpha=1.0, normalize=True):
    """w = 1 + alpha * v / mean(v) for a non-negative per-point signal v (the unsettled map, or |residual| for the comparison arm).

    With normalize=True the weights are rescaled to mean 1 so that the weighted loss keeps the scale of the unweighted one.
    A constant (or all-zero) signal gives uniform weights.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(v).all() or (v < 0).any():
        raise ValueError('values must be finite and non-negative')
    if alpha < 0:
        raise ValueError('alpha must be >= 0')
    mean = float(v.mean())
    w = np.ones_like(v) if mean <= 0 else 1.0 + alpha * v / mean
    return w / w.mean() if normalize else w
