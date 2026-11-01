"""Self-gated collocation sampler (EXP-0014): where to put effort, decided from the model's own signals.

Motivation (EXP-0013): the PDE residual points at the error in some problems (DFG) and not in others (Heat, Allen-Cahn, NS2d-CG, piezo), while the
model's own solution structure (|grad u|) is informative in front problems (Burgers). The sampler therefore mixes two reference-free cues and lets the
measured agreement between them decide which one to trust:
    lam = clip(Spearman(|r|, s), 0, 1)           # alignment of residual |r| with structure s on a fixed candidate set
    p   = u/N + (1-u) * ( lam * w(|r|) + (1-lam) * w(s) )      w(x) = x^power / sum x^power, u = uniform floor
mode='structure' forces lam = 0 (ablation arm: fixed cue). mode='residual' forces lam = 1 (plain RAD-like reference).
No solution values, evaluation grids or verifier imports enter this module; it receives arrays computed by the training code. Sampling is with
replacement and the loss is intentionally not importance-corrected (same convention as src/mechanisms/rad.py).
"""
import numpy as np


def _rank(x):
    """Average ranks (ties share the mean rank), so constant arrays carry no association."""
    x = np.asarray(x, dtype=np.float64)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    return (start + (counts - 1) / 2.0)[inverse]


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 3:
        raise ValueError('arrays must have the same size >= 3')
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def power_weights(x, power=2.0):
    m = np.abs(np.asarray(x, dtype=np.float64).reshape(-1))
    if not m.size or not np.isfinite(m).all():
        raise ValueError('values must be nonempty and finite')
    if not np.isfinite(power) or power <= 0:
        raise ValueError('power must be positive')
    scale = float(m.max())
    if scale == 0:
        return np.full(m.size, 1.0 / m.size)
    w = (m / scale) ** power
    return w / w.sum()


def gate(residual, structure):
    """Trust in the residual cue: alignment of |residual| with structure, clipped to [0, 1]."""
    return float(min(1.0, max(0.0, spearman(np.abs(residual), np.abs(structure)))))


def probabilities(residual, structure, power=2.0, uniform_fraction=0.5, mode='gated'):
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    s = np.asarray(structure, dtype=np.float64).reshape(-1)
    if r.size != s.size or not r.size:
        raise ValueError('residual and structure must have the same nonzero size')
    if not 0 <= uniform_fraction <= 1:
        raise ValueError('uniform_fraction must be within [0, 1]')
    if mode == 'gated':
        lam = gate(r, s)
    elif mode == 'structure':
        lam = 0.0
    elif mode == 'residual':
        lam = 1.0
    else:
        raise ValueError("mode must be 'gated', 'structure' or 'residual'")
    cue = lam * power_weights(r, power) + (1.0 - lam) * power_weights(s, power)
    p = uniform_fraction / r.size + (1.0 - uniform_fraction) * cue
    return p / p.sum(), lam


def sample(candidates, residual, structure, n_points, rng, power=2.0, uniform_fraction=0.5, mode='gated'):
    pts = np.asarray(candidates, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0 or not np.isfinite(pts).all():
        raise ValueError('candidates must be a finite nonempty point matrix')
    if type(n_points) is not int or n_points < 1:
        raise ValueError('n_points must be a positive integer')
    p, lam = probabilities(residual, structure, power, uniform_fraction, mode)
    if len(p) != len(pts):
        raise ValueError('residual/structure and candidate counts differ')
    idx = rng.choice(len(pts), size=n_points, replace=True, p=p)
    return {'points': pts[idx].copy(), 'indices': idx, 'probabilities': p, 'lambda': lam,
            'effective_candidates': float(1 / np.square(p).sum()), 'unique_count': int(np.unique(idx).size)}
