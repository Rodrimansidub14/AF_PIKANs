"""Stall handoff (EXP-0016): hand the model from Adam to L-BFGS when its own training loss stops improving.

Motivation (EXP-0011/0012, EXP-0013): the state Adam hands to L-BFGS matters, and the fixed Adam budget of the recipe is not tied to anything the
model itself measures. This module turns the logged training loss into one decision, the Adam step k_t at which to hand off.

Signal (fixed in the pre-registration, no tuning on evaluation data):
    - the loss history (one value every `every` steps) is grouped into blocks of `block` steps; m_j = mean log L over block j;
    - r_j = (m_{j-1} - m_j) / block is the improvement of log-loss per step;
    - r_ref = median of r_j over the blocks that end in (ref_lo, ref_hi]  (past the initial transient);
    - rho_j = r_j / r_ref;  the handoff fires at the end of the first block that ends at or after `k_min` and completes `persistence`
      consecutive blocks with rho_j < rho_star.
    - if it never fires, k_t = k_max (the recipe's Adam budget): the mechanism never hands off later than the control.

The function is causal: the decision at step k depends only on the history up to k, so a closed-loop trainer can call it on the running prefix and
get the same k_t as the offline analysis. The module receives losses only. It reads no reference solution, no evaluation grid, and imports no
verifier (see tests/test_mechanism_isolation.py).
"""
import math

import numpy as np

DEFAULTS = dict(block=500, k_min=4000, ref_lo=2000, ref_hi=4000, rho_star=0.10, persistence=3)


def _check(steps, losses):
    steps = np.asarray(steps, dtype=np.int64).reshape(-1)
    losses = np.asarray(losses, dtype=np.float64).reshape(-1)
    if steps.size != losses.size or steps.size < 2:
        raise ValueError('steps and losses must have the same size >= 2')
    if not np.isfinite(losses).all() or (losses <= 0).any():
        raise ValueError('losses must be finite and positive')
    if (np.diff(steps) <= 0).any():
        raise ValueError('steps must be strictly increasing')
    return steps, losses


def block_rates(steps, losses, block=500):
    """Complete blocks only. Returns (block_end_steps, r) with r[0] = nan (no previous block).

    Block j holds the logged steps in (j*block, (j+1)*block] and is complete when the history reaches (j+1)*block.
    """
    steps, losses = _check(steps, losses)
    if block <= 0:
        raise ValueError('block must be positive')
    last = int(steps[-1])
    n_blocks = last // block
    ends, means = [], []
    for j in range(n_blocks):
        lo, hi = j * block, (j + 1) * block
        sel = (steps > lo) & (steps <= hi)
        if not sel.any():
            break
        ends.append(hi)
        means.append(float(np.log(losses[sel]).mean()))
    ends = np.asarray(ends, dtype=np.int64)
    means = np.asarray(means, dtype=np.float64)
    r = np.full(means.size, np.nan)
    if means.size > 1:
        r[1:] = (means[:-1] - means[1:]) / block
    return ends, r


def handoff_step(steps, losses, k_max, **kw):
    """Return the handoff decision for a loss history.

    Result keys: k_t (int), fired (bool), reason ('stall' | 'no_stall' | 'no_reference_rate' | 'history_too_short'),
    r_ref (float or None), block_ends, rho (arrays, rho is nan where undefined).
    """
    p = {**DEFAULTS, **kw}
    block, k_min, rho_star, persistence = p['block'], p['k_min'], p['rho_star'], p['persistence']
    if k_max < k_min:
        raise ValueError('k_max must be >= k_min')
    if not (0 < rho_star < 1) or persistence < 1:
        raise ValueError('rho_star must be in (0, 1) and persistence >= 1')
    ends, r = block_rates(steps, losses, block)
    keep = ends <= k_max
    ends, r = ends[keep], r[keep]
    out = dict(k_t=int(k_max), fired=False, reason='no_stall', r_ref=None, block_ends=ends, rho=np.full(r.size, np.nan))
    ref = (ends > p['ref_lo']) & (ends <= p['ref_hi']) & np.isfinite(r)
    if ref.sum() < 2:
        out['reason'] = 'history_too_short'
        return out
    r_ref = float(np.median(r[ref]))
    out['r_ref'] = r_ref
    if not math.isfinite(r_ref) or r_ref <= 0:
        out['reason'] = 'no_reference_rate'
        return out
    rho = r / r_ref
    out['rho'] = rho
    run = 0
    for i in range(rho.size):
        if ends[i] < k_min or not np.isfinite(rho[i]):
            run = 0
            continue
        run = run + 1 if rho[i] < rho_star else 0
        if run >= persistence:
            out.update(k_t=int(ends[i]), fired=True, reason='stall')
            return out
    return out
