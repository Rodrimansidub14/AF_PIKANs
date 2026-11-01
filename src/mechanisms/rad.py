"""Residual-adaptive distribution with fixed count and explicit random generator.

No solution values, validation grids or verifier imports enter this module.
Sampling is with replacement: multiplicity implements the target discrete
distribution. Loss is intentionally not importance-corrected.
"""
import numpy as np


def residual_probabilities(residual, power=2., uniform_fraction=.5):
    r=np.asarray(residual,dtype=np.float64).reshape(-1)
    if not r.size or not np.isfinite(r).all():
        raise ValueError('residual must be nonempty and finite')
    if not np.isfinite(power) or power<=0 or not 0<=uniform_fraction<=1:
        raise ValueError('power must be positive and uniform_fraction within [0,1]')
    magnitude=np.abs(r)
    scale=float(magnitude.max())
    if scale==0:
        return np.full(r.size,1/r.size)
    weights=(magnitude/scale)**power
    p=uniform_fraction/r.size+(1-uniform_fraction)*weights/weights.sum()
    return p/p.sum()


def sample_rad(candidates,residual,n_points,rng,power=2.,uniform_fraction=.5):
    points=np.asarray(candidates,dtype=np.float64)
    if points.ndim!=2 or len(points)==0 or not np.isfinite(points).all():
        raise ValueError('candidates must be a finite nonempty point matrix')
    if type(n_points) is not int or n_points<1:
        raise ValueError('n_points must be a positive integer')
    p=residual_probabilities(residual,power,uniform_fraction)
    if len(p)!=len(points):
        raise ValueError('residual and candidate counts differ')
    indices=rng.choice(len(points),size=n_points,replace=True,p=p)
    return {'points':points[indices].copy(),'indices':indices,'probabilities':p,
            'effective_candidates':float(1/np.square(p).sum()),
            'unique_count':int(np.unique(indices).size)}

