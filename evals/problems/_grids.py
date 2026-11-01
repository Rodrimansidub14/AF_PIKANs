"""Verifier-owned deterministic grids; never used by candidate sampling."""
import numpy as np


def tensor_grid(first, second, nfirst, nsecond, *, interior=False):
    for n in (nfirst,nsecond):
        if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n < 2:
            raise ValueError('grid sizes must be integers >= 2')
    def axis(bounds,n):
        return np.linspace(*bounds,n+2)[1:-1] if interior else np.linspace(*bounds,n)
    a,b=np.meshgrid(axis(first,nfirst),axis(second,nsecond),indexing='ij')
    return a.ravel(),b.ravel()
