"""Scrambled Sobol pools; independent seeds, powers of two, no adaptation."""
import numpy as np
from scipy.stats import qmc


def sobol(n,d,seed):
    if n<2 or n&(n-1): raise ValueError('Sobol pool sizes must be powers of two >=2')
    return qmc.Sobol(d=d,scramble=True,seed=seed).random_base2(int(np.log2(n)))


def pools(problem,seed,n=8192,boundary=512):
    if problem not in ('heat','burgers'): raise ValueError('unknown problem')
    low=0. if problem=='heat' else -1.
    def interior(count,offset):
        p=sobol(count,2,10*seed+offset); p[:,0]=low+(1-low)*p[:,0]; return p
    t=sobol(boundary,1,10*seed+3)[:,0]
    initial=sobol(2*boundary,1,10*seed+4)[:,0]
    return {'interior':interior(n,1),'diagnostic':interior(min(n,2048),2),
            'left':np.column_stack([np.full(boundary,low),t]),
            'right':np.column_stack([np.ones(boundary),t]),
            'initial':np.column_stack([low+(1-low)*initial,np.zeros(2*boundary)])}
