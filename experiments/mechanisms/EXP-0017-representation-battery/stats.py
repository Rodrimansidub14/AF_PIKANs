"""EXP-0017: estadisticos del spec (Spearman, solape del 5 % superior, permutacion con signo, Holm). Solo numpy y scipy; no lee referencias."""
import numpy as np
from scipy.stats import spearmanr


def spearman(a, b):
    r = spearmanr(a, b)[0]
    return float(r) if np.isfinite(r) else 0.0


def top_overlap(a, b, frac=0.05):
    """Solape entre el `frac` superior de a y el de b (azar = frac)."""
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    n = max(1, int(round(frac * a.size)))
    return len(set(np.argsort(a)[-n:].tolist()) & set(np.argsort(b)[-n:].tolist())) / n


def sign_flip_pvalue(d, n_perm=10000, seed=20261020, stat=np.median):
    """p unilateral de H1: stat(d) > 0, invirtiendo al azar el signo de cada diferencia pareada (permutacion con signo, H0 simetrica)."""
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    obs = stat(d)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = np.array([stat(s * d) for s in signs]) if stat is not np.median else np.median(signs * d, axis=1)
    return float((1 + np.sum(perm >= obs - 1e-15)) / (n_perm + 1))


def holm(pvalues):
    """Ajuste de Holm-Bonferroni (valores p ajustados, en el orden de entrada)."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj
