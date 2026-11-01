"""EXP-0022: Gauss-Newton (GN = J^T J / N) de la perdida física de una RBF-PIKAN, y sus bloques por arista.

J = d r / d theta, r = residuo de la PDE por punto (misma expresion que `prob.loss`, sin la media). Se calcula en doble precision con torch.func
(jacrev anidado en x para u_x y u_xx, y en theta para J). Los autovalores distintos de cero de GN son los de K = J J^T / N (N x N), asi que el
espectro sale de una descomposicion densa y exacta, sin Lanczos. Nada de este modulo lee la referencia ni importa `evals`.
"""
import math

import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap


def make_residual(prob, model):
    """Devuelve res_single(p, x): residuo escalar en el punto x (2,) con parametros p (dict)."""
    def u_single(p, x):
        xb = x.unsqueeze(0)
        return prob.transform(xb, functional_call(model, p, (xb,)))[0, 0]

    ux = jacrev(u_single, argnums=1)

    def res_single(p, x):
        g = ux(p, x)
        uxx = jacrev(lambda xx: ux(p, xx)[0])(x)[0]
        u = u_single(p, x)
        if prob.name == "burgers_pinnacle_1d":
            return g[1] + u * g[0] - prob.nu * uxx
        if prob.name == "allen_cahn":
            return g[1] - 0.001 * uxx - 5.0 * (u - u ** 3)
        raise ValueError(prob.name)

    return res_single


def param_dict(model):
    return {n: q.detach() for n, q in model.named_parameters()}


def residual(prob, model, X):
    res = make_residual(prob, model)
    return vmap(res, in_dims=(None, 0))(param_dict(model), X)


def jacobian(prob, model, X, chunk=128, dtype=torch.float32):
    """J (N, P) con las columnas en el orden de model.named_parameters(). Se calcula en el dtype del modelo y se guarda en `dtype`."""
    res = make_residual(prob, model)
    p = param_dict(model)
    names = [n for n, _ in model.named_parameters()]
    P = sum(p[n].numel() for n in names)
    J = torch.empty(X.shape[0], P, dtype=dtype)
    for lo in range(0, X.shape[0], chunk):
        xs = X[lo:lo + chunk]
        Jd = jacrev(lambda pp: vmap(res, in_dims=(None, 0))(pp, xs))(p)
        J[lo:lo + xs.shape[0]] = torch.cat([Jd[n].reshape(xs.shape[0], -1) for n in names], 1).to(dtype)
    return J


def layout(model):
    """Indices de columna: edges[(l, o, i)] -> array(G) de columnas de c[o, i, :]; rest -> columnas de `base` (y cualquier otro parametro)."""
    off, edges, rest = 0, {}, []
    for n, q in model.named_parameters():
        k = q.numel()
        if n.endswith("coefficients"):
            l = int(n.split(".")[1])
            O, I, G = q.shape
            idx = off + np.arange(k).reshape(O, I, G)
            for o in range(O):
                for i in range(I):
                    edges[(l, o, i)] = idx[o, i]
        else:
            rest.extend(range(off, off + k))
        off += k
    return edges, np.array(rest, dtype=np.int64), off


def gram_spectrum(J, scale=None):
    """Autovalores (descendentes) de K = (J S)(J S)^T / N, con S diagonal opcional (vector). Producto en float32, descomposicion en float64."""
    N = J.shape[0]
    A = J if scale is None else J * scale.to(J.dtype)
    K = (A @ A.T).double() / N
    del A
    return torch.linalg.eigvalsh(K).flip(0).clamp_min(0).numpy()


def edge_blocks(J, edges, rows=512):
    """B_e = J_e^T J_e / N (G x G) por arista, en float64, acumulado por bloques de filas (poca memoria). Devuelve (E, G, G), claves, columnas (E, G)."""
    keys = list(edges.keys())
    cols = np.stack([edges[k] for k in keys])                   # (E, G)
    flat = torch.as_tensor(cols.reshape(-1))
    B = torch.zeros(len(keys), cols.shape[1], cols.shape[1], dtype=torch.float64)
    for lo in range(0, J.shape[0], rows):
        Jc = J[lo:lo + rows][:, flat].reshape(-1, len(keys), cols.shape[1]).double()
        B += torch.einsum("nea,neb->eab", Jc, Jc)
    return B / J.shape[0], keys, cols


def inv_sqrt_psd(B, eps_rel):
    """(B + eps I)^(-1/2) para un lote de matrices simetricas; eps = eps_rel * traza/n por matriz."""
    n = B.shape[-1]
    eps = eps_rel * B.diagonal(dim1=-2, dim2=-1).sum(-1) / n
    w, V = torch.linalg.eigh(B + eps[:, None, None] * torch.eye(n, dtype=B.dtype))
    return V @ torch.diag_embed(w.clamp_min(1e-300).rsqrt()) @ V.transpose(-1, -2)


def _sumsq(J, cols, rows=512):
    acc = torch.zeros(len(cols), dtype=torch.float64)
    c = torch.as_tensor(cols)
    for lo in range(0, J.shape[0], rows):
        acc += J[lo:lo + rows][:, c].double().square().sum(0)
    return acc / J.shape[0]


def jacobi_scale(J, rows=512):
    d = torch.cat([_sumsq(J, np.arange(lo, min(lo + 4096, J.shape[1])), rows) for lo in range(0, J.shape[1], 4096)])
    return d.clamp_min(1e-300).rsqrt().to(J.dtype)


def apply_edge_precond(J, edges, rest, eps_rel=1e-3, rows=512):
    """J S con S = bloque diagonal (B_e + eps I)^(-1/2) por arista y 1/sqrt(diag GN) en el resto. Devuelve J S (float32)."""
    B, keys, cols = edge_blocks(J, edges, rows)
    S = inv_sqrt_psd(B, eps_rel).to(J.dtype)                      # (E, G, G)
    flat = torch.as_tensor(cols.reshape(-1))
    r = torch.as_tensor(rest)
    d = _sumsq(J, rest, rows).clamp_min(1e-300).rsqrt().to(J.dtype)
    out = torch.empty_like(J)
    for lo in range(0, J.shape[0], rows):
        blk = J[lo:lo + rows]
        Jc = blk[:, flat].reshape(-1, len(keys), cols.shape[1])
        out[lo:lo + rows][:, flat] = torch.einsum("nea,eab->neb", Jc, S).reshape(blk.shape[0], -1)
        out[lo:lo + rows][:, r] = blk[:, r] * d
    return out


def apply_input_precond(J, edges, rest, eps_rel=1e-3, rows=512):
    """Descriptivo: bloque por entrada (l, i): todos los coeficientes c[:, i, :] que leen la misma entrada (O*G columnas)."""
    groups = {}
    for (l, o, i), c in edges.items():
        groups.setdefault((l, i), []).append(c)
    out = torch.empty_like(J)
    for key, cl in groups.items():
        c = torch.as_tensor(np.concatenate(cl))
        Jb = J[:, c].double()
        B = (Jb.T @ Jb / J.shape[0])[None]
        S = inv_sqrt_psd(B, eps_rel)[0].to(J.dtype)
        out[:, c] = J[:, c] @ S
    r = torch.as_tensor(rest)
    d = _sumsq(J, rest, rows).clamp_min(1e-300).rsqrt().to(J.dtype)
    out[:, r] = J[:, r] * d
    return out


def kappa(eigs, m):
    """kappa_eff(m) = lambda_1 / lambda_m (m en 1..len)."""
    return float(eigs[0] / eigs[m - 1]) if len(eigs) >= m and eigs[m - 1] > 0 else float("inf")


def rayleigh_rel(J, d, lam1):
    """(d^T GN d / d^T d) / lambda_1 con GN = J^T J / N."""
    Jd = (J.double() @ d.double()) if J.shape[0] * J.shape[1] < 3e8 else torch.cat([J[lo:lo + 1024].double() @ d.double() for lo in range(0, J.shape[0], 1024)])
    return float((Jd.square().sum() / J.shape[0]) / (d.double().square().sum()) / lam1)


def flat_params(model):
    return torch.cat([q.detach().reshape(-1) for _, q in model.named_parameters()])
