"""Genera y valida las referencias de la ronda 1-C. No entrena nada.

Solver: colocacion de Chebyshev (N+1 nodos) en x, Radau (rtol 1e-10) en t, interpolacion
baricentrica a la rejilla uniforme (t: 101, x: 201), la misma forma que Allen_Cahn.mat.

Validaciones (se escriben en provenance_round1c.json):
  V1  AC d=1e-3 contra Allen_Cahn.mat (DeepXDE v1.12.2)  -> valida el solver
  V2  Fisher-KPP: solver contra la solucion exacta        -> valida solver y el BC aproximado
  V3  convergencia en N para cada referencia generada
  V4  residual de la solucion exacta por autograd (torch) -> valida residual_kpp / transformacion
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).parent))
import problems_round1c as P  # noqa: E402

OUT = Path(__file__).parent
NX, NT = 201, 101
MAT_URL = "https://raw.githubusercontent.com/lululxvi/deepxde/v1.12.2/examples/dataset/Allen_Cahn.mat"
MAT_SHA = "f9175685675685ab8d2a850f0d212d16c709f55ed89b8fe078493ddcd3724aa0"


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def bary_interp(xn, vals, xq):
    """Interpolacion baricentrica en nodos de Chebyshev (puntos de 2da especie)."""
    N = len(xn) - 1
    w = (-1.0) ** np.arange(N + 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    out = np.empty((vals.shape[0], len(xq)))
    diff = xq[:, None] - xn[None, :]
    exact = np.isclose(diff, 0.0, atol=1e-14)
    diff[exact] = 1.0
    wt = w[None, :] / diff
    num = vals @ wt.T
    den = wt.sum(axis=1)
    out = num / den[None, :]
    rows, cols = np.nonzero(exact)
    for q, j in zip(rows, cols):
        out[:, q] = vals[:, j]
    return out


def solve_reaction_diffusion(d, f, fp, u0_fn, bc_left, bc_right, N, T=1.0, nt=NT, rtol=1e-10, atol=1e-12):
    D, x = cheb(N)
    D2 = D @ D
    xi = x[1:-1]
    L = d * D2[1:-1, 1:-1]
    # contribucion de frontera (Dirichlet constante): u_0 = bc_right en x=+1 (indice 0), bc_left en x=-1 (indice N)
    bnd = d * (D2[1:-1, 0] * bc_right + D2[1:-1, -1] * bc_left)

    def rhs(t, u):
        return L @ u + bnd + f(u)

    def jac(t, u):
        return L + np.diag(fp(u))

    t_eval = np.linspace(0.0, T, nt)
    sol = solve_ivp(rhs, (0.0, T), u0_fn(xi), method="Radau", jac=jac, t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    full = np.empty((nt, N + 1))
    full[:, 0] = bc_right
    full[:, -1] = bc_left
    full[:, 1:-1] = sol.y.T
    xg = np.linspace(-1.0, 1.0, NX)
    # los nodos de cheb estan en orden decreciente; la interpolacion es simetrica en el orden
    return xg, t_eval, bary_interp(x, full, xg), sol.nfev


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def ac_solver(d, N):
    a = P.AC_A
    return solve_reaction_diffusion(
        d, lambda u: a * (u - u**3), lambda u: a * (1 - 3 * u**2), P.ac_ic, -1.0, -1.0, N)


def main():
    prov = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}

    # ---- V1: validar solver con Allen_Cahn.mat -------------------------------------------
    mat_path = OUT / "Allen_Cahn.mat"
    if not mat_path.exists():
        urllib.request.urlretrieve(MAT_URL, mat_path)
    sha = sha256(mat_path)
    mat = loadmat(mat_path)
    u_mat = mat["u"]  # (101, 201)
    prov["V1_AC_base_vs_mat"] = {"mat_sha256": sha, "sha_matches_registered": sha == MAT_SHA}
    for N in (400, 600, 800):
        xg, tg, u, _ = ac_solver(P.AC_D_BASE, N)
        prov["V1_AC_base_vs_mat"][f"N={N}_relL2"] = rel_l2(u, u_mat)
    mat_path.unlink()  # no se redistribuye el .mat; se conserva el sha

    # ---- Referencia AC con coeficiente de dosis ------------------------------------------
    d = P.AC_D_DOSE
    refs = {}
    for N in (600, 900):
        refs[N] = ac_solver(d, N)
    xg, tg, u_ac, _ = refs[600]
    prov["V3_AC_dose_convergence"] = {"d": d, "relL2_N600_vs_N900": rel_l2(refs[600][2], refs[900][2]),
                                     "range_u": [float(u_ac.min()), float(u_ac.max())]}
    np.savez_compressed(OUT / "allen_cahn_d2e-4_ref.npz", x=xg, t=tg, u=u_ac, d=d)

    # ---- Fisher-KPP -----------------------------------------------------------------------
    D_, r_ = P.KPP_D, P.KPP_R
    ex = lambda xi: P.kpp_ic(xi)
    kp = {}
    for N in (600, 900):
        kp[N] = solve_reaction_diffusion(
            D_, lambda u: r_ * u * (1 - u), lambda u: r_ * (1 - 2 * u), ex, 1.0, 0.0, N)
    xg, tg, u_num, _ = kp[600]
    X, T = np.meshgrid(xg, tg)  # (101, 201)
    u_exact = P.kpp_exact(X, T)
    prov["V2_KPP_solver_vs_exact"] = {
        "relL2_N600": rel_l2(u_num, u_exact),
        "relL2_N900": rel_l2(kp[900][2], u_exact),
        "max_abs_N600": float(np.abs(u_num - u_exact).max()),
        "bc_mismatch_left": float(abs(1.0 - P.kpp_exact(-1.0, 1.0))),
        "bc_mismatch_right": float(abs(0.0 - P.kpp_exact(1.0, 0.0))),
        "ic_mismatch_left": float(abs(1.0 - P.kpp_exact(-1.0, 0.0))),
        "front_center_t0_t1": [P.KPP_X0, P.KPP_X0 + P.kpp_c()],
        "k": P.kpp_k(), "c": P.kpp_c(),
    }
    prov["V3_KPP_convergence"] = {"relL2_N600_vs_N900": rel_l2(kp[600][2], kp[900][2])}
    np.savez_compressed(OUT / "fisher_kpp_ref.npz", x=xg, t=tg, u=u_exact, u_numeric=u_num,
                        D=D_, r=r_, x0=P.KPP_X0)

    # ---- V4: residual por autograd de la solucion exacta ---------------------------------
    try:
        import torch

        torch.set_default_dtype(torch.float64)
        rng = np.random.default_rng(0)
        xs = torch.tensor(rng.uniform(-1, 1, 2000)[:, None], requires_grad=True)
        ts = torch.tensor(rng.uniform(0, 1, 2000)[:, None], requires_grad=True)
        z = P.kpp_k() * (xs - P.KPP_X0 - P.kpp_c() * ts)
        u = torch.sigmoid(-z) ** 2
        res = P.residual_kpp(u, xs, ts)
        prov["V4_residual_kpp_exact"] = {"max_abs": float(res.abs().max()), "mean_abs": float(res.abs().mean())}

        # transformacion dura: con NN=0 reproduce la IC; en t=0 cumple IC y en x=+-1 cumple BC
        xb = torch.tensor([[-1.0], [1.0], [-1.0], [1.0]])
        tb = torch.tensor([[0.0], [0.0], [0.7], [0.7]])
        ub = P.hard_transform(torch.randn(4, 1), xb, tb, P.kpp_ic_torch)
        prov["V4_hard_transform_kpp_bc_err"] = [float(ub[0] - 1), float(ub[1] - 0), float(ub[2] - 1), float(ub[3] - 0)]
    except ImportError:
        prov["V4_residual_kpp_exact"] = "torch no disponible; no ejecutado"

    # ---- hashes y salida -------------------------------------------------------------------
    prov["files_sha256"] = {f: sha256(OUT / f) for f in ("fisher_kpp_ref.npz", "allen_cahn_d2e-4_ref.npz")}
    prov["specs"] = {k: v.to_dict() for k, v in P.SPECS.items()}
    (OUT / "provenance_round1c.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in prov.items() if k.startswith("V")}, indent=2))


if __name__ == "__main__":
    main()
