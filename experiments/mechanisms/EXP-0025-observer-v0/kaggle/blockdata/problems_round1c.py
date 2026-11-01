"""Problemas de la ronda 1-C (BORRADOR SIN FIRMA): Fisher-KPP 1D y Allen-Cahn con otro coeficiente.

Contrato comun con el Allen-Cahn de la linea base (DeepXDE v1.12.2):
  dominio x in [-1, 1], t in [0, 1]; solucion de referencia en la rejilla (t:101, x:201);
  transformacion dura  u = u0(x) + t * (1 - x^2) * NN(x, t)   (cumple IC y BC exactamente
  siempre que u0(-1), u0(1) coincidan con los valores de frontera).

Nada aqui lanza entrenamiento. Solo define ecuacion, IC, BC, residual, transformacion dura,
receta nominal y carga de la referencia.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------
# Fisher-KPP:  u_t = D u_xx + r u (1 - u)
# Onda viajera exacta de Ablowitz-Zeppetella (velocidad c = 5 sqrt(r D / 6)):
#     u(x,t) = [1 + exp( k (x - x0 - c t) )]^(-2),   k = sqrt(r / (6 D))
# ----------------------------------------------------------------------------------------
KPP_D = 1.0e-3
KPP_R = 10.0
KPP_X0 = -0.2


def kpp_k(D: float = KPP_D, r: float = KPP_R) -> float:
    return float(np.sqrt(r / (6.0 * D)))


def kpp_c(D: float = KPP_D, r: float = KPP_R) -> float:
    return float(5.0 * np.sqrt(r * D / 6.0))


def kpp_exact(x, t, D: float = KPP_D, r: float = KPP_R, x0: float = KPP_X0):
    """Solucion exacta (numpy). Estable: usa expit para evitar overflow de exp."""
    from scipy.special import expit

    z = kpp_k(D, r) * (np.asarray(x) - x0 - kpp_c(D, r) * np.asarray(t))
    # [1+e^z]^-2 = expit(-z)^2
    return expit(-z) ** 2


def kpp_ic(x, D: float = KPP_D, r: float = KPP_R, x0: float = KPP_X0):
    return kpp_exact(x, 0.0, D, r, x0)


# Allen-Cahn:  u_t = d u_xx + 5 (u - u^3),  u(x,0) = x^2 cos(pi x),  u(+-1,t) = -1
AC_D_BASE = 1.0e-3
AC_D_DOSE = 2.0e-4   # elegido por el investigador (2026-10-21): interfaz ~sqrt(5) mas fina que la base (max|u_x| +57 %).
AC_A = 5.0


def ac_ic(x):
    x = np.asarray(x)
    return x**2 * np.cos(np.pi * x)


# ----------------------------------------------------------------------------------------
# Residuales y transformacion dura (torch, import perezoso)
# ----------------------------------------------------------------------------------------
def _grads(u, x, t):
    import torch

    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t, u_xx


def residual_kpp(u, x, t, D: float = KPP_D, r: float = KPP_R):
    """x, t: tensores con requires_grad=True, u = u(x, t)."""
    u_t, u_xx = _grads(u, x, t)
    return u_t - D * u_xx - r * u * (1.0 - u)


def residual_ac(u, x, t, d: float = AC_D_DOSE, a: float = AC_A):
    u_t, u_xx = _grads(u, x, t)
    return u_t - d * u_xx - a * (u - u**3)


def hard_transform(net_out, x, t, ic_fn):
    """u = u0(x) + t (1 - x^2) NN(x,t). ic_fn recibe y devuelve tensores torch."""
    return ic_fn(x) + t * (1.0 - x**2) * net_out


def kpp_ic_torch(x, D: float = KPP_D, r: float = KPP_R, x0: float = KPP_X0):
    import torch

    z = kpp_k(D, r) * (x - x0)
    return torch.sigmoid(-z) ** 2


def ac_ic_torch(x):
    import math
    import torch

    return x**2 * torch.cos(math.pi * x)


# ----------------------------------------------------------------------------------------
# Especificacion declarativa (para el registro y la firma)
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProblemSpec:
    name: str
    equation: str
    params: dict
    domain: dict
    initial_condition: str
    boundary_condition: str
    hard_transform: str
    reference: str
    reference_file: str
    # receta de 1-B T1/V (Adam 4000, L-BFGS 500 it.), verificada en los JSON registrados; no firmada para 1-C
    n_domain: int = 8000
    n_boundary: int = 400
    n_initial: int = 800
    adam_steps: int = 4000
    adam_lr: float = 1e-3
    lbfgs_iters: int = 500      # tope por iteraciones, como en 1-B (lbfgs_status = lbfgs_iter_cap en las 40 corridas AC de T1+V)
    model: str = "RBF-PIKAN ancho 16, precision simple"

    def to_dict(self) -> dict:
        return asdict(self)


SPECS = {
    "fisher_kpp": ProblemSpec(
        name="fisher_kpp_1d",
        equation="u_t - D u_xx - r u (1 - u) = 0",
        params={"D": KPP_D, "r": KPP_R, "x0": KPP_X0,
                "k": kpp_k(), "c": kpp_c(), "frente_ancho~1/k": 1.0 / kpp_k()},
        domain={"x": [-1, 1], "t": [0, 1]},
        initial_condition="u(x,0) = [1+exp(k (x-x0))]^-2",
        boundary_condition="u(-1,t)=1, u(1,t)=0 (la onda exacta difiere en <1e-12; ver provenance)",
        hard_transform="u0(x) + t (1-x^2) NN(x,t)",
        reference="analitica (Ablowitz-Zeppetella), verificada contra Chebyshev+Radau",
        reference_file="fisher_kpp_ref.npz",
    ),
    "allen_cahn_dose": ProblemSpec(
        name="allen_cahn_d2e-4",
        equation="u_t - d u_xx - 5 (u - u^3) = 0",
        params={"d": AC_D_DOSE, "a": AC_A, "d_base": AC_D_BASE},
        domain={"x": [-1, 1], "t": [0, 1]},
        initial_condition="u(x,0) = x^2 cos(pi x)",
        boundary_condition="u(-1,t) = u(1,t) = -1",
        hard_transform="x^2 cos(pi x) + t (1-x^2) NN(x,t)",
        reference="numerica: Chebyshev (N=600) + Radau; solver validado reproduciendo Allen_Cahn.mat en d=1e-3",
        reference_file="allen_cahn_d2e-4_ref.npz",
        n_domain=8000, n_boundary=400, n_initial=800,
    ),
}


def load_reference(key: str, base: str | Path = ".") -> dict:
    """Devuelve dict con x (201,), t (101,), u (101, 201)."""
    f = Path(base) / SPECS[key].reference_file
    z = np.load(f)
    return {"x": z["x"], "t": z["t"], "u": z["u"]}


def relative_l2(u_pred: np.ndarray, u_ref: np.ndarray) -> float:
    return float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))
