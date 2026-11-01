"""Residual terms of the training loss as a vector (EXP-0028, phase 4): the Gauss-Newton blocks of the Gram preconditioner for problems with
several equations and boundary terms (NS2d-CG, DFG, piezo), computed on the REAL model with torch.func.functional_call.

Interface (per problem):  prob.residual_terms(model_fn, X) -> [(r, w, name), ...]
    model_fn : x (n, d) -> (n, m), differentiable in x (the real model, or the model with replaced coefficients).
    X        : the points of the problem: a tensor, or a dict of tensors (domain, inlet, ...), as `prob.pools` gives them (or a subset of them).
    r        : (n_t,) residual of one term at its own points; w: the weight of the term in the loss; name: label.
The loss of the problem is  sum_t w_t * mean(r_t^2)  (checked against `prob.loss` in tests/test_residual_terms.py), and the Gauss-Newton matrix is
    B = sum_t (w_t / N_t) * J_t^T J_t,   J_t = d r_t / d (effective coefficients),
which for a single term with w = 1 is the scalar J^T J / N of `gram_preconditioning.gn_blocks`. A dict X is processed in chunks of the SAME index range
for every key; a term whose points are exhausted is simply absent from that chunk (adapters skip empty keys).

`gram_preconditioning.gn_blocks` calls `gn_blocks_terms` only if the problem defines `residual_terms`; Burgers / Allen-Cahn do not (their scalar path is
untouched) but have adapters here so the two paths can be compared. Adapters are installed with `install(prob)` (no change to the campaign files):
    ns2d_cg (NSProblem, 8 terms, weight 1, full batch)    burgers_pinnacle_1d / allen_cahn (hard transform, one term)
    DFGTerms (src/research_pipeline/sciml/dfg.py: DFGRBFPIKAN, output_scales are inside the model's forward, so functional_call carries them)
    piezo (converse and direct, PiezoProblem of EXP-0028/piezo): `piezo_residual_terms`, full model (trunk + phase construction + scales + hard BCs) in physical units.
Derivatives wrt x are taken with torch.func (vmap(jacrev(jacrev))), not autograd.grad: the latter does not compose with jacrev over the coefficients.

This module reads no reference solution, no evaluation grid and imports no verifier (tests/test_residual_terms.py, tests/test_mechanism_isolation.py).
"""
import functools
import math

import torch
from torch.func import functional_call, jacrev, vmap

NS_NU = 0.01             # NSProblem's module constant NU; the tests check it through the loss
DFG_NU, DFG_U_MEAN, DFG_U_MAX, DFG_DIAMETER = 0.001, 0.2, 0.3, 0.1
DFG_Y = (0.0, 0.41)
MAX_CHUNK = 16           # jacrev over the coefficients of a point-batched graph costs O(chunk^2) time and memory: 64 points of NS at width 16 asked 6.6 GB; the result does not depend on the chunk


# ------------------------------------------------------------------ the real model as a function of the effective coefficients
def _layers_path(model):
    """Attribute path of the RBF edge layers: model.layers (campaign RBFKAN, DFGRBFPIKAN) or model.net.layers (piezo trunk)."""
    if hasattr(model, "layers"):
        return "layers", model.layers
    return "net.layers", model.net.layers


def coefficient_overrides(model, coefs):
    """Dict for functional_call: the layers' `coefficients` := effective coefficients, and `pre_P` := I where a layer is preconditioned
    (so the Jacobian is wrt the effective coefficients c, whatever P is: same convention as gram_preconditioning.gn_blocks)."""
    path, layers = _layers_path(model)
    over = {}
    for k, layer in enumerate(layers):
        over[f"{path}.{k}.coefficients"] = coefs[k]
        if hasattr(layer, "pre_P"):
            over[f"{path}.{k}.pre_P"] = layer.pre_layout.identity(coefs[k].dtype, coefs[k].device)
    return over


def make_model_fn(model, coefs):
    """x (n, d) -> model(x) with the layers' coefficients replaced by `coefs` (tuple of (O, I, G) tensors)."""
    over = coefficient_overrides(model, coefs)
    return lambda x: functional_call(model, over, (x,))


# ------------------------------------------------------------------ derivatives wrt x with torch.func
def point_derivs(f, X, order=2):
    """f: (d,) -> (m,). Returns y (n, m), dy (n, m, d) and, if order == 2, d2y (n, m, d, d) (else None) at the rows of X (n, d)."""
    def fa(x):
        y = f(x)
        return y, y

    if order == 1:
        dy, y = vmap(jacrev(fa, has_aux=True))(X)
        return y, dy, None

    def g(x):
        dy, y = jacrev(fa, has_aux=True)(x)
        return dy, (dy, y)

    d2, (dy, y) = vmap(jacrev(g, has_aux=True))(X)
    return y, dy, d2


def _has(X, key):
    return key in X and X[key].shape[0] > 0


# ------------------------------------------------------------------ adapters
def scalar_pde_terms(prob, model_fn, X):
    """Burgers / Allen-Cahn (one output, hard transform, one residual term of weight 1): the same residual as gram_preconditioning.make_residual."""
    xt = X["residual"] if isinstance(X, dict) else X
    f = lambda x: prob.transform(x.unsqueeze(0), model_fn(x.unsqueeze(0)))[0]
    u, g, d2 = point_derivs(f, xt, 2)
    u, ux, ut, uxx = u[:, 0], g[:, 0, 0], g[:, 0, 1], d2[:, 0, 0, 0]
    if prob.name == "burgers_pinnacle_1d":
        r = ut + u * ux - prob.nu * uxx
    elif prob.name == "allen_cahn":
        r = ut - 0.001 * uxx - 5.0 * (u - u ** 3)
    else:
        raise ValueError(prob.name)
    return [(r, 1.0, "pde")]


def ns2d_cg_terms(prob, model_fn, X, nu=None):
    """PINNacle NS2d-CG (NSProblem.loss): 8 terms, weight 1, full batch. X = {'domain','inlet','outlet','other'}."""
    nu = getattr(prob, "nu", NS_NU) if nu is None else nu
    terms = []
    if _has(X, "domain"):
        y, dy, d2 = point_derivs(lambda x: model_fn(x.unsqueeze(0))[0], X["domain"], 2)
        u, v = y[:, 0], y[:, 1]
        ux, uy, vx, vy, px, py = dy[:, 0, 0], dy[:, 0, 1], dy[:, 1, 0], dy[:, 1, 1], dy[:, 2, 0], dy[:, 2, 1]
        lap_u, lap_v = d2[:, 0, 0, 0] + d2[:, 0, 1, 1], d2[:, 1, 0, 0] + d2[:, 1, 1, 1]
        terms += [(u * ux + v * uy + px - nu * lap_u, 1.0, "momentum_x"),
                  (u * vx + v * vy + py - nu * lap_v, 1.0, "momentum_y"),
                  (ux + vy, 1.0, "continuity")]
    if _has(X, "inlet"):
        oi, yi = model_fn(X["inlet"]), X["inlet"][:, 1]
        terms += [(oi[:, 0] - 4.0 * yi * (1.0 - yi), 1.0, "inlet_u"), (oi[:, 1], 1.0, "inlet_v")]
    if _has(X, "outlet"):
        terms += [(model_fn(X["outlet"])[:, 2], 1.0, "outlet_p")]
    if _has(X, "other"):
        ot = model_fn(X["other"])
        terms += [(ot[:, 0], 1.0, "other_u"), (ot[:, 1], 1.0, "other_v")]
    return terms


def dfg_terms(model_fn, X, nu=DFG_NU, u_mean=DFG_U_MEAN, u_max=DFG_U_MAX, diameter=DFG_DIAMETER):
    """DFG 2D-1 Re=20 as in sciml/dfg.py train_dfg: dimensionless momentum/continuity, velocity BCs normalized by U_MEAN, do-nothing outlet.
    The loss there sums per-term means; a (n, 2) velocity mean is half the sum of the two component means, hence weight 1/2. X = {'interior',
    'inlet','lower_wall','upper_wall','cylinder','outlet'} (full pools or a minibatch). Output scales live inside the model's forward."""
    p_scale = u_mean ** 2
    m_scale, c_scale = u_mean ** 2 / diameter, u_mean / diameter
    terms = []
    if _has(X, "interior"):
        y, dy, d2 = point_derivs(lambda x: model_fn(x.unsqueeze(0))[0], X["interior"], 2)
        u, v = y[:, 0], y[:, 1]
        ux, uy, vx, vy, px, py = dy[:, 0, 0], dy[:, 0, 1], dy[:, 1, 0], dy[:, 1, 1], dy[:, 2, 0], dy[:, 2, 1]
        lap_u, lap_v = d2[:, 0, 0, 0] + d2[:, 0, 1, 1], d2[:, 1, 0, 0] + d2[:, 1, 1, 1]
        terms += [((u * ux + v * uy + px - nu * lap_u) / m_scale, 1.0, "momentum_x"),
                  ((u * vx + v * vy + py - nu * lap_v) / m_scale, 1.0, "momentum_y"),
                  ((ux + vy) / c_scale, 1.0, "continuity")]
    if _has(X, "inlet"):
        yi = X["inlet"][:, 1]
        o = model_fn(X["inlet"])
        target = 4.0 * u_max * yi * (DFG_Y[1] - yi) / (DFG_Y[1] - DFG_Y[0]) ** 2
        terms += [((o[:, 0] - target) / u_mean, 0.5, "inlet_u"), (o[:, 1] / u_mean, 0.5, "inlet_v")]
    for name in ("lower_wall", "upper_wall", "cylinder"):
        if _has(X, name):
            o = model_fn(X[name])
            terms += [(o[:, 0] / u_mean, 0.5, f"{name}_u"), (o[:, 1] / u_mean, 0.5, f"{name}_v")]
    if _has(X, "outlet"):
        y, dy, _ = point_derivs(lambda x: model_fn(x.unsqueeze(0))[0], X["outlet"], 1)
        terms += [((nu * dy[:, 0, 0] - y[:, 2]) / p_scale, 1.0, "outlet_sx"), (nu * dy[:, 1, 0] / p_scale, 1.0, "outlet_sy")]
    return terms


class DFGTerms:
    """Stand-in `prob` for gn_blocks / GramPreconditioner.refresh with the DFG model (dfg.py has no Problem object). Pool = dict of float32 tensors
    (sciml.dfg.generate_dfg_pools). The GN uses a deterministic subset of each key of the pool (gram_preconditioning.subset), not the minibatch."""
    name = "dfg_2d1_re20"

    def residual_terms(self, model_fn, X):
        return dfg_terms(model_fn, X)


def piezo_residual_terms(prob, model_fn, X):
    """Bimorfo PVDF 2D del paper (Gonzalez Carrillo et al., MCA 31, 214), problemas CONVERSO y DIRECTO (`PiezoProblem`, experiments/mechanisms/EXP-0028-failure-attribution/piezo/piezo_problem.py).
    model_fn(x) es el MODELO COMPLETO (tronco + construccion por fase + escalas + bordes duros) en metros: x (n, 2) -> (n, 8) = (u, v, phi, sx, sz, tau, Dx, Dy). `prob.spec` trae todo lo
    que el adaptador necesita (material, escalas, divisores constitutivos, pesos, enrutamiento): este modulo no importa el problema. X = {'interior','top','bottom','right','left'}
    (las llaves vacias se saltan; 'interface' no entra: L_int es identicamente 0 en el modelo mixto y no aporta ni valor ni gradiente).
    Terminos (peso = el de la perdida): 8 de interior (3 constitutivas mecanicas, 2 electricas, 2 equilibrios, Gauss; peso lambda_PDE) y los de borde (peso lambda_BC):
      converso (16): top/bottom (sz, tau), right (sx, tau), Dn en right y left;  directo (17): top/bottom (sz, tau), right (sx; tau - t_y), Dn en right y left, y la carga neta cero del
      electrodo superior. Con electrodo flotante duro el termino de equipotencial es identicamente 0 y se omite.
    Enrutamiento (converso): en las tres constitutivas mecanicas la tension independiente va con .detach(): el J del Gauss-Newton es el del gradiente que ve el optimizador
    (la perdida y sus valores no cambian). Es una ELECCION declarada (spec['stopgrad']), no la derivada exacta del escalar.
    Carga neta cero: su perdida es (media de Dy/sD)^2, no una media de cuadrados; se entrega como un residuo constante r_i = media(Dy/sD) en los puntos dados: el VALOR es exacto
    cuando X trae todo el conjunto, y bajo troceo (gn_blocks_terms) el GN es el sustituto sum_c J_c^T J_c / N (cota superior del exacto), declarado.
    """
    s = prob.spec
    m, sc, cs, w = s["mat"], s["scales"], s["cscales"], s["w"]
    direct = s["mode"] == "direct"
    terms = []
    if _has(X, "interior"):
        xi = X["interior"]
        y, dy, _ = point_derivs(lambda x: model_fn(x.unsqueeze(0))[0], xi, 1)
        yy = xi[:, 1]
        bot = yy < m["center"]
        e31 = torch.where(bot, torch.full_like(yy, m["e31_bottom"]), torch.full_like(yy, m["e31_top"]))
        e33 = torch.where(bot, torch.full_like(yy, m["e33_bottom"]), torch.full_like(yy, m["e33_top"]))
        exx, eyy, gam = dy[:, 0, 0], dy[:, 1, 1], dy[:, 0, 1] + dy[:, 1, 0]
        Ex, Ey = -dy[:, 2, 0], -dy[:, 2, 1]
        p_sx, p_sz, p_ta = y[:, 3], y[:, 4], y[:, 5]
        if s["stopgrad"]:
            p_sx, p_sz, p_ta = p_sx.detach(), p_sz.detach(), p_ta.detach()
        wp = w["pde"]
        terms += [((p_sx - (m["C11"] * exx + m["C12"] * eyy - e31 * Ey)) / cs["sxx"], wp * w["nw"], "constitutive_sigma_xx"),
                  ((p_sz - (m["C12"] * exx + m["C22"] * eyy - e33 * Ey)) / cs["syy"], wp * w["nw"], "constitutive_sigma_yy"),
                  ((p_ta - m["G"] * gam) / cs["tau"], wp, "constitutive_tau_xy"),
                  ((y[:, 6] - m["eps1"] * Ex) / cs["dx"], wp, "constitutive_D_x"),
                  ((y[:, 7] - (e31 * exx + e33 * eyy + m["eps2"] * Ey)) / cs["dy"], wp, "constitutive_D_y"),
                  ((dy[:, 3, 0] + dy[:, 5, 1]) / sc["equilibrium_x"], wp, "equilibrium_x"),
                  ((dy[:, 5, 0] + dy[:, 4, 1]) / sc["equilibrium_y"], wp, "equilibrium_y"),
                  ((dy[:, 6, 0] + dy[:, 7, 1]) / sc["gauss"], wp, "gauss")]
    wb = w["bc"]
    for key in ("top", "bottom"):
        if _has(X, key):
            o = model_fn(X[key])
            terms += [(o[:, 4] / sc["stress_y"], wb, f"bc_{key}_sigma_yy"), (o[:, 5] / sc["shear"], wb, f"bc_{key}_tau_xy")]
    if _has(X, "right"):
        o = model_fn(X["right"])
        if direct:
            terms += [(o[:, 3] / sc["stress_x"], wb, "bc_right_traction_x"), ((o[:, 5] - s["force"] / sc["height"]) / sc["traction"], wb, "bc_right_traction_y")]
        else:
            terms += [(o[:, 3] / sc["stress_x"], wb, "bc_right_sigma_xx"), (o[:, 5] / sc["shear"], wb, "bc_right_tau_xy")]
        terms += [(o[:, 6] / sc["electric_displacement"], wb, "bc_right_Dn")]
    if _has(X, "left"):
        terms += [(model_fn(X["left"])[:, 6] / sc["electric_displacement"], wb, "bc_left_Dn")]
    if direct and _has(X, "top"):
        d = model_fn(X["top"])[:, 7] / sc["electric_displacement"]
        terms += [(d.mean().expand(d.shape[0]), wb, "bc_top_zero_net_charge")]
    return terms


_ADAPTERS = {"ns2d_cg": ns2d_cg_terms, "burgers_pinnacle_1d": scalar_pde_terms, "allen_cahn": scalar_pde_terms,
             "piezo_bimorph_indirect_2d": piezo_residual_terms, "piezo_bimorph_direct_2d": piezo_residual_terms}


def install(prob):
    """Give `prob` a `residual_terms(model_fn, X)` method (an adapter by prob.name) unless it has one. Returns prob."""
    if not hasattr(prob, "residual_terms"):
        if prob.name not in _ADAPTERS:
            raise ValueError(f"no residual_terms adapter for {prob.name!r}")
        prob.residual_terms = functools.partial(_ADAPTERS[prob.name], prob)
    return prob


# ------------------------------------------------------------------ loss and Gauss-Newton blocks from the terms
def terms_loss(prob, model_fn, X):
    """sum_t w_t * mean(r_t^2) (equals prob.loss when the adapter is right)."""
    terms = prob.residual_terms(model_fn, X)
    return sum(w * r.square().mean() for r, w, _ in terms), {n: w * r.square().mean() for r, w, n in terms}


def _slice(X, lo, n):
    return {k: v[lo:lo + n] for k, v in X.items()} if isinstance(X, dict) else X[lo:lo + n]


def _length(X):
    return max(v.shape[0] for v in X.values()) if isinstance(X, dict) else X.shape[0]


def _part(j, structure):
    if structure == "edge":
        return [torch.einsum("noig,noih->oigh", a, a) for a in j]
    if structure == "input":
        return [torch.einsum("noig,npih->iogph", a, a).reshape(a.shape[2], a.shape[1] * a.shape[3], a.shape[1] * a.shape[3]) for a in j]
    if structure == "diag":
        return [a.square().sum(0) for a in j]
    raise ValueError(structure)


def gn_blocks_terms(prob, model, X, structure, coefs, chunk=128):
    """B = sum_t (w_t / N_t) J_t^T J_t per layer, float64, J_t wrt the effective coefficients `coefs` (tuple, detached). Shapes as gn_blocks:
    'edge' -> (O, I, G, G); 'input' -> (I, O*G, O*G); 'diag' -> (O, I, G). N_t = number of points of term t over all chunks."""
    acc, count, weight = {}, {}, {}
    chunk = min(chunk, MAX_CHUNK)
    for lo in range(0, _length(X), chunk):
        xs = _slice(X, lo, chunk)
        meta = {}

        def f(cc):
            terms = prob.residual_terms(make_model_fn(model, cc), xs)
            meta["names"], meta["w"] = [n for _, _, n in terms], [w for _, w, _ in terms]
            return tuple(r.reshape(-1) for r, _, _ in terms)

        J = jacrev(f)(tuple(coefs))                                  # J[t][layer]: (n_t, O, I, G)
        for t, (name, w) in enumerate(zip(meta["names"], meta["w"])):
            Jt = [a.detach().double() for a in J[t]]
            part = _part(Jt, structure)
            acc[name] = part if name not in acc else [a + p for a, p in zip(acc[name], part)]
            count[name] = count.get(name, 0) + Jt[0].shape[0]
            weight[name] = w
    out = None
    for name, parts in acc.items():
        scaled = [p * (weight[name] / count[name]) for p in parts]
        out = scaled if out is None else [a + s for a, s in zip(out, scaled)]
    return out
