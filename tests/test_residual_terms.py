"""EXP-0028 fase 4: residuo vectorial con todos los términos de la pérdida (src/mechanisms/residual_terms.py) y su uso en gn_blocks (CPU, redes chicas).

No usa referencia ni evals: NSProblem se crea sin leer su .dat (solo se usan pools y loss), AC/Burgers solo para el residuo.
"""
import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[1]))
MOD = ROOT / "src/mechanisms/gram_preconditioning.py"
RT = ROOT / "src/mechanisms/residual_terms.py"
CAMPAIGN = ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning/kaggle/dataset/pikan_campaign.py"
torch.set_num_threads(2)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gp = _load(MOD, "exp0028_gp")                       # by path, as the campaign scripts do: exercises the file-based import of residual_terms
pc = _load(CAMPAIGN, "exp0028_campaign")


def _rt():
    return gp._residual_terms_module()


def _perturb(model, seed, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for q in model.parameters():
            q.add_(scale * torch.randn(q.shape, generator=g, dtype=q.dtype))


def _scalar_problem(kind, dtype=torch.float64, width=4, seed=7, n=48):
    cls = {"ac": pc.ACProblem, "burgers": pc.BurgersProblem}[kind]
    prob = cls.__new__(cls)                          # no reference file
    model = prob.build("rbf", width, 11, seed).to(dtype)
    _perturb(model, seed)
    P = prob.pools(3, torch.device("cpu"), dtype)
    return prob, model, P["residual"][:n]


def _ns(dtype=torch.float64, width=4, seed=7, n=40, perturb=0.3):
    prob = pc.NSProblem.__new__(pc.NSProblem)
    model = prob.build("rbf", width, 11, seed).to(dtype)
    if perturb:
        _perturb(model, seed, perturb)
    P = prob.pools(3, torch.device("cpu"), dtype)
    X = {k: v[:n] for k, v in P.items()}
    return prob, model, P, X


# ------------------------------------------------------------------ the vector path equals the scalar path in AC / Burgers
@pytest.mark.parametrize("kind", ["ac", "burgers"])
def test_vector_blocks_equal_scalar_blocks_in_ac_and_burgers(kind):
    prob, model, X = _scalar_problem(kind)
    assert not hasattr(prob, "residual_terms")                                   # (a) the scalar path is the one in use without the interface
    scalar = {s: gp.gn_blocks(prob, model, X, s, chunk=32) for s in ("edge", "input", "diag")}
    _rt().install(prob)
    assert hasattr(prob, "residual_terms")
    for s, ref in scalar.items():
        got = gp.gn_blocks(prob, model, X, s, chunk=20)                          # different chunking on purpose
        for a, b in zip(ref, got):
            assert a.shape == b.shape
            assert float((a - b).abs().max()) < 1e-10 * float(a.abs().max())


@pytest.mark.parametrize("kind", ["ac", "burgers"])
def test_terms_loss_equals_problem_loss_scalar(kind):
    prob, model, X = _scalar_problem(kind)
    _rt().install(prob)
    tot, _ = _rt().terms_loss(prob, model, X)
    ref, _ = prob.loss(model, {"residual": X})
    assert abs(float(tot) - float(ref)) < 1e-12 * abs(float(ref))


def test_burgers_ac_without_interface_never_touch_the_new_module():
    prob, model, X = _scalar_problem("burgers")
    sys.modules.pop("mechanisms_residual_terms", None)
    gp.gn_blocks(prob, model, X, "diag", chunk=48)
    assert "mechanisms_residual_terms" not in sys.modules


# ------------------------------------------------------------------ NS2d-CG, 3 outputs, 8 terms
def test_ns_terms_loss_matches_problem_loss_term_by_term():
    prob, model, P, X = _ns()
    _rt().install(prob)
    full = {k: v for k, v in P.items()}
    tot, per = _rt().terms_loss(prob, model, full)
    ref, ref_terms = prob.loss(model, P)
    assert set(per) == set(ref_terms) and len(per) == 8
    for k in per:
        assert abs(float(per[k]) - float(ref_terms[k])) < 1e-10 * max(1.0, float(ref_terms[k]))
    assert abs(float(tot) - float(ref)) < 1e-10 * float(ref)


def test_ns_diag_blocks_match_finite_differences_of_the_weighted_residual_vector():
    prob, model, P, X = _ns(n=12, width=3)
    _rt().install(prob)
    coefs = gp.effective_coefficients(model)
    got = gp.gn_blocks(prob, model, X, "diag", chunk=7)

    def weighted(cc):
        terms = prob.residual_terms(_rt().make_model_fn(model, cc), X)
        return torch.cat([(w / r.shape[0]) ** 0.5 * r.reshape(-1) for r, w, _ in terms])

    g = torch.Generator().manual_seed(1)
    h = 1e-6
    for layer in range(3):
        O, I, G = coefs[layer].shape
        for _ in range(4):
            o, i, k = (int(torch.randint(n, (1,), generator=g)) for n in (O, I, G))
            plus = [c.clone() for c in coefs]
            minus = [c.clone() for c in coefs]
            plus[layer][o, i, k] += h
            minus[layer][o, i, k] -= h
            dr = (weighted(tuple(plus)) - weighted(tuple(minus))) / (2 * h)
            assert abs(float(dr.square().sum()) - float(got[layer][o, i, k])) < 1e-6 * max(1e-12, float(got[layer][o, i, k])) + 1e-10


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_ns_blocks_are_finite_symmetric_psd_and_P_has_reasonable_singular_values(dtype):
    prob, model, P, X = _ns(dtype=dtype, width=4, n=64)
    _rt().install(prob)
    edge = gp.gn_blocks(prob, model, X, "edge", chunk=32)
    inp = gp.gn_blocks(prob, model, X, "input", chunk=32)
    dia = gp.gn_blocks(prob, model, X, "diag", chunk=32)
    for B in edge + inp + dia:
        assert B.dtype == torch.float64 and bool(torch.isfinite(B).all())
    for l, B in enumerate(edge):
        O, I, G, _ = B.shape
        assert float((B - B.transpose(-1, -2)).abs().max()) <= 1e-6 * float(B.abs().max())
        assert float((dia[l] - torch.diagonal(B, dim1=-2, dim2=-1)).abs().max()) < 1e-12 + 1e-5 * float(dia[l].abs().max())
        Bk = B.reshape(O * I, G, G)
        Bk = 0.5 * (Bk + Bk.transpose(-1, -2))
        assert float(torch.linalg.eigvalsh(Bk).min()) >= -1e-6 * float(Bk.abs().max())      # PSD up to float32 noise
        Bn, eps = gp._normalize(Bk, gp.EPS_REL)
        Pm = gp._inv_sqrt_and_sqrt(Bn, eps)[0]
        sv = torch.linalg.svdvals(Pm)
        assert bool(torch.isfinite(sv).all())
        assert float(sv.min()) >= 1.0 / (G + gp.EPS_REL) ** 0.5 - 1e-6                      # [1/sqrt(n+eps), 1/sqrt(eps)] ~ [0.3, 31.6] for n = 11
        assert float(sv.max()) <= 1.0 / gp.EPS_REL ** 0.5 + 1e-6
    for B in inp:
        assert float((B - B.transpose(-1, -2)).abs().max()) <= 1e-6 * float(B.abs().max())


def test_ns_blocks_of_preconditioned_model_equal_blocks_of_plain_model_and_refresh_preserves_function():
    prob, model, P, _ = _ns(dtype=torch.float32, width=4)
    _rt().install(prob)
    pre = gp.GramPreconditioner("E", n_points=48, chunk=48)
    X = gp.subset(P, 48)
    assert isinstance(X, dict) and X["domain"].shape[0] == 48
    ref = gp.gn_blocks(prob, model, X, "edge", chunk=48)
    pre.attach(model)
    before = [c.clone() for c in gp.effective_coefficients(model)]
    d = pre.refresh(model, prob, P)                                               # P != I afterwards
    assert d["n_points"] == sum(v.shape[0] for v in X.values()) and d["seconds"] > 0
    assert not torch.equal(pre._layers[1].pre_P, torch.eye(11).expand_as(pre._layers[1].pre_P))
    for a, b in zip(before, gp.effective_coefficients(model)):
        assert float((a - b).abs().max()) < 1e-5 * float(a.abs().max())            # the function does not change
    got = gp.gn_blocks(prob, model, X, "edge", chunk=48)                           # functional_call with pre_P := I
    for a, b in zip(ref, got):
        assert float((a - b).abs().max()) < 1e-4 * float(a.abs().max())
    for arm in ("J", "I"):
        _, m2, P2, _ = _ns(dtype=torch.float32, width=4)
        p2 = gp.GramPreconditioner(arm, n_points=32, chunk=32)
        p2.attach(m2)
        p2.refresh(m2, prob, P2)
        assert all(bool(torch.isfinite(l.pre_P).all()) for l in p2._layers)


# ------------------------------------------------------------------ DFG: real model (output_scales), terms equal train_dfg's loss
def test_dfg_terms_loss_equals_the_loss_of_train_dfg_and_blocks_are_finite():
    sys.path.insert(0, str(ROOT))
    from src.research_pipeline.sciml import dfg
    pools = dfg.generate_dfg_pools(5, interior_count=64, boundary_count=16)
    T = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in pools.items()}
    model = dfg.DFGRBFPIKAN(seed=3)
    _perturb(model, 3, 0.2)
    prob = _rt().DFGTerms()
    tot, per = _rt().terms_loss(prob, model, T)
    # the loss of train_dfg on the same (full) pools
    xy = T["interior"].clone().requires_grad_(True)
    res = dfg.steady_ns_residual(model(xy), xy)
    ref = sum(v.square().mean() for v in res.values())
    inlet_u = 4.0 * dfg.U_MAX * T["inlet"][:, 1:2] * (dfg.DOMAIN_Y[1] - T["inlet"][:, 1:2]) / (dfg.DOMAIN_Y[1] - dfg.DOMAIN_Y[0]) ** 2
    ref = ref + ((model(T["inlet"])[:, :2] - torch.cat((inlet_u, torch.zeros_like(inlet_u)), 1)) / dfg.U_MEAN).square().mean()
    for k in ("lower_wall", "upper_wall", "cylinder"):
        ref = ref + (model(T[k])[:, :2] / dfg.U_MEAN).square().mean()
    out = T["outlet"].clone().requires_grad_(True)
    f = model(out)
    ux, _, vx, _, _, _ = dfg._first_derivatives(f, out)
    ref = ref + ((dfg.NU * ux - f[:, 2:3]) / dfg.PRESSURE_SCALE).square().mean() + (dfg.NU * vx / dfg.PRESSURE_SCALE).square().mean()
    assert abs(float(tot) - float(ref)) < 1e-4 * float(ref)
    X = gp.subset(T, 24)
    B = gp.gn_blocks(prob, model, X, "edge", chunk=24)
    assert all(bool(torch.isfinite(b).all()) for b in B) and all(float(b.abs().max()) > 0 for b in B)
    assert hasattr(model, "output_scales")


def test_piezo_adapter_is_registered_and_needs_the_problem_spec():
    """El adaptador de piezo ya no es un contrato: lo prueba tests/test_piezo_problem.py (perdida, gradiente y bloques). Aqui: esta registrado y exige `prob.spec`."""
    rt = _rt()
    assert {"piezo_bimorph_indirect_2d", "piezo_bimorph_direct_2d"} <= set(rt._ADAPTERS)
    with pytest.raises(AttributeError):
        rt.piezo_residual_terms(object(), None, {})


# ------------------------------------------------------------------ isolation
def test_residual_terms_isolation_no_evals_no_reference():
    tree = ast.parse(RT.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    assert "evals" not in mods
    assert mods <= {"functools", "math", "torch"}
    argnames = {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) for a in n.args.args + n.args.kwonlyargs}
    assert not (argnames & {"ref", "reference", "u_ref", "score"})
    assert "score" not in {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
