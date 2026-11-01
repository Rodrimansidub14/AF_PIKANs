"""EXP-0023: pruebas previas 1-5 del precondicionamiento por Gram (CPU, red chica). Las pruebas 6 (presupuesto) y 7 (CPU/GPU) se hacen aparte."""
import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[1]))
MOD = ROOT / "src/mechanisms/gram_preconditioning.py"
GN = ROOT / "experiments/mechanisms/EXP-0022-conditioning-diagnosis/gn.py"
EXP16 = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
REF = BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gp = _load(MOD, "exp0023_gp")
gn = _load(GN, "exp0022_gn_for_0023")
sw = _load(EXP16 / "sweep.py", "exp0016_sweep_for_0023")
pc = sw.load_campaign(BL / "burgers-close-20261003/pikan_campaign.py")
torch.set_num_threads(2)


def _setup(dtype=torch.float64, width=4, seed=7, n_col=128, perturb=0.3):
    prob = pc.make_problem("burgers", str(REF))
    prob.n_col = n_col
    model = prob.build("rbf", width, 11, seed).to(dtype)
    if perturb:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for q in model.parameters():
                q.add_(perturb * torch.randn(q.shape, generator=g, dtype=q.dtype))
    P = prob.pools(3, torch.device("cpu"), dtype)
    return prob, model, P


def _eff(model):
    return [c.clone() for c in gp.effective_coefficients(model)]


def test_forward_matches_model_and_blocks_match_gn_edge_and_input():
    prob, model, P = _setup()
    X = P["residual"][:64]
    assert torch.allclose(gp._forward(model, gp.effective_coefficients(model), X), model(X), atol=1e-14)
    J = gn.jacobian(prob, model, X, chunk=32, dtype=torch.float64)
    edges, rest, _ = gn.layout(model)
    Bg, keys, cols = gn.edge_blocks(J, edges)                                   # (E, G, G)
    Be = gp.gn_blocks(prob, model, X, "edge", chunk=32)
    for l, B in enumerate(Be):
        O, I, G, _ = B.shape
        for o in range(O):
            for i in range(I):
                ref = Bg[keys.index((l, o, i))]
                assert float((B[o, i] - ref).abs().max()) < 1e-8
    Bi = gp.gn_blocks(prob, model, X, "input", chunk=32)
    for l, B in enumerate(Bi):
        for i in range(B.shape[0]):
            cidx = torch.as_tensor(__import__("numpy").concatenate([edges[(l, o, i)] for o in range(model.layers[l].coefficients.shape[0])]))
            ref = J[:, cidx].T @ J[:, cidx] / X.shape[0]
            assert float((B[i] - ref).abs().max()) < 1e-8
    Bd = gp.gn_blocks(prob, model, X, "diag", chunk=32)
    for l, d in enumerate(Bd):
        assert float((d - torch.diagonal(Be[l], dim1=-2, dim2=-1)).abs().max()) < 1e-12


def test_blocks_of_wrapped_model_equal_blocks_of_plain_model_with_same_effective_coefficients():
    prob, model, P = _setup()
    X = P["residual"][:48]
    ref = gp.gn_blocks(prob, model, X, "edge", chunk=48)
    pre = gp.GramPreconditioner("E", n_points=48)
    pre.attach(model)
    pre.refresh(model, prob, P["residual"][:48])                                 # P != I now
    assert not torch.equal(pre._layers[1].pre_P, torch.eye(11, dtype=torch.float64).expand_as(pre._layers[1].pre_P))
    got = gp.gn_blocks(prob, model, X, "edge", chunk=48)
    for a, b in zip(ref, got):
        assert float((a - b).abs().max()) < 1e-9 * float(a.abs().max())


def test1_identity_P_reproduces_baseline_bit_for_bit():
    prob, mb, P = _setup(torch.float32, perturb=0.0)
    _, mi, _ = _setup(torch.float32, perturb=0.0)
    gp.GramPreconditioner("ID").attach(mi)
    assert all(torch.equal(a, b) for a, b in zip(_eff(mb), _eff(mi)))
    sw.adam_stage(prob, mb, P, 25)
    sw.adam_stage(prob, mi, P, 25)
    assert all(torch.equal(a, b) for a, b in zip(_eff(mb), _eff(mi)))
    sw.lbfgs_stage(prob, mb, P, 1e12, max_iter=15, chunk=50, snapshot_every=1e12)
    sw.lbfgs_stage(prob, mi, P, 1e12, max_iter=15, chunk=50, snapshot_every=1e12)
    assert all(torch.equal(a, b) for a, b in zip(_eff(mb), _eff(mi)))
    assert torch.equal(mb.layers[0].base, mi.layers[0].base)


@pytest.mark.parametrize("arm", ["J", "E", "I", "F"])
def test3_gradient_wrt_cprime_is_Pt_times_gradient_wrt_c(arm):
    prob, model, P = _setup()
    pre = gp.GramPreconditioner(arm, n_points=64)
    pre.attach(model)
    if arm != "F":
        pre.refresh(model, prob, P["residual"])
    for layer in pre._layers:                                                    # P generic (not symmetric) to make the check strict
        with torch.no_grad():
            layer.pre_P.add_(0.05 * torch.randn_like(layer.pre_P))
    prob2, plain, _ = _setup()
    with torch.no_grad():
        for layer, c in zip(plain.layers, _eff(model)):
            layer.coefficients.copy_(c)
    prob.loss(model, P)[0].backward()
    prob2.loss(plain, P)[0].backward()
    for ml, orig, layer in zip(pre._layers, pre.originals(model), plain.layers):
        lay = ml.pre_layout
        gc = lay.to_groups(layer.coefficients.grad)
        expected = lay.from_groups(torch.einsum("kba,kb->ka", ml.pre_P, gc))    # P^T g
        assert torch.allclose(orig.grad, expected, atol=1e-12, rtol=1e-9)


def test4_F_never_recomputes_and_E_I_J_recompute_at_the_fixed_steps():
    sched = {}
    for arm in ("J", "E", "I", "F"):
        prob, model, P = _setup(torch.float32, n_col=64)
        pre = gp.GramPreconditioner(arm, every=5, n_points=32, chunk=32)
        pre.attach(model)
        P0 = [l.pre_P.clone() for l in pre._layers]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        at = []
        for step in range(1, 13):
            if pre.due(step - 1):
                before = _eff(model)
                pre.refresh(model, prob, P["residual"], opt)
                at.append(step - 1)
                assert all(len(opt.state[p]) == 0 for p in pre.originals(model))              # optimizer state cleared
                assert all(float((a - b).abs().max()) < 1e-5 * float(a.abs().max()) for a, b in zip(before, _eff(model)))   # function preserved
            opt.zero_grad(set_to_none=True)
            prob.loss(model, P)[0].backward()
            opt.step()
        if pre.due_lbfgs():
            pre.refresh(model, prob, P["residual"])
            at.append("lbfgs")
        sched[arm] = (at, pre.n_recomputes, all(torch.equal(a, l.pre_P) for a, l in zip(P0, pre._layers)))
    for arm in ("J", "E", "I"):
        assert sched[arm][0] == [0, 5, 10, "lbfgs"] and sched[arm][1] == 4 and not sched[arm][2]
    assert sched["F"] == ([], 0, True)
    with pytest.raises(RuntimeError):
        gp.GramPreconditioner("F").refresh(model, prob, P["residual"])
    # F's fixed P is the (regularized) uniform-input Gram, not the identity
    prob, model, _ = _setup(torch.float32)
    pref = gp.GramPreconditioner("F")
    pref.attach(model)
    B = gp.uniform_gram(model.layers[1])
    Bn, en = gp._normalize(B[None], gp.EPS_REL)
    Pexp = gp._inv_sqrt_and_sqrt(Bn, en)[0][0]
    assert torch.allclose(pref._layers[1].pre_P[0].double(), Pexp, atol=1e-4, rtol=1e-4)


def test5_isolation_no_evals_no_reference():
    tree = ast.parse(MOD.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    assert "evals" not in mods
    assert mods <= {"math", "time", "torch"}
    argnames = {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) for a in n.args.args + n.args.kwonlyargs}
    assert not (argnames & {"ref", "reference", "u_ref", "score"})
    assert "score" not in {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
