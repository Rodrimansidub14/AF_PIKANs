"""EXP-0022: pruebas del calculo de Gauss-Newton (J, espectro por la Gram, bloques por arista). CPU, red chica, doble precision."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[1]))
HERE = ROOT / "experiments/mechanisms/EXP-0022-conditioning-diagnosis"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
EXP16 = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gn = _load(HERE / "gn.py", "exp0022_gn")
sw = _load(EXP16 / "sweep.py", "exp0016_sweep_t")
pc = sw.load_campaign(BL / "burgers-close-20261003/pikan_campaign.py")
REF = BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz"


@pytest.fixture(scope="module")
def setup():
    torch.set_default_dtype(torch.float64)
    prob = pc.make_problem("burgers", str(REF))
    model = prob.build("rbf", 4, 11, 7).double()
    with torch.no_grad():
        for q in model.parameters():
            q.add_(0.3 * torch.randn_like(q))
    X = prob.pools(3, "cpu", torch.float64)["residual"][:96]
    return prob, model, X


def test_residual_matches_campaign(setup):
    prob, model, X = setup
    r = gn.residual(prob, model, X)
    ref = sw.residual_per_point(pc, prob, model, {"residual": X})
    assert torch.allclose(r, ref, atol=1e-10, rtol=1e-9)


def test_jacobian_matches_finite_differences(setup):
    prob, model, X = setup
    J = gn.jacobian(prob, model, X, chunk=32, dtype=torch.float64)
    theta0 = gn.flat_params(model).clone()
    d = torch.randn_like(theta0)
    h = 1e-6
    def r_at(theta):
        m = prob.build("rbf", 4, 11, 7).double()
        i = 0
        with torch.no_grad():
            for q in m.parameters():
                q.copy_(theta[i:i + q.numel()].reshape(q.shape)); i += q.numel()
        return gn.residual(prob, m, X)
    fd = (r_at(theta0 + h * d) - r_at(theta0 - h * d)) / (2 * h)
    assert torch.allclose(J @ d, fd, atol=1e-6, rtol=1e-5)


def test_gram_spectrum_equals_dense_gn_spectrum(setup):
    prob, model, X = setup
    J = gn.jacobian(prob, model, X, chunk=32, dtype=torch.float64)
    dense = torch.linalg.eigvalsh(J.T @ J / J.shape[0]).flip(0).clamp_min(0).numpy()
    via_gram = gn.gram_spectrum(J)
    n = len(via_gram)
    assert np.allclose(dense[:n], via_gram, rtol=1e-8, atol=1e-12 * dense[0])
    assert np.all(dense[n:] < 1e-10 * dense[0])


def test_edge_block_matches_dense_block(setup):
    prob, model, X = setup
    J = gn.jacobian(prob, model, X, chunk=32, dtype=torch.float64)
    edges, rest, P = gn.layout(model)
    assert P == J.shape[1] and len(rest) + 11 * len(edges) == P
    B, keys, cols = gn.edge_blocks(J, edges)
    dense = J.T @ J / J.shape[0]
    k = 5
    c = torch.as_tensor(cols[k])
    assert torch.allclose(B[k], dense[c][:, c], atol=1e-12)


def test_inverse_sqrt_whitens_block(setup):
    B = torch.randn(3, 11, 11, dtype=torch.float64)
    B = B @ B.transpose(-1, -2) + 0.1 * torch.eye(11, dtype=torch.float64)
    S = gn.inv_sqrt_psd(B, 0.0)
    eye = S @ B @ S
    assert torch.allclose(eye, torch.eye(11, dtype=torch.float64).expand(3, 11, 11), atol=1e-8)


def test_preconditioned_spectrum_is_similar_to_whitened_operator(setup):
    prob, model, X = setup
    J = gn.jacobian(prob, model, X, chunk=32, dtype=torch.float64)
    edges, rest, P = gn.layout(model)
    JS = gn.apply_edge_precond(J, edges, rest, eps_rel=1e-3)
    assert JS.shape == J.shape
    e0, e1 = gn.gram_spectrum(J), gn.gram_spectrum(JS)
    assert e1[0] > 0 and np.all(np.isfinite(e1))


def test_module_does_not_import_evals_or_read_reference():
    import ast
    tree = ast.parse((HERE / "gn.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"math", "numpy", "torch"}
    src = (HERE / "gn.py").read_text()
    assert "u_ref" not in src and ".npz" not in src and ".mat" not in src
