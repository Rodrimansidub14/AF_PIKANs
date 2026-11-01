"""EXP-0029, brazo S (E + balance por grupo fisico P2 + guarda P4 + arranque denso P5). CPU, piezo chico.
(1) Con b=0 el bloque mezclado de S es el de E y su P es el de E. (2) El balance cambia el bloque con b>0 y deja simetria y PSD. (3) P4: la guarda dispara a >5x el minimo de la ventana,
respeta el enfriamiento, y damp() conserva la funcion (coeficientes efectivos) y mezcla P con I. (4) P5: due() incluye 250 y 500. (5) Con a_body=1 y sin guarda, S==E tras el mismo refresco (b=0)."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[1]))
PZ = ROOT / "experiments/mechanisms/EXP-0028-failure-attribution/piezo"
sys.path.insert(0, str(PZ))
sys.path.insert(1, str(ROOT / "src/mechanisms"))
sys.path.insert(2, str(ROOT))
import run_piezo as rp  # noqa: E402

gp = rp.gp
dt = torch.float64


def _setup(arm, mode="converse", b=None):
    prob = rp.make_problem(mode, 16)
    m = rp.build(prob, "B", 4001, dt)
    pre = gp.GramPreconditioner(arm, every=1000, n_points=32, chunk=4)
    if b is not None:
        pre.b_mix = b
    pre.attach(m)
    P = prob.pools(4001, torch.device("cpu"), dt, 0)
    return prob, m, pre, P


def test_due_schedule():
    pre = gp.GramPreconditioner("S")
    assert all(pre.due(e) for e in (0, 250, 500, 1000, 2000))
    assert not pre.due(300)
    assert not gp.GramPreconditioner("E").due(250)


def test_b0_equals_E():
    _, mE, pE, P = _setup("E")
    prob, mS, pS, _ = _setup("S", b=0.0)
    pE.refresh(mE, prob, P)
    pS.refresh(mS, prob, P)
    for lE, lS in zip(mE.layers, mS.layers):
        assert torch.allclose(lE.pre_P, lS.pre_P, rtol=1e-6, atol=1e-8)


def test_balance_changes_block_and_stays_psd():
    prob, m, pre, P = _setup("S", b=0.3)
    X = gp.subset(P, 32)
    rt = gp._residual_terms_module()
    B, Bbal = rt.gn_blocks_terms_groups(prob, m, X, gp.effective_coefficients(m), 4)
    for b, bb in zip(B, Bbal):
        assert torch.allclose(b, b.transpose(-1, -2), atol=1e-10 * float(b.abs().max()) + 1e-300)
        assert float(torch.linalg.eigvalsh(bb).min()) > -1e-8 * float(bb.abs().max()) - 1e-300
    assert any(not torch.allclose(b / b.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-300)[..., None, None],
                                  bb / bb.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-300)[..., None, None], rtol=1e-3) for b, bb in zip(B, Bbal))


def test_guard_and_damp_keep_function():
    prob, m, pre, P = _setup("S", b=0.3)
    pre.refresh(m, prob, P)
    x = prob.pools(1, torch.device("cpu"), dt, 0)["interior"][:20]
    f0 = m(x).detach().clone() if not isinstance(x, dict) else None
    for i in range(60):
        assert not pre.observe(i, 1.0)
    assert not pre.observe(60, 4.9)
    assert pre.observe(61, 6.0)
    assert not pre.observe(62, 7.0)                      # enfriamiento
    pre.damp(m)
    assert pre.a_body == 0.5
    for layer in m.layers:
        raw = pre._raw[layer]
        I = torch.eye(raw.shape[-1], dtype=raw.dtype).expand_as(raw)
        assert torch.allclose(layer.pre_P, 0.5 * I + 0.5 * raw, atol=1e-12)
    if f0 is not None:
        assert torch.allclose(m(x).detach(), f0, atol=1e-8)


def test_arm_M_chooses_b_within_cap_and_only_observes():
    prob, m, pre, P = _setup("M")
    assert pre.damp_enabled is False and pre.arm in gp.GRADED
    pre.refresh(m, prob, P)
    assert 0.0 <= pre.b_mix <= gp.B_CAP_W
    assert pre.b_mix in (0.0,) + gp.B_GRID
    assert pre.due(250) and pre.due(500)
    old = gp.COND_P90_MAX
    gp.COND_P90_MAX = 1.0000001                     # imposible de cumplir salvo b=0: M == E
    try:
        prob2, m2, pre2, P2 = _setup("M")
        pre2.refresh(m2, prob2, P2)
    finally:
        gp.COND_P90_MAX = old
    assert pre2.b_mix == 0.0
