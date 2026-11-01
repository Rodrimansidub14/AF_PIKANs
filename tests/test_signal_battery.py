"""Tier-0 tests for the training-visible signal battery (CPU, seconds)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.sciml import signal_battery as sb  # noqa: E402
from src.sciml.model import build_model  # noqa: E402


@pytest.fixture(scope="module")
def burgers_model():
    torch.manual_seed(0)
    return build_model("burgers", {"id": "rbf_fixed"}).double()


def test_dev_grid_shape_and_domain():
    grid = sb.dev_grid("burgers", nx=9, nt=5)
    assert grid.shape == (45, 2)
    assert grid[:, 0].min() == -1.0 and grid[:, 0].max() == 1.0
    assert grid[:, 1].min() == 0.0 and grid[:, 1].max() == 1.0


def test_dense_features_keys_finite_and_no_reference(burgers_model):
    grid = sb.dev_grid("burgers", nx=9, nt=5)
    feats = sb.dense_features(burgers_model, "burgers", grid, sensitivity_points=8)
    for key in ("u", "u_x", "u_t", "u_xx", "residual", "grad_residual_norm", "basis0_entropy", "saturation_l0", "sensitivity"):
        assert key in feats and len(feats[key]) == 45
    assert np.isfinite(feats["residual"]).all()
    assert np.isfinite(feats["grad_residual_norm"]).all()
    assert np.isfinite(feats["sensitivity"][~np.isnan(feats["sensitivity"])]).all()
    # no label-like keys: features must never contain reference errors
    assert not any("error" in k or "ref" in k for k in feats)


def test_residual_matches_finite_difference(burgers_model):
    # The battery's residual uses autograd; check it against central differences of u.
    p = np.array([[0.3, 0.4]])
    feats = sb.dense_features(burgers_model, "burgers", p, sensitivity_points=1)
    h = 1e-5
    f = lambda q: burgers_model(torch.tensor(q, dtype=torch.float64)).item()
    ut = (f(p + [0, h]) - f(p - [0, h])) / (2 * h)
    ux = (f(p + [h, 0]) - f(p - [h, 0])) / (2 * h)
    assert abs(feats["u_t"][0] - ut) < 1e-6 and abs(feats["u_x"][0] - ux) < 1e-6


def test_jacobian_spectrum_and_conflict(burgers_model):
    grid = sb.dev_grid("burgers", nx=9, nt=5)
    spec = sb.jacobian_spectrum(burgers_model, grid, count=16)
    assert 1.0 <= spec["jac/effective_rank"] <= 16.0 and spec["jac/cond"] >= 1.0
    params = list(burgers_model.parameters())
    a = lambda: sum((p**2).sum() for p in params)
    b = lambda: sum(p.sum() for p in params) ** 2
    out = sb.grad_conflict(burgers_model, {"pde": a, "ic": b})
    assert -1.0 <= out["grad/conflict_cos_ic"] <= 1.0


def test_basis_influence_restores_parameters(burgers_model):
    before = burgers_model.layers[0].coefficients.detach().clone()
    grid = sb.dev_grid("burgers", nx=9, nt=5)
    inf = sb.basis_influence(burgers_model, "burgers", grid, layer=0, max_points=20)
    assert inf["basis/influence_u_layer0"].shape == (11,)
    assert torch.equal(before, burgers_model.layers[0].coefficients.detach())


def test_mlp_has_no_basis_features_but_battery_runs():
    torch.manual_seed(1)
    model = build_model("burgers", {"id": "mlp_tanh"}).double()
    grid = sb.dev_grid("burgers", nx=9, nt=5)
    dense, scalars = sb.battery(model, "burgers", grid, grid_shape=(9, 5), sensitivity_points=4)
    assert "basis0_entropy" not in dense and "jac/effective_rank" in scalars


def test_physics_summary_burgers_antisymmetry_is_a_feature(burgers_model):
    out = sb.physics_summary(burgers_model, "burgers", nx=65, nt=17)
    assert 0.0 <= out["burgers/antisym_err"] <= 2.0
    assert "burgers/energy_balance_rel" in out
