"""Focused contract tests for the isolated DFG 2D-1 SciML baseline."""

import math

import numpy as np
import pytest
import torch

from src.sciml.dfg import (
    DFGConfig,
    DFGRBFPIKAN,
    generate_dfg_pools,
    qoi_from_cylinder,
    resolve_device,
    steady_ns_residual,
    train_dfg,
)


def test_sobol_pools_are_reproducible_and_respect_dfg_geometry():
    first = generate_dfg_pools(seed=37, interior_count=128, boundary_count=32)
    second = generate_dfg_pools(seed=37, interior_count=128, boundary_count=32)

    assert set(first) == {"interior", "inlet", "lower_wall", "upper_wall", "cylinder", "outlet"}
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert first[name].shape == ((128 if name == "interior" else 32), 2)

    interior = first["interior"]
    assert np.all((interior[:, 0] > 0.0) & (interior[:, 0] < 2.2))
    assert np.all((interior[:, 1] > 0.0) & (interior[:, 1] < 0.41))
    radius = np.hypot(interior[:, 0] - 0.2, interior[:, 1] - 0.2)
    assert np.all(radius > 0.05)
    assert np.all(first["inlet"][:, 0] == 0.0)
    assert np.all(first["outlet"][:, 0] == 2.2)
    assert np.all(first["lower_wall"][:, 1] == 0.0)
    assert np.all(first["upper_wall"][:, 1] == 0.41)
    cylinder_radius = np.hypot(first["cylinder"][:, 0] - 0.2, first["cylinder"][:, 1] - 0.2)
    np.testing.assert_allclose(cylinder_radius, 0.05, atol=1e-14, rtol=0.0)


def test_rbf_pikan_has_fixed_centers_and_three_dfg_fields():
    model = DFGRBFPIKAN(seed=7).double()
    points = torch.tensor([[0.3, 0.2], [1.0, 0.1]], dtype=torch.float64)

    output = model(points)

    assert output.shape == (2, 3)
    assert [layer.in_features for layer in model.layers] == [2, 16, 16]
    assert [layer.out_features for layer in model.layers] == [16, 16, 3]
    assert all(not layer.centers.requires_grad for layer in model.layers)


def test_normalized_steady_ns_residual_is_zero_for_constant_manufactured_state():
    xy = torch.tensor([[0.3, 0.1], [1.0, 0.2], [2.0, 0.3]], dtype=torch.float64, requires_grad=True)
    fields = torch.cat(
        (0.2 + 0.0 * xy[:, 0:1], 0.0 * xy[:, 1:2], 4.0 + 0.0 * xy[:, 0:1]),
        dim=1,
    )

    residuals = steady_ns_residual(fields, xy)

    assert set(residuals) == {"momentum_x", "momentum_y", "continuity"}
    for value in residuals.values():
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-12, rtol=0.0)


def test_cuda_selection_never_silently_falls_back_to_cpu():
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")


def test_dfg_qois_follow_featflow_signs_and_are_pressure_gauge_invariant():
    count = 4096
    theta = 2.0 * math.pi * np.arange(count) / count
    x = 0.2 + 0.05 * np.cos(theta)
    y = 0.2 + 0.05 * np.sin(theta)
    zeros = np.zeros(count)

    # A pressure field p=-x gives positive drag under the FEATFLOW convention;
    # adding a constant gauge offset must leave integrated force and delta-p fixed.
    first = qoi_from_cylinder(theta, -x, zeros, zeros, zeros, zeros, -0.15, -0.25)
    shifted = qoi_from_cylinder(theta, -x + 17.0, zeros, zeros, zeros, zeros, 16.85, 16.75)
    expected_cd = 2.0 * math.pi * 0.05**2 / (0.2**2 * 0.1)

    assert first["drag_coefficient"] == pytest.approx(expected_cd, rel=1e-12)
    assert first["lift_coefficient"] == pytest.approx(0.0, abs=1e-12)
    assert first["pressure_difference"] == pytest.approx(0.1)
    assert shifted["drag_coefficient"] == pytest.approx(first["drag_coefficient"], abs=1e-12)
    assert shifted["lift_coefficient"] == pytest.approx(first["lift_coefficient"], abs=1e-12)
    assert shifted["pressure_difference"] == pytest.approx(first["pressure_difference"], abs=1e-12)

    # Positive lift tests the sign of the FEATFLOW pressure term in the y direction.
    lift = qoi_from_cylinder(theta, -y, zeros, zeros, zeros, zeros, -0.2, -0.2)
    assert lift["lift_coefficient"] == pytest.approx(expected_cd, rel=1e-12)


def test_dfg_viscous_force_uses_featflow_stress_traction_not_tangent_orientation():
    count = 4096
    theta = 2.0 * math.pi * np.arange(count) / count
    nx, ny = np.cos(theta), np.sin(theta)
    zeros = np.zeros(count)

    # A trace-free wall-normal velocity gradient c*t_clockwise with c=n_y
    # integrates to a positive drag under sigma*n, independent of tangent sign.
    du_dn, dv_dn = ny**2, -nx * ny
    du_dx, du_dy = du_dn * nx, du_dn * ny
    dv_dx, dv_dy = dv_dn * nx, dv_dn * ny
    result = qoi_from_cylinder(
        theta, zeros, du_dx, du_dy, dv_dx, dv_dy, 0.0, 0.0
    )
    expected_drag = 0.001 * math.pi * 0.05
    expected_cd = 2.0 * expected_drag / (0.2**2 * 0.1)

    assert result["drag_coefficient"] == pytest.approx(expected_cd, rel=1e-12)
    assert result["lift_coefficient"] == pytest.approx(0.0, abs=1e-12)


def test_training_budget_requires_a_finite_positive_cap_within_the_campaign_limit():
    with pytest.raises(ValueError, match="finite"):
        DFGConfig(time_limit_s=float("inf"))
    with pytest.raises(ValueError, match="positive"):
        DFGConfig(time_limit_s=0.0)
    with pytest.raises(ValueError, match="maximum"):
        DFGConfig(time_limit_s=1201.0)


def test_cpu_smoke_run_saves_exact_pools_checkpoint_and_nonaccepted_metrics(tmp_path):
    config = DFGConfig(
        seed=11,
        interior_count=8,
        boundary_count=4,
        batch_size=4,
        boundary_batch_size=4,
        max_steps=1,
        time_limit_s=120.0,
        device="cpu",
    )

    previous_default_dtype = torch.get_default_dtype()
    try:
        # Other SciML tests set the global default to float64; this runner must
        # remain internally dtype-consistent under that process-wide setting.
        torch.set_default_dtype(torch.float64)
        result = train_dfg(config, tmp_path)
    finally:
        torch.set_default_dtype(previous_default_dtype)

    assert result["run"]["device"] == "cpu"
    assert result["run"]["training_steps"] == 1
    assert result["run"]["scientific_accepted"] is False
    assert result["metrics"]["field_relative_l2"] is None
    for filename in (
        "dfg_training_pools.npz",
        "dfg_rbf_pikan.pt",
        "run.json",
        "metrics.json",
        "training_history.csv",
    ):
        assert (tmp_path / filename).is_file()
    import csv

    with (tmp_path / "training_history.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert int(rows[0]["step"]) == 1
    assert float(rows[0]["total_loss"]) > 0.0
    with np.load(tmp_path / "dfg_training_pools.npz") as archive:
        assert set(archive.files) == {"interior", "inlet", "lower_wall", "upper_wall", "cylinder", "outlet"}
        assert archive["interior"].shape == (8, 2)
