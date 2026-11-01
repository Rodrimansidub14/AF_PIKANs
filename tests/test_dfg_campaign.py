"""Contract tests for DFG campaign evaluation helpers."""

import numpy as np
import pytest
import torch
import warnings

from src.sciml.dfg_campaign import (
    _boundary_signals,
    flatten_finite_metrics,
    summarize_residual_regions,
)
from src.sciml.dfg import DFGRBFPIKAN, generate_dfg_pools


def test_metric_flattening_emits_only_finite_numeric_contract_values():
    nested = {
        "qoi": {"drag_coefficient": 5.5, "lift_coefficient": 0.01},
        "heldout": {"global": {"momentum_x_rms": 0.25}},
        "field_relative_l2": None,
        "scientific_accepted": False,
        "diagnostic_name": "sobol",
    }

    flat = flatten_finite_metrics(nested)

    assert flat == {
        "qoi/drag_coefficient": 5.5,
        "qoi/lift_coefficient": 0.01,
        "heldout/global/momentum_x_rms": 0.25,
    }
    assert all(type(value) in (int, float) and np.isfinite(value) for value in flat.values())


def test_residual_summary_quantifies_global_near_cylinder_wake_and_far_field():
    points = np.array([[0.2, 0.2], [0.4, 0.2], [1.5, 0.2]], dtype=float)
    residuals = {
        "momentum_x": np.array([1.0, 2.0, 3.0]),
        "momentum_y": np.array([0.0, 0.0, 0.0]),
        "continuity": np.array([0.0, 0.0, 0.0]),
    }

    summary = summarize_residual_regions(points, residuals)

    assert summary["global"]["count"] == 3
    assert summary["near_cylinder"]["count"] == 1
    assert summary["wake"]["count"] == 1
    assert summary["far_field"]["count"] == 1
    assert summary["global"]["momentum_x_rms"] == pytest.approx(np.sqrt(14.0 / 3.0))
    assert summary["near_cylinder"]["momentum_x_p95_abs"] == pytest.approx(1.0)


def test_residual_summary_rejects_mismatched_or_nonfinite_arrays():
    points = np.array([[0.2, 0.2], [0.4, 0.2]], dtype=float)

    with pytest.raises(ValueError, match="same number"):
        summarize_residual_regions(points, {"momentum_x": np.array([1.0])})

    with pytest.raises(ValueError, match="finite"):
        summarize_residual_regions(points, {"momentum_x": np.array([1.0, np.nan])})


def test_boundary_signal_collection_detaches_values_without_autograd_warning():
    model = DFGRBFPIKAN(seed=9)
    pools = generate_dfg_pools(seed=9, interior_count=8, boundary_count=4)

    with warnings.catch_warnings(record=True) as warnings_seen:
        warnings.simplefilter("always")
        values = _boundary_signals(model, pools, torch.device("cpu"))

    assert len(values) == 6
    assert not warnings_seen
