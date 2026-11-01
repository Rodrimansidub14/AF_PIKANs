import numpy as np
import torch

from src.sciml.model import build_model


def test_allen_cahn_hard_transform_enforces_initial_and_dirichlet_traces():
    torch.set_default_dtype(torch.float64)
    model = build_model("allen_cahn", {"id": "rbf_fixed"}).double()
    x0 = torch.linspace(-1.0, 1.0, 33)
    initial_points = torch.stack((x0, torch.zeros_like(x0)), dim=1)
    initial = model(initial_points).detach().numpy().reshape(-1)
    np.testing.assert_allclose(initial, x0.numpy() ** 2 * np.cos(np.pi * x0.numpy()), atol=1e-14)

    times = torch.linspace(0.0, 1.0, 17)
    for boundary_x in (-1.0, 1.0):
        boundary_points = torch.stack((torch.full_like(times, boundary_x), times), dim=1)
        boundary = model(boundary_points).detach().numpy().reshape(-1)
        np.testing.assert_allclose(boundary, -1.0, atol=1e-14)

