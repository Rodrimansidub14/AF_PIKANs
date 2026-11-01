"""Parameter-matched neural and Kolmogorov-Arnold models for SciML screening."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RBFLayer(nn.Module):
    """Fixed Gaussian radial bases plus a trainable SiLU edge function."""

    def __init__(self, inputs: int, outputs: int, basis_count: int = 11):
        super().__init__()
        self.register_buffer("centers", torch.linspace(-1.05, 1.05, basis_count))
        self.register_buffer("width", torch.tensor(0.21))
        self.coefficients = nn.Parameter(torch.empty(outputs, inputs, basis_count))
        self.base = nn.Parameter(torch.empty(outputs, inputs))
        nn.init.normal_(self.coefficients, std=0.1 / (inputs**0.5))
        nn.init.normal_(self.base, std=0.1 / (inputs**0.5))

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width) ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("big,oig->bo", self.basis(x), self.coefficients) + F.linear(
            F.silu(x), self.base
        )


class BSplineLayer(nn.Module):
    """Cubic open-uniform B-spline edge functions with fixed knots."""

    def __init__(
        self,
        inputs: int,
        outputs: int,
        basis_count: int = 11,
        degree: int = 3,
        low: float = -1.05,
        high: float = 1.05,
    ):
        super().__init__()
        if basis_count <= degree:
            raise ValueError("basis_count must exceed spline degree")
        if not low < high:
            raise ValueError("spline interval must have positive width")
        interior_count = basis_count - degree - 1
        interior = (
            torch.linspace(low, high, interior_count + 2)[1:-1]
            if interior_count
            else torch.empty(0)
        )
        knots = torch.cat(
            (
                torch.full((degree + 1,), low),
                interior,
                torch.full((degree + 1,), high),
            )
        )
        self.register_buffer("knots", knots)
        self.degree = degree
        self.coefficients = nn.Parameter(torch.empty(outputs, inputs, basis_count))
        self.base = nn.Parameter(torch.empty(outputs, inputs))
        nn.init.normal_(self.coefficients, std=0.1 / (inputs**0.5))
        nn.init.normal_(self.base, std=0.1 / (inputs**0.5))

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the Cox-de Boor basis, retaining input autograd derivatives."""
        coordinate = x.unsqueeze(-1)
        basis = ((coordinate >= self.knots[:-1]) & (coordinate < self.knots[1:])).to(x.dtype)
        # Inputs in this experiment lie strictly inside the padded knot interval.
        # Keeping the endpoint convention makes the layer well-defined if the
        # interval is later used directly as a model domain.
        at_right_endpoint = coordinate.squeeze(-1) == self.knots[-1]
        if at_right_endpoint.any():
            endpoint = torch.zeros_like(basis)
            endpoint[..., -1] = at_right_endpoint.to(x.dtype)
            basis = torch.where(at_right_endpoint.unsqueeze(-1), endpoint, basis)

        for order in range(1, self.degree + 1):
            left_denominator = self.knots[order:-1] - self.knots[: -(order + 1)]
            right_denominator = self.knots[order + 1 :] - self.knots[1:-order]
            left_weight = (coordinate - self.knots[: -(order + 1)]) / left_denominator.clamp_min(
                torch.finfo(x.dtype).eps
            )
            right_weight = (self.knots[order + 1 :] - coordinate) / right_denominator.clamp_min(
                torch.finfo(x.dtype).eps
            )
            left_weight = torch.where(left_denominator > 0, left_weight, 0.0)
            right_weight = torch.where(right_denominator > 0, right_weight, 0.0)
            basis = left_weight * basis[..., :-1] + right_weight * basis[..., 1:]
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("big,oig->bo", self.basis(x), self.coefficients) + F.linear(
            F.silu(x), self.base
        )


class _KANBase(nn.Module):
    model_family: str

    def normalized(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(x, "tensor"):
            x = x.tensor
        x_space = 2.0 * x[:, 0:1] - 1.0 if self.problem == "heat" else x[:, 0:1]
        return torch.cat((x_space, 2.0 * x[:, 1:2] - 1.0), dim=1)

    def constrain(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        if self.problem == "heat":
            return torch.sin(torch.pi * x[:, 0:1]) + x[:, 1:2] * x[:, 0:1] * (1.0 - x[:, 0:1]) * raw
        if self.problem == "allen_cahn":
            space = x[:, 0:1]
            time = x[:, 1:2]
            initial = space.square() * torch.cos(torch.pi * space)
            return initial + time * (1.0 - space.square()) * raw
        return raw

    def forward_with_trace(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if hasattr(x, "tensor"):
            x = x.tensor
        h = self.normalized(x)
        traces: list[dict[str, torch.Tensor]] = []
        for index, layer in enumerate(self.layers):
            trace: dict[str, torch.Tensor] = {}
            if hasattr(layer, "basis"):
                trace["basis"] = layer.basis(h)
            preactivation = layer(h)
            trace["preactivation"] = preactivation
            h = torch.tanh(preactivation) if index < len(self.layers) - 1 else preactivation
            trace["activation"] = h
            traces.append(trace)
        return self.constrain(x, h), traces

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_trace(x)
        return output


class RBFPIKAN(_KANBase):
    """Fixed-center RBF KAN, retained as the existing model control."""

    model_id = "rbf_fixed"
    model_family = "rbf"

    def __init__(self, problem: str, basis_count: int = 11):
        super().__init__()
        if problem not in ("heat", "burgers", "allen_cahn"):
            raise ValueError("unknown problem")
        self.problem = problem
        self.layers = nn.ModuleList(
            [RBFLayer(a, b, basis_count) for a, b in zip([2, 16, 16], [16, 16, 1])]
        )


class BSplinePIKAN(_KANBase):
    """Fixed cubic B-spline KAN using the same edge and layer dimensions."""

    model_id = "bspline_cubic"
    model_family = "bspline"

    def __init__(self, problem: str, basis_count: int = 11, degree: int = 3):
        super().__init__()
        if problem not in ("heat", "burgers", "allen_cahn"):
            raise ValueError("unknown problem")
        self.problem = problem
        self.layers = nn.ModuleList(
            [
                BSplineLayer(a, b, basis_count, degree)
                for a, b in zip([2, 16, 16], [16, 16, 1])
            ]
        )


class PINNMLP(_KANBase):
    """Tanh PINN with a hidden width chosen to match the RBF parameter count."""

    model_id = "mlp_tanh"
    model_family = "mlp"

    def __init__(self, problem: str, width: int = 58):
        super().__init__()
        if problem not in ("heat", "burgers", "allen_cahn"):
            raise ValueError("unknown problem")
        self.problem = problem
        self.width = width
        self.layers = nn.ModuleList(
            [nn.Linear(2, width), nn.Linear(width, width), nn.Linear(width, 1)]
        )
        for layer in self.layers[:-1]:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward_with_trace(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if hasattr(x, "tensor"):
            x = x.tensor
        h = self.normalized(x)
        traces: list[dict[str, torch.Tensor]] = []
        for index, layer in enumerate(self.layers):
            preactivation = layer(h)
            h = torch.tanh(preactivation) if index < len(self.layers) - 1 else preactivation
            traces.append({"preactivation": preactivation, "activation": h})
        return self.constrain(x, h), traces


def build_model(problem: str, config: dict | None = None) -> nn.Module:
    """Create an explicitly named model; RBF remains the backwards-compatible default."""
    config = {} if config is None else dict(config)
    model_id = config.get("id", config.get("model_id", "rbf_fixed"))
    if model_id == "rbf_fixed":
        return RBFPIKAN(problem, basis_count=int(config.get("basis_count", 11)))
    if model_id == "bspline_cubic":
        return BSplinePIKAN(
            problem,
            basis_count=int(config.get("basis_count", 11)),
            degree=int(config.get("degree", 3)),
        )
    if model_id == "mlp_tanh":
        return PINNMLP(problem, width=int(config.get("width", 58)))
    raise ValueError(f"unknown model id: {model_id!r}")


def derivatives(y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    gradient = torch.autograd.grad(
        y, x, torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]
    ux = gradient[:, 0:1]
    ut = gradient[:, 1:2]
    uxx = torch.autograd.grad(
        ux, x, torch.ones_like(ux), create_graph=True, retain_graph=True
    )[0][:, 0:1]
    return ux, ut, uxx
