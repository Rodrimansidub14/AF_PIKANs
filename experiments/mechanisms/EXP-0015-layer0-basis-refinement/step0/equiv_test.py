"""Equivalencia de la capa 0 de fit_step0 (centros/ancho [in, G]) con RBFLayer del repo (centros [G], ancho escalar)."""
import importlib.util, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
torch.set_default_dtype(torch.float64)
from fit_step0 import RBFLayer as New
spec = importlib.util.spec_from_file_location("m", REPO / "src/research_pipeline/sciml/model.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for lo, hi, sig in [(-1.05, 1.05, 0.21), (-1.0, 1.0, 0.35)]:
    old = m.RBFLayer(2, 16, 11).double(); old.centers = torch.linspace(lo, hi, 11); old.width = torch.tensor(sig)
    c = np.linspace(lo, hi, 11); new = New(2, 16, np.stack([c, c]), np.full((2, 11), sig))
    new.coefficients.data.copy_(old.coefficients.data); new.base.data.copy_(old.base.data)
    x = (torch.rand(500, 2) * 2 - 1).requires_grad_(True)
    a, b = old(x), new(x)
    ga = torch.autograd.grad(a.sum(), x)[0]; gb = torch.autograd.grad(b.sum(), x)[0]
    dy, dg = (a - b).abs().max().item(), (ga - gb).abs().max().item()
    print(f"centros [{lo},{hi}] sigma {sig}: max|dy| {dy:.1e}  max|dgrad| {dg:.1e}")
    assert dy < 1e-12 and dg < 1e-12
print("ok")
