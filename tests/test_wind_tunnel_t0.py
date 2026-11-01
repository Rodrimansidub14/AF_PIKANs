"""EXP-0024 T0: identidad del control con fit_step0.fit, propiedades del reubicador de centros y aislamiento."""
import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[1]))
D = ROOT / "experiments/mechanisms/EXP-0024-wind-tunnel-protocol"
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("t0_harness_t", str(D / "t0_harness.py"))
t0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t0)


@pytest.fixture(scope="module")
def P():
    torch.set_num_threads(2)
    return t0.fs.load_problem("allen_cahn")


def test_control_is_identical_to_step0_fit(P):
    a = t0.fs.fit(P, "C", 1, 60, 25, 800)
    b = t0.fit_t0(P, "C", 1, 60, 25, 800, regrid=None)
    assert a["final"]["u"] == b["final"]["u"] and a["adam"]["u"] == b["adam"]["u"]


def test_regrid_properties(P):
    torch.manual_seed(0)
    net = t0.fs.Net(P, *t0.fs.arm_layer0(P, "C"))
    s = 2 * (torch.as_tensor(P["pts"][:2000]) - net.lo) / (net.hi - net.lo) - 1
    with torch.no_grad():
        z1 = torch.tanh(net.l0(s))
        before = net.l1(z1).clone()
    d = t0.rg.regrid_net(net, s, mix=1.0)
    c = net.l1.centers
    assert bool((torch.diff(c, dim=1) > 0).all()) and bool((net.l1.width > 0).all())
    assert d["l1_fit"]["fit_rel_error"] < 1e-2 and d["l2_fit"]["fit_rel_error"] < 1e-2
    with torch.no_grad():
        after = net.l1(z1)
    assert float(torch.linalg.norm(after - before) / torch.linalg.norm(before)) < 5e-2


def test_regrid_module_uses_only_torch_and_no_reference():
    src = (D / "regrid.py").read_text()
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert mods <= {"torch"}
    assert "u_ref" not in src and "fields" not in src and ".npz" not in src


def test_regrid_mix_below_one_runs_and_keeps_centers_increasing(P):
    torch.manual_seed(0)
    net = t0.fs.Net(P, *t0.fs.arm_layer0(P, "C"))
    s = 2 * (torch.as_tensor(P["pts"][:1500]) - net.lo) / (net.hi - net.lo) - 1
    d = t0.rg.regrid_net(net, s, mix=0.5)
    assert bool((torch.diff(net.l1.centers, dim=1) > 0).all()) and bool((torch.diff(net.l2.centers, dim=1) > 0).all())
    assert d["l1_fit"]["fit_rel_error"] < 1e-2
