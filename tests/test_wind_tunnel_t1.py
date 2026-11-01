"""EXP-0024 T1: identidad del control con sweep.adam_stage + lbfgs_stage, y reubicación en el modelo de la campaña."""
import importlib.util
import os
import sys
from pathlib import Path

import torch

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[1]))
D = ROOT / "experiments/mechanisms/EXP-0024-wind-tunnel-protocol"
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("t1_trainer_t", str(D / "t1_trainer.py"))
t1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t1)
torch.set_num_threads(2)


def _setup(seed=1):
    cfg = t1.PROBLEMS["ac"]
    prob = t1.pc.make_problem("ac", str(cfg["ref"]))
    model = prob.build("rbf", cfg["width"], 11, seed).to(dtype=torch.float32)
    P = prob.pools(seed, torch.device("cpu"), torch.float32)
    return prob, model, P


def test_control_identical_to_sweep_stages():
    prob, m1, P = _setup()
    _, m2, _ = _setup()
    A = t1.sw.adam_stage(prob, m1, P, 40, ckpt_steps=[40])
    B = t1.adam_with_hook(prob, m2, P, 40, regrid=None)
    for k, v in m1.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k]), k
    L1 = t1.sw.lbfgs_stage(prob, m1, P, 1e12, max_iter=20, chunk=50, snapshot_every=1e12)
    L2 = t1.sw.lbfgs_stage(prob, m2, P, 1e12, max_iter=20, chunk=50, snapshot_every=1e12)
    for k, v in L1["final_state"].items():
        assert torch.equal(v, L2["final_state"][k]), k


def test_regrid_on_campaign_model_changes_shapes_and_keeps_function():
    prob, m, P = _setup()
    t1.adam_with_hook(prob, m, P, 30, regrid=None)
    x = P["residual"][:512].detach()
    with torch.no_grad():
        before = m(x).clone()
    d = t1.rg.regrid_kan(m, P["residual"].detach(), mix=1.0)
    assert m.layers[1].centers.shape == (m.layers[1].coefficients.shape[1], 11)
    assert bool((torch.diff(m.layers[1].centers, dim=1) > 0).all())
    with torch.no_grad():
        after = m(x)
    assert float(torch.linalg.norm(after - before) / torch.linalg.norm(before)) < 5e-2
    assert d["l1_after"]["eff_rank"] >= d["l1_before"]["eff_rank"]
    assert m.layers[1].coefficients.dtype == torch.float32 and m.layers[1].centers.dtype == torch.float32


def test_training_path_never_touches_reference():
    import inspect
    src = inspect.getsource(t1.adam_with_hook)
    assert "score" not in src and "u_ref" not in src and ".u" not in src.replace("prob.loss", "")


def test_regrid_creates_every_tensor_on_the_device_of_the_activations():
    """Fallo real de EXP-0018 en T4: un linspace/eye creado en CPU rompe quantile y solve en GPU."""
    import inspect
    src = inspect.getsource(t1.rg.regrid_layer)
    for needle in ("torch.linspace(0, 1, G, dtype=z.dtype, device=z.device)", "torch.linspace(-1, 1, G, dtype=z.dtype, device=z.device)",
                   "torch.eye(G, dtype=z.dtype, device=z.device)"):
        assert needle in src, needle


def test_warmup_runs_and_leaves_the_following_run_unchanged():
    prob, m1, P = _setup()
    t1.adam_with_hook(prob, m1, P, 30, regrid=None)
    info = t1.warmup_t1("ac", "cpu", steps=4)
    prob, m2, P = _setup()
    t1.adam_with_hook(prob, m2, P, 30, regrid=None)
    assert info["n_params"] > 0
    for k, v in m1.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k]), k
