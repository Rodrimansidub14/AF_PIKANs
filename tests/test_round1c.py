"""Ronda 1-C (BORRADOR): pruebas de humo en CPU del arnes (problemas, referencias, instrumentacion y analisis). Sin GPU; <= 10 pasos de entrenamiento."""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[1]))
R1C = ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning/round1C"
EXP23 = R1C.parent
sys.path.insert(0, str(R1C))
sys.path.insert(0, str(EXP23))
import arnes_round1c as ar          # noqa: E402
import analyze_round1c as an        # noqa: E402

torch.set_num_threads(2)


@pytest.fixture()
def tmp_path():
    with tempfile.TemporaryDirectory(dir=str(EXP23 / "round1C")) as d:      # el tmp de pytest en Windows no es escribible aqui
        yield Path(d)


gp, pt, p1c = ar.gp, ar.pt, ar.p1c
PROV = json.loads((R1C / "provenance_round1c.json").read_text())


def _model(key, seed=3, dtype=torch.float64):
    prob = ar.make_problem_1c(key)
    return prob, prob.build("rbf", 16, 11, seed).to(dtype)


@pytest.mark.parametrize("fname", ["fisher_kpp_ref.npz", "allen_cahn_d2e-4_ref.npz"])
def test_reference_sha256_matches_provenance(fname):
    assert hashlib.sha256((R1C / fname).read_bytes()).hexdigest() == PROV["files_sha256"][fname]


@pytest.mark.parametrize("key", ["kpp", "ac_d2e-4"])
def test_reference_shape_and_score(key):
    prob, model = _model(key, dtype=torch.float32)
    assert prob.u.shape == (101, 201) and prob.x.shape == (201,) and prob.t.shape == (101,)
    s = prob.score(model, "cpu", torch.float32)
    assert np.isfinite(s["l2re_mat"]) and s["l2re_mat"] > 0
    assert next(iter(s.values())) == s["l2re_mat"]               # sweep.primary toma el primer valor


@pytest.mark.parametrize("key", ["kpp", "ac_d2e-4"])
def test_loss_residual_finite_and_hard_transform(key):
    prob, model = _model(key)
    P = prob.pools(2, torch.device("cpu"), torch.float64)
    loss, _ = prob.loss(model, P)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(q.grad).all() for q in model.parameters())
    x = torch.linspace(-1, 1, 41, dtype=torch.float64)
    raw = torch.randn(41, 1, dtype=torch.float64)
    u0 = prob.transform(torch.stack([x, torch.zeros_like(x)], 1), raw)[:, 0]          # t = 0 -> IC exacta
    ic = p1c.kpp_ic(x.numpy()) if key == "kpp" else p1c.ac_ic(x.numpy())
    assert np.allclose(u0.numpy(), ic, atol=1e-14)
    t = torch.linspace(0, 1, 21, dtype=torch.float64)
    raw = torch.randn(21, 1, dtype=torch.float64)
    left = prob.transform(torch.stack([-torch.ones_like(t), t], 1), raw)[:, 0].numpy()
    right = prob.transform(torch.stack([torch.ones_like(t), t], 1), raw)[:, 0].numpy()
    bl, br = (1.0, 0.0) if key == "kpp" else (-1.0, -1.0)
    assert np.abs(left - bl).max() < 1e-12 and np.abs(right - br).max() < 1e-12      # BC dura (KPP: <= 1.3e-14 por la cola de la onda)


def test_kpp_exact_solution_has_zero_residual_and_matches_reference():
    prob = ar.make_problem_1c("kpp")
    z = np.load(R1C / "fisher_kpp_ref.npz")
    assert float(np.abs(z["u"] - p1c.kpp_exact(z["x"][None, :], z["t"][:, None])).max()) < 1e-12
    xt = torch.rand(256, 2, dtype=torch.float64).requires_grad_(True)
    xt.data[:, 0] = 2 * xt.data[:, 0] - 1
    x, t = xt[:, 0:1], xt[:, 1:2]
    k, c = p1c.kpp_k(), p1c.kpp_c()
    u = torch.sigmoid(-k * (x - p1c.KPP_X0 - c * t)) ** 2
    g = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0:1], xt, torch.ones_like(u))[0][:, 0:1]
    assert float(prob.pde_residual(u, (g[:, 0:1], g[:, 1:2]), uxx).abs().max()) < 1e-8


def test_dose_problem_with_base_coefficient_equals_campaign_allen_cahn():
    base = pt.pc.make_problem("ac", str(pt.PROBLEMS["ac"]["ref"]))
    dose = ar.make_problem_1c("ac_d2e-4")
    dose.d = 1.0e-3
    model = base.build("rbf", 16, 11, 5).to(torch.float64)
    P = base.pools(4, torch.device("cpu"), torch.float64)
    assert float(abs(base.loss(model, P)[0] - dose.loss(model, P)[0])) < 1e-12
    X = gp.subset(P["residual"], 48)
    Bd = gp.gn_blocks(dose, model, X, "edge", 48)          # via el gancho pde_residual
    Bb = gp.gn_blocks(base, model, X, "edge", 48)          # via el nombre 'allen_cahn' (camino de 1-B, intacto)
    assert all(float((a - b).abs().max()) < 1e-10 * float(b.abs().max()) for a, b in zip(Bd, Bb))


def test_region_errors_definition():
    prob = ar.make_problem_1c("ac_d2e-4")
    r = ar.region_errors(prob.u, prob.u, prob.x)
    assert r["l2re_top"] == 0 and r["l2re_rest"] == 0
    assert abs(r["n_top"] / prob.u.size - 0.05) < 0.002 and r["n_top"] + r["n_rest"] == prob.u.size
    pred = prob.u + 0.01
    r = ar.region_errors(pred, prob.u, prob.x)
    assert 0 < r["share_sq_err_top"] < 1 and r["l2re_top"] > 0


def test_cond_probe_matches_E_arm_step0_diag_and_is_arm_independent():
    prob, model = _model("kpp", dtype=torch.float32)
    P = prob.pools(7, torch.device("cpu"), torch.float32)
    probe = ar.p_cond_probe(prob, model, P["residual"], n_points=64)
    pre = gp.GramPreconditioner("E", n_points=64)
    m2 = prob.build("rbf", 16, 11, 3).to(torch.float32)
    pre.attach(m2)
    diag = pre.refresh(m2, prob, P["residual"])
    assert all(abs(a - b) / b < 2e-3 for a, b in zip(probe, diag["p_cond"]))
    probe_after = ar.p_cond_probe(prob, m2, P["residual"], n_points=64)        # con el modelo ya precondicionado: la funcion es la misma
    assert all(abs(a - b) / b < 5e-2 for a, b in zip(probe, probe_after))


@pytest.mark.parametrize("key", ["kpp", "ac_d2e-4"])
def test_smoke_instrumented_run(key, tmp_path):
    for arm in ("B", "E", "F"):
        r = ar.run_one_1c(key, arm, 6, 6, 2, tmp_path, pre_kw=dict(every=3, n_points=32))
        assert r["adam_status"] == "adam_complete" and r["finite_params"] and np.isfinite(r["primary_final"])
        assert len(r["cond_init"]) == 3 and len(r["cond_final"]) == 3
        assert {"top", "rest"} <= {k[5:] for k in r["region_final"] if k.startswith("l2re_")}
        assert (tmp_path / f"{r['run_id']}.note.json").exists() and (tmp_path / f"{r['run_id']}-adam_end.pt").exists() and (tmp_path / f"{r['run_id']}-lbfgs_end.pt").exists()
    cs = {json.loads((tmp_path / f"{key}-{a}-seed6-a6-l2.json").read_text())["cond_init"][1] for a in ("B", "E", "F")}
    assert len(cs) == 1                                             # misma semilla -> mismo modelo inicial -> misma sonda
    res = an.analyze(an.load_rows([tmp_path]), key)
    assert res["E"]["n"] == 1 and res["F"]["n"] == 1


@pytest.mark.skipif(not (EXP23 / "results/t1-block/exp0023_t1block").exists(), reason="datos de 1-B no disponibles")
def test_analysis_script_reproduces_1B_burgers_diagnostic():
    rows = an.load_rows([EXP23 / "results/t1-block/exp0023_t1block", EXP23 / "results/v-block/exp0023_vblock"])
    res = an.analyze(rows, "burgers")
    assert res["E"]["n"] == 10 and res["F"]["n"] == 10
    assert abs(res["E"]["spearman_cond_init_vs_log_ratio"]["rho"] - (-0.72)) < 0.01        # diagnostic-burgers.md
    assert abs(res["E"]["median"] - 2.335) < 0.01 and res["E"]["improved"] == 5 and abs(res["E"]["max_ratio"] - 11.08) < 0.01
    assert abs(res["F"]["median"] - 0.490) < 0.005
    ac = an.analyze(rows, "ac")
    assert abs(ac["E"]["median"] - 0.138) < 0.01 and ac["E"]["improved"] == 10


def test_gpu_guard_requires_authorization(monkeypatch):
    monkeypatch.delenv("ROUND1C_GPU_AUTHORIZED", raising=False)
    monkeypatch.setattr(sys, "argv", ["x", "--problem", "kpp", "--device", "cuda:0"])
    with pytest.raises(SystemExit):
        ar.main()


def test_note_run_uses_wandb_offline_and_never_online(monkeypatch, tmp_path):
    calls = {}

    class _Run:
        def finish(self):
            calls["finished"] = True

    class _Fake:
        @staticmethod
        def init(**kw):
            calls["init"] = kw
            return _Run()

        @staticmethod
        def log(d):
            calls["log"] = d

    monkeypatch.setitem(sys.modules, "wandb", _Fake)
    monkeypatch.setenv("WANDB_MODE", "online")
    st = ar.note_run(tmp_path, "rid", dict(a=1), metrics=dict(x=1.5, bad=float("nan")))
    assert st == "wandb_offline_logged" and calls["init"]["mode"] == "offline" and calls["log"] == {"x": 1.5} and calls["finished"]
    assert json.loads((tmp_path / "rid.note.json").read_text())["wandb"] == "wandb_offline_logged"


def test_note_run_without_wandb_still_writes_note(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "wandb", None)            # import wandb -> ImportError
    assert ar.note_run(tmp_path, "rid2", dict(a=1)) == "wandb_not_installed"
    assert (tmp_path / "rid2.note.json").exists()
