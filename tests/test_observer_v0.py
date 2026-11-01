"""EXP-0025 (BORRADOR): pruebas Tier 0 del observador (CPU, red chica, Burgers). Numeración = la del spec (`EXP-0025-observer-v0/spec.md`, «Tier 0»)."""
import ast
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[1]))
MOD = ROOT / "src/mechanisms/observer_v0.py"
GRAM = ROOT / "src/mechanisms/gram_preconditioning.py"
EXP16 = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff"
BUNDLE = ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning/kaggle/dataset"
REF = BUNDLE / "burgers_coleHopf_1025x129.npz"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ob = _load(MOD, "exp0025_observer")
gp = _load(GRAM, "exp0025_gp")
sw = _load(EXP16 / "sweep.py", "exp0016_sweep_for_0025")
pc = sw.load_campaign(BUNDLE / "pikan_campaign.py")
torch.set_num_threads(2)

CFG = ob.ObserverConfig(K=10, n_p=3, cells_per_axis=4, delta=0.05, gamma=0.25)


def _setup(seed=7, width=4, n_col=128, dtype=torch.float64, perturb=0.3):
    prob = pc.make_problem("burgers", str(REF))
    prob.n_col = n_col
    model = prob.build("rbf", width, 11, seed).to(dtype)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for q in model.parameters():
            q.add_(perturb * torch.randn(q.shape, generator=g, dtype=q.dtype))
    P = prob.pools(3, torch.device("cpu"), dtype)
    return prob, model, P


def _observer(prob, model, mode="probe", seed=5, cfg=CFG, extra=(), levers=("aR", "aG")):
    lo, hi = prob.lo, prob.hi
    H = ob.heldout_points(lo, hi, 256, seed)
    L = []
    if "aR" in levers:
        L.append(ob.RadLever(lo, hi, n_candidates=256))
    if "aG" in levers:
        pre = gp.GramPreconditioner("E", n_points=64)
        pre.attach(model)
        L.append(ob.GramLever(pre, "E"))
    return ob.Observer(L + list(extra), lo, hi, H, seed, cfg, mode), H


# ---- 1. aislamiento
def test1_module_does_not_import_evals_or_touch_the_reference():
    tree = ast.parse(MOD.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module)
    assert not any(m.split(".")[0] == "evals" for m in mods)
    src = MOD.read_text(encoding="utf-8")
    assert "prob.u" not in src and ".score(" not in src and "load_reference" not in src       # la referencia solo puntúa fuera del mecanismo


# ---- 2. restauración exacta
def test2_rejecting_every_action_restores_theta_and_optimizer_state_bit_for_bit():
    prob, model, P = _setup()
    obs, _ = _observer(prob, model, cfg=ob.ObserverConfig(K=10, n_p=3, cells_per_axis=4, delta=1e9))     # delta imposible: siempre a0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs._train(prob, model, P, opt, 4)                                                                   # estado de Adam no trivial
    before = ob.take_snapshot(model, opt, P)
    chosen = obs.decide(0, 4, prob, model, P, opt)
    assert chosen == "a0"
    assert ob.states_equal(before, ob.take_snapshot(model, opt, P))
    assert obs.extra_steps == 3 * CFG.n_p                                                                # se sondearon a0, aR y aG


def test2_every_lever_applied_then_restored_is_exact():
    prob, model, P = _setup()
    obs, _ = _observer(prob, model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs._train(prob, model, P, opt, 4)
    snap = ob.take_snapshot(model, opt, P)
    for name, lever in obs.levers.items():
        lever.apply(prob, model, P, opt, np.random.default_rng(0))
        assert not ob.states_equal(snap, ob.take_snapshot(model, opt, P)), name                           # la palanca sí cambió algo
        ob.restore_snapshot(snap, model, opt, P)
        assert ob.states_equal(snap, ob.take_snapshot(model, opt, P)), name


# ---- 3. presupuesto de puntos constante
@pytest.mark.parametrize("mode", ["probe", "fixed:aR", "random"])
def test3_collocation_budget_is_constant_across_decisions(mode):
    prob, model, P = _setup()
    obs, _ = _observer(prob, model, mode=mode)
    n0, sizes = P["residual"].shape, []
    orig = obs.decide

    def spy(*a, **k):
        r = orig(*a, **k)
        sizes.append(tuple(a[4]["residual"].shape))
        return r
    obs.decide = spy
    obs.run_adam(prob, model, P, 4 * CFG.K)
    assert sizes and all(s == tuple(n0) for s in sizes)


# ---- 4. determinismo
def _run(mode, seed=5):
    prob, model, P = _setup()
    obs, _ = _observer(prob, model, mode=mode, seed=seed)
    out = obs.run_adam(prob, model, P, 3 * CFG.K)
    return [q.detach().clone() for q in model.parameters()], [d["chosen"] for d in out["decisions"]], out


@pytest.mark.parametrize("mode", ["probe", "random"])
def test4_same_seed_same_run(mode):
    a, ca, _ = _run(mode)
    b, cb, _ = _run(mode)
    assert ca == cb and all(torch.equal(x, y) for x, y in zip(a, b))


def test4_probe_and_real_application_use_the_same_batch_stream():
    prob, model, P = _setup()
    obs, _ = _observer(prob, model)
    lever = obs.levers["aR"]
    p1, p2 = dict(P), dict(P)
    lever.apply(prob, model, p1, None, obs._rng(3, "aR"))
    lever.apply(prob, model, p2, None, obs._rng(3, "aR"))
    assert torch.equal(p1["residual"], p2["residual"])
    p3 = dict(P)
    lever.apply(prob, model, p3, None, obs._rng(4, "aR"))
    assert not torch.equal(p1["residual"], p3["residual"])


# ---- 5. H independiente
def test5_heldout_points_share_nothing_with_training_pool_or_eval_grid():
    prob, model, P = _setup()
    H = ob.heldout_points(prob.lo, prob.hi, 8192, seed=5)
    assert H.shape == (8192, 2) and (H >= np.array(prob.lo)).all() and (H <= np.array(prob.hi)).all()
    train = {tuple(p) for p in P["residual"].numpy().round(12)}
    assert not any(tuple(p) in train for p in H.round(12))
    xx, tt = np.meshgrid(prob.x, prob.t, indexing="xy")
    grid = {tuple(p) for p in np.stack((xx.reshape(-1), tt.reshape(-1)), 1).round(12)}
    assert not any(tuple(p) in grid for p in H.round(12))
    other = ob.heldout_points(prob.lo, prob.hi, 8192, seed=6)
    assert not np.array_equal(H, other)


# ---- 6. sobrecosto (la medición de reloj va en measure_overhead.py; aquí, la cuenta exacta de pasos)
def test6_extra_steps_are_exactly_one_probe_per_action_per_decision():
    prob, model, P = _setup()
    obs, _ = _observer(prob, model)
    steps = 4 * CFG.K
    out = obs.run_adam(prob, model, P, steps)
    n_dec = len(out["decisions"])
    assert n_dec == 4 and out["extra_steps"] == n_dec * 3 * CFG.n_p
    assert out["extra_steps"] / steps == pytest.approx(3 * CFG.n_p / CFG.K)                              # 3 n_p / K, como dice el spec
    prob, model, P = _setup()
    obs2, _ = _observer(prob, model, mode="random")
    out2 = obs2.run_adam(prob, model, P, steps)
    assert out2["extra_steps"] == 0 and out2["probe_seconds"] == 0.0                                      # O- no sondea


# ---- 7. null con razón, nunca un número inventado
class _Boom(ob.Lever):
    name = "boom"

    def apply(self, prob, model, P, opt, rng):
        raise RuntimeError("no medible")


class _NaN(ob.Lever):
    name = "nan"

    def apply(self, prob, model, P, opt, rng):
        with torch.no_grad():
            next(model.parameters()).fill_(float("nan"))


def test7_unmeasurable_lever_returns_null_with_reason_and_is_not_chosen():
    prob, model, P = _setup()
    obs, _ = _observer(prob, model, extra=[_Boom(), _NaN()], levers=("aR",))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    snap = ob.take_snapshot(model, opt, P)
    obs.decide(0, 0, prob, model, P, opt)
    d = obs.decisions[0]
    assert d.improvement["boom"] is None and "no medible" in d.null_reason["boom"]
    assert d.improvement["nan"] is None and ("nonfinite" in d.null_reason["nan"].lower() or "no finito" in d.null_reason["nan"].lower())
    assert d.chosen in ("a0", "aR")
    assert all(torch.isfinite(q).all() for q in model.parameters())
    if d.chosen == "a0":
        assert ob.states_equal(snap, ob.take_snapshot(model, opt, P))


def test7_unmeasurable_baseline_makes_the_decision_inconclusive_and_keeps_a0():
    prob, model, P = _setup()
    obs, _ = _observer(prob, model, levers=("aR",))
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert obs.decide(0, 0, prob, model, P, opt) == "a0"
    assert "a0" in obs.decisions[0].null_reason


# ---- lógica de elección
class _Calm(ob.Lever):
    """Palanca que baja el paso de Adam: peor que a0 en el sondeo de residuo medio al inicio (control negativo)."""
    name = "calm"

    def apply(self, prob, model, P, opt, rng):
        for g in opt.param_groups:
            g["lr"] = 1e-9


class _Fast(ob.Lever):
    name = "fast"

    def apply(self, prob, model, P, opt, rng):
        for g in opt.param_groups:
            g["lr"] = 3e-2


def test_probe_does_not_choose_a_lever_that_gives_no_improvement():
    prob, model, P = _setup()
    obs, _ = _observer(prob, model, extra=[_Calm(), _Fast()], levers=())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs.decide(0, 0, prob, model, P, opt)
    d = obs.decisions[0]
    assert d.improvement["calm"] <= 1e-6                      # no mejora: nada que ganar con lr ~ 0
    assert d.chosen != "calm"


def test_inert_flag_and_modes_are_reported():
    obs = ob.Observer([], (-1, 0), (1, 1), ob.heldout_points((-1, 0), (1, 1), 64, 0), 0, CFG, "probe")
    obs.decisions = [ob.Decision(step=i, chosen="a0", mode="probe") for i in range(10)]
    assert obs.is_inert()
    obs.decisions[0].chosen = "aR"
    obs.decisions[1].chosen = "aR"
    assert not obs.is_inert()
    with pytest.raises(ValueError):
        ob.Observer([], (-1, 0), (1, 1), ob.heldout_points((-1, 0), (1, 1), 64, 0), 0, CFG, "inventado")


# ---- 5b. H_d fresco por decisión
def test5_fresh_blocks_are_disjoint_from_each_other_training_and_eval_grid():
    prob, model, P = _setup()
    fh = ob.FreshHeldout(prob.lo, prob.hi, 8192, seed=5)
    blocks = [fh.block(d) for d in range(4)]
    train = {tuple(p) for p in P["residual"].numpy().round(12)}
    xx, tt = np.meshgrid(prob.x, prob.t, indexing="xy")
    grid = {tuple(p) for p in np.stack((xx.reshape(-1), tt.reshape(-1)), 1).round(12)}
    seen = set()
    for B in blocks:
        assert B.shape == (8192, 2) and (B >= np.array(prob.lo)).all() and (B <= np.array(prob.hi)).all()
        pts = {tuple(p) for p in B.round(12)}
        assert not (pts & train) and not (pts & grid) and not (pts & seen)
        seen |= pts
    assert np.array_equal(fh.block(2), ob.FreshHeldout(prob.lo, prob.hi, 8192, seed=5).block(2))      # reproducible por (semilla, d)
    assert not np.array_equal(fh.block(2), ob.FreshHeldout(prob.lo, prob.hi, 8192, seed=6).block(2))
    assert not np.array_equal(fh.block(0), ob.heldout_points(prob.lo, prob.hi, 8192, seed=5))         # flujo distinto del H fijo


def test5_observer_draws_a_new_block_in_every_decision_and_records_it():
    prob, model, P = _setup()
    lo, hi = prob.lo, prob.hi
    fh = ob.FreshHeldout(lo, hi, 256, seed=5)
    obs = ob.Observer([ob.RadLever(lo, hi, n_candidates=256)], lo, hi, fh, 5, CFG, "probe")
    used = []
    orig = obs._set_H
    obs._set_H = lambda H: (used.append(np.array(H)), orig(H))[1]
    used.clear()
    out = obs.run_adam(prob, model, P, 3 * CFG.K)
    assert [d["h_block"] for d in out["decisions"]] == [0, 1, 2]
    assert len(used) == 3 and not np.array_equal(used[0], used[1]) and not np.array_equal(used[1], used[2])


# ---- 8. API privada de GramPreconditioner que usa la variante F
def test8_gram_private_api_used_by_variant_F_has_not_changed():
    import inspect
    assert hasattr(gp.GramPreconditioner, "_set") and hasattr(gp.GramPreconditioner, "_fixed_blocks")
    assert list(inspect.signature(gp.GramPreconditioner._set).parameters) == ["self", "model", "blocks", "optimizer"]
    assert list(inspect.signature(gp.GramPreconditioner._fixed_blocks).parameters) == ["self", "layer"]


def test8_variant_F_lever_runs_and_changes_the_model_gradient_geometry():
    prob, model, P = _setup()
    pre = gp.GramPreconditioner("ID", n_points=64)
    pre.attach(model)
    lever = ob.GramLever(pre, "F")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    snap = ob.take_snapshot(model, opt, P)
    assert lever.apply(prob, model, P, opt, np.random.default_rng(0)) == {"variant": "F"}
    assert not ob.states_equal(snap, ob.take_snapshot(model, opt, P))


# ---- O-m: acciones al azar con la frecuencia de O
def test_matched_mode_follows_given_frequencies_and_never_probes():
    prob, model, P = _setup()
    lo, hi = prob.lo, prob.hi
    obs = ob.Observer([ob.RadLever(lo, hi, n_candidates=64)], lo, hi, ob.heldout_points(lo, hi, 64, 0), 3, CFG, "matched", probs={"a0": 0.8, "aR": 0.2})
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ch = [obs.decide(k, k, prob, model, P, opt) for k in range(300)]
    assert obs.extra_steps == 0 and obs.probe_seconds == 0.0
    assert 0.70 < ch.count("a0") / 300 < 0.90
    obs2 = ob.Observer([ob.RadLever(lo, hi, n_candidates=64)], lo, hi, ob.heldout_points(lo, hi, 64, 0), 3, CFG, "matched", probs={"a0": 0.8, "aR": 0.2})
    assert ch[:20] == [obs2.decide(k, k, prob, _setup()[1], _setup()[2], torch.optim.Adam(model.parameters())) for k in range(20)]    # determinista por (semilla, k)


def test_matched_mode_rejects_bad_probs():
    for bad in (None, {"a0": 0.5}, {"a0": 0.5, "zz": 0.5}):
        with pytest.raises(ValueError):
            ob.Observer([], (-1, 0), (1, 1), ob.heldout_points((-1, 0), (1, 1), 64, 0), 0, CFG, "matched", probs=bad)
