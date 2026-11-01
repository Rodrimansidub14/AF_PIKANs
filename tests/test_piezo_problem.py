"""EXP-0028, piezo: PiezoProblem (bimorfo PVDF, converso y directo del paper de Gonzalez Carrillo et al.) y su adaptador `piezo_residual_terms`. CPU, redes chicas.

(1) La perdida del PiezoProblem iguala paso a paso la del codigo de los autores (repo de referencia solo lectura en var/agent_runs/piezo-ref-20260930/daniel-ref-20261003; se
    omite si no esta) con las mismas semillas y puntos: modelo inicial, muestreo, perdida y primeros pasos de Adam (cruzando un remuestreo), en los dos problemas.
(2) Con los pesos entrenados del paper (results/runs/paper20k_*) el puntaje contra el FEM regenerado reproduce las metricas publicadas, y la historia de perdida del paper se
    reproduce desde la semilla 20260728.
(3) sum_t w_t mean(r_t^2) del adaptador == prob.loss, y su gradiente == el gradiente de prob.loss (incluye el enrutamiento stop-gradient del converso).
(4) Bloques B del Gauss-Newton finitos, simetricos y PSD con 8 salidas; el enrutamiento cambia el J como se declara.
(5) Aislamiento: ni la perdida ni los pools ni el adaptador leen la referencia FEM; sin imports de evals.
"""
import ast
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
PZ = ROOT / "experiments/mechanisms/EXP-0028-failure-attribution/piezo"
REF = ROOT / "var/agent_runs/piezo-ref-20260930/daniel-ref-20261003"
torch.set_num_threads(2)
sys.path.insert(0, str(PZ))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gp = _load(ROOT / "src/mechanisms/gram_preconditioning.py", "exp0028_gp")
import piezo_problem as pp  # noqa: E402

MODES = ["converse", "direct"]
needs_ref = pytest.mark.skipif(not (REF / "src").is_dir(), reason="repo de referencia (solo lectura) no disponible")


def _rt():
    return gp._residual_terms_module()


def _perturb(model, seed, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for q in model.parameters():
            q.add_(scale * torch.randn(q.shape, generator=g, dtype=q.dtype))


def _small(mode, width=4, seed=7, layer=24, perturb=0.3):
    prob = pp.PiezoProblem(mode, interior_per_layer=layer, boundary=16, interface=16)
    model = prob.build("rbf", width, 11, seed)
    if perturb:
        _perturb(model, seed, perturb)
    return prob, model, prob.pools(3)


# ------------------------------------------------------------------ el codigo de los autores (solo en pruebas)
@pytest.fixture(scope="module")
def gonzalez():
    os.environ["PINN_PIEZO_OUTPUTS_DIR"] = tempfile.mkdtemp()
    sys.path.insert(0, str(REF / "src"))
    torch.set_default_dtype(torch.float64)
    import pinn_piezo.indirect.model as im
    import pinn_piezo.indirect.losses as il
    import pinn_piezo.indirect.train as it
    import pinn_piezo.indirect.sampling as isamp
    import pinn_piezo.direct.model as dm
    import pinn_piezo.direct.losses as dl
    import pinn_piezo.direct.train as dt
    dl.APPLIED_FORCE_Y = -0.1
    yield dict(im=im, il=il, it=it, isamp=isamp, dm=dm, dl=dl, dt=dt)
    torch.set_default_dtype(torch.float32)


class _GRBFNet(torch.nn.Module):
    """Copia de daniel_arm.RBFNet (tronco RBF en `model.net`)."""

    def __init__(self, width):
        super().__init__()
        self.layers = torch.nn.ModuleList([pp.RBFEdge(4, width), pp.RBFEdge(width, width), pp.RBFEdge(width, 8)])

    def forward(self, z):
        h = torch.cat([2 * z[:, 0:1] - 1, 2 * z[:, 1:2] - 1, 4 * z[:, 2:3] - 1, z[:, 3:4]], dim=1)
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)


def _gonzalez_model(g, mode, seed, arm, width=16):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if mode == "converse":
        m = g["im"].build_default_model(device=None, model_type="pyramid", hidden_sizes=(50, 50, 50), phase_enriched=True, hard_natural_bcs=False,
                                        activation=torch.nn.SiLU, zero_output=False, split_trunks=False)
    else:
        m = g["dm"].build_default_model(device=None, reference_force=-0.1, hidden_sizes=(50, 50, 50), phase_enriched=True, hard_natural_bcs=True,
                                        hard_floating_electrode=True, slender_warping=True, split_trunks=False, output_init_gain=1.0, activation=torch.nn.SiLU)
    if arm == "rbf":
        m.net = _GRBFNet(width)
    return m.to(torch.float64)


def _gonzalez_run(g, mode, model, tensors, steps, resample_fn, resample_every):
    if mode == "converse":
        out = g["it"].run_adam(model, tensors, epochs=steps, lr=1e-3, resample_fn=resample_fn, resample_every=resample_every, mechanical_constitutive_stopgrad=True)
    else:
        out = g["dt"].run_adam(model, tensors, epochs=steps, lr=1e-3, loss_weights={"pde": 1.0, "bc": 10.0}, f=200, resample_fn=resample_fn, resample_every=resample_every,
                               electrical_mode="floating_electrode", constitutive_normalization="dominant", strict_transverse_constitutive=True)
    return out[0]


@needs_ref
@pytest.mark.parametrize("mode", MODES)
def test_pools_and_initial_models_equal_the_authors_code(gonzalez, mode):
    seed, n = 20260728, 64
    prob = pp.PiezoProblem(mode, interior_per_layer=n, boundary=16, interface=16)
    for arm in ("mlp", "rbf"):
        gm = _gonzalez_model(gonzalez, mode, seed, arm)
        mine = prob.build(arm, 16, 11, seed)
        gs, ms = gm.state_dict(), mine.state_dict()
        # nombres: el tronco RBF vive en `layers` (mio) y en `net.layers` (autores)
        gs = {k.replace("net.layers.", "layers."): v for k, v in gs.items()}
        assert set(gs) == set(ms), (set(gs) ^ set(ms))
        for k in gs:
            assert torch.equal(gs[k], ms[k]), (arm, k)
    for batch in (0, 1, 7):
        theirs = gonzalez["isamp"].sample_training_tensors(n_interior_per_layer=n, n_boundary=16, n_interface=16, seed=seed + batch, device="cpu", dtype=torch.float64)
        P = prob.pools(seed, "cpu", torch.float64, batch)
        assert torch.equal(P["interior"], torch.cat((theirs["x_collocation"].detach(), theirs["y_collocation"].detach()), dim=1))
        for k, kk in (("top", "xy_top"), ("bottom", "xy_bottom"), ("right", "xy_right"), ("left", "xy_left"), ("interface", "xy_interface")):
            assert torch.equal(P[k], theirs[kk]), k


@needs_ref
@pytest.mark.parametrize("mode,arm,steps", [("converse", "mlp", 6), ("direct", "mlp", 6), ("converse", "rbf", 3), ("direct", "rbf", 3)])
def test_loss_equals_the_authors_loss_step_by_step_across_a_resample(gonzalez, mode, arm, steps):
    seed, n, every = 20260728, 48, 3
    prob = pp.PiezoProblem(mode, interior_per_layer=n, boundary=16, interface=16)
    gm = _gonzalez_model(gonzalez, mode, seed, arm)
    sample = lambda k: gonzalez["isamp"].sample_training_tensors(n_interior_per_layer=n, n_boundary=16, n_interface=16, seed=seed + k, device="cpu", dtype=torch.float64)
    theirs = _gonzalez_run(gonzalez, mode, gm, sample(0), steps, sample, every)
    from run_piezo import adam_piezo
    mine = prob.build(arm, 16, 11, seed)
    trace = []
    adam_piezo(prob, mine, seed, steps, torch.device("cpu"), torch.float64, None, eval_every=10 ** 9, resample_every=every, trace=trace)
    assert len(theirs) == len(trace) == steps
    for a, b in zip(theirs, trace):
        assert abs(a - b) <= 1e-10 * abs(a), (a, b)
    for (k1, p1), (k2, p2) in zip(sorted(gm.state_dict().items()), sorted((k.replace("layers.", "net.layers.") if arm == "rbf" else k, v) for k, v in mine.state_dict().items())):
        assert k1 == k2
        assert float((p1 - p2).abs().max()) <= 1e-9 * max(1.0, float(p1.abs().max())), k1


# ------------------------------------------------------------------ pesos entrenados del paper y su historia de perdida
RUNS = REF / "results/runs"


@needs_ref
@pytest.mark.parametrize("mode,name,fname", [("converse", "indirect_mixed", "model_PINN_indirect.pt"), ("direct", "direct_mixed", "model_PINN_direct.pt")])
def test_paper_weights_reproduce_the_published_errors_and_loss_history(mode, name, fname):
    if not (RUNS / f"paper20k_{name}").is_dir():
        pytest.skip("sin resultados del paper")
    prob = pp.PiezoProblem(mode)
    m = prob.build("mlp", seed=1)
    m.load_state_dict(torch.load(RUNS / f"paper20k_{name}/models/{fname}", map_location="cpu"))
    sc = prob.score(m)
    pub = json.loads((RUNS / f"paper20k_{name}/metrics.json").read_text())["metrics"]
    for k, kk in (("l2re_u", "u"), ("l2re_v", "v"), ("l2re_phi", "phi")):
        assert abs(sc[k] - pub[kk]["rel_L2"]) <= 2e-3 * pub[kk]["rel_L2"], (k, sc[k], pub[kk]["rel_L2"])
    if mode == "direct":
        assert abs(sc["phi_top_mean"] - (-2.92)) < 0.01 and abs(sc["phi_top_mean_fem"] - (-3.10)) < 0.01
        assert abs(sc["tip_v"] * 1e6 - (-175.7)) < 0.3 and abs(sc["tip_v_fem"] * 1e6 - (-198.5)) < 0.1
    else:
        assert abs(sc["tip_v"] * 1e6 - (-29.3)) < 0.1 and abs(sc["tip_v_fem"] * 1e6 - (-32.9)) < 0.1
    # historia de perdida: semilla 20260728, lote 0, primeros pasos (los pares de la A100 del paper)
    ref = np.load(RUNS / f"paper20k_{name}/loss_{mode if mode == 'direct' else 'indirect'}.npy")
    prob2 = pp.PiezoProblem(mode)
    m2 = prob2.build("mlp", seed=pp.PAPER_SEED)
    P = prob2.pools(pp.PAPER_SEED)
    opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
    for e in range(4):
        opt.zero_grad()
        l, _ = prob2.loss(m2, P)
        l.backward()
        opt.step()
        assert abs(float(l) - ref[e]) <= 1e-8 * ref[e], (e, float(l), ref[e])


# ------------------------------------------------------------------ geometria, signos y bordes duros
def test_material_signs_scales_and_hard_boundary_conditions():
    m = pp.MAT
    assert m["e31_bottom"] > 0 > m["e33_bottom"] and m["e31_top"] == -m["e31_bottom"] and m["e33_top"] == -m["e33_bottom"]       # Tabla 2 del paper
    assert abs(m["e31_bottom"] - 2.904e-2) < 1e-5 and abs(m["e33_bottom"] + 5.157e-2) < 1e-5
    assert abs(m["C11"] - 2.1836e9) < 1e6 and abs(m["C12"] - 0.6332e9) < 1e6 and abs(m["eps1"] - 1.0625e-10) < 1e-13 and abs(m["eps2"] - 1.0406e-10) < 1e-13
    assert abs(pp.direct_scales(0.1)["stress_x"] - 6e4) < 1 and abs(pp.direct_scales(0.1)["traction"] - 100.0) < 1e-9
    for mode in MODES:
        prob = pp.PiezoProblem(mode)
        model = prob.build("rbf", 4, 11, 5)
        _perturb(model, 5, 0.5)
        y = torch.linspace(0, pp.HEIGHT, 7, dtype=torch.float64)
        left = model(torch.stack((torch.zeros_like(y), y), 1))
        bottom = model(torch.stack((torch.linspace(0, pp.WIDTH, 5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64)), 1))
        top = model(torch.stack((torch.linspace(0, pp.WIDTH, 5, dtype=torch.float64), torch.full((5,), pp.HEIGHT, dtype=torch.float64)), 1))
        assert float(left[:, 0:2].abs().max()) == 0.0 and float(bottom[:, 2].abs().max()) == 0.0         # empotramiento y electrodo a tierra
        if mode == "converse":
            assert float((top[:, 2] - 100.0).abs().max()) < 1e-12
        else:
            assert float((top[:, 2] - top[0, 2]).abs().max()) == 0.0                                      # electrodo flotante: potencial uniforme
            r = model(torch.tensor([[pp.WIDTH, 0.3e-3], [pp.WIDTH, 0.8e-3]], dtype=torch.float64))
            assert float((r[:, 5] + 100.0).abs().max()) < 1e-9 and float(r[:, 3].abs().max()) == 0.0        # t_y = Fy/(bH) = -100 Pa en el extremo
            assert float(model(torch.tensor([[0.0, 0.4e-3], [pp.WIDTH, 0.4e-3]], dtype=torch.float64))[:, 6].abs().max()) == 0.0     # aislamiento lateral


def test_phase_construction_gives_continuity_and_allows_the_two_jumps():
    prob, model, P = _small("converse")
    xy = P["interface"]
    assert prob.interface_gap(model, P) == 0.0                       # L_int identicamente 0 (ecs. 33-35)
    up = model.forward_trace(xy, torch.ones_like(xy[:, 0:1]))
    dn = model.forward_trace(xy, -torch.ones_like(xy[:, 0:1]))
    assert float((up[:, 3] - dn[:, 3]).abs().max()) > 0 and float((up[:, 6] - dn[:, 6]).abs().max()) > 0     # sx y Dx pueden saltar


# ------------------------------------------------------------------ adaptador: valor y gradiente de la perdida
@pytest.mark.parametrize("mode", MODES)
def test_terms_loss_equals_problem_loss_and_the_gradient_of_the_optimizer(mode):
    prob, model, P = _small(mode)
    _rt().install(prob)
    tot, per = _rt().terms_loss(prob, model, P)
    ref, ref_terms = prob.loss(model, P)
    assert abs(float(tot) - float(ref)) < 1e-10 * float(ref)
    for k, v in per.items():
        assert abs(float(v) - float(ref_terms[k])) < 1e-10 * max(1e-30, abs(float(ref_terms[k])), float(ref)) + 1e-30, k
    assert float(ref_terms["bc_top_equipotential"]) == 0.0 if mode == "direct" else True
    assert len(per) == (16 if mode == "converse" else 17)
    params = list(model.parameters())
    g1 = torch.autograd.grad(ref, params, allow_unused=True)
    g2 = torch.autograd.grad(tot, params, allow_unused=True)
    num = sum(float((a - b).square().sum()) for a, b in zip(g1, g2) if a is not None and b is not None)
    den = sum(float(a.square().sum()) for a in g1 if a is not None)
    assert (num / den) ** 0.5 < 1e-10                              # con enrutamiento (converso) el J del adaptador es el gradiente que ve el optimizador


def test_installed_adapter_and_names():
    for mode in MODES:
        prob = pp.PiezoProblem(mode)
        assert prob.name in ("piezo_bimorph_indirect_2d", "piezo_bimorph_direct_2d")
        assert prob.n_out == 8 and prob.w == ({"pde": 1.0, "bc": 1.0, "nw": 1.0} if mode == "converse" else {"pde": 1.0, "bc": 10.0, "nw": 1.0})
    assert pp.PiezoProblem("converse").routing and not pp.PiezoProblem("direct").routing


# ------------------------------------------------------------------ bloques B
@pytest.mark.parametrize("mode", MODES)
def test_blocks_are_finite_symmetric_psd_with_eight_outputs(mode):
    prob, model, P = _small(mode, width=4, layer=24)
    X = gp.subset(P, 24)
    edge = gp.gn_blocks(prob, model, X, "edge", chunk=16)
    inp = gp.gn_blocks(prob, model, X, "input", chunk=16)
    dia = gp.gn_blocks(prob, model, X, "diag", chunk=16)
    assert edge[-1].shape[0] == 8 and edge[0].shape[1] == 4            # 8 salidas, 4 entradas
    for B in edge + inp + dia:
        assert B.dtype == torch.float64 and bool(torch.isfinite(B).all())
    for l, B in enumerate(edge):
        O, I, G, _ = B.shape
        assert float((B - B.transpose(-1, -2)).abs().max()) <= 1e-9 * float(B.abs().max())
        assert float((dia[l] - torch.diagonal(B, dim1=-2, dim2=-1)).abs().max()) < 1e-12 + 1e-8 * float(dia[l].abs().max())
        Bk = B.reshape(O * I, G, G)
        Bk = 0.5 * (Bk + Bk.transpose(-1, -2))
        assert float(torch.linalg.eigvalsh(Bk).min()) >= -1e-9 * float(Bk.abs().max())
    for B in inp:
        assert float((B - B.transpose(-1, -2)).abs().max()) <= 1e-9 * float(B.abs().max())


def test_routing_changes_the_jacobian_as_declared_in_the_converse_problem():
    """Con enrutamiento, las filas de sx, sz, tau de la ULTIMA capa no reciben gradiente de las constitutivas mecanicas; sin el, si (el J coincide con el gradiente exacto)."""
    prob, model, P = _small("converse", layer=12)
    names = ("constitutive_sigma_xx", "constitutive_sigma_yy", "constitutive_tau_xy")
    X = {"interior": P["interior"][:8]}
    coefs = gp.effective_coefficients(model)

    def jac(stop):
        prob.spec["stopgrad"] = stop
        def f(cc):
            terms = prob.residual_terms(_rt().make_model_fn(model, cc), X)
            return tuple(r for r, _, n in terms if n in names)
        return torch.func.jacrev(f)(tuple(coefs))
    j_stop, j_full = jac(True), jac(False)
    prob.spec["stopgrad"] = True
    own = lambda J: [float(t[2][:, i].abs().sum()) for i, t in zip((3, 4, 5), J)]     # J de cada residuo respecto de SU fila de salida (capa final)
    assert own(j_stop) == [0.0, 0.0, 0.0]
    assert all(v > 0 for v in own(j_full))


# ------------------------------------------------------------------ aislamiento
def test_loss_pools_and_adapter_never_read_the_reference_and_imports_are_clean():
    prob = pp.PiezoProblem("direct", fem_path="/nonexistent/fem.npz", interior_per_layer=8, boundary=8, interface=8)
    model = prob.build("rbf", 3, 11, 1)
    P = prob.pools(1)
    prob.loss(model, P)
    _rt().terms_loss(prob, model, P)
    gp.gn_blocks(prob, model, gp.subset(P, 8), "diag", chunk=8)
    assert prob._fem is None
    for fname in ("piezo_problem.py", "run_piezo.py", "kaggle_block_piezo.py"):
        tree = ast.parse((PZ / fname).read_text(encoding="utf-8"))
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and n.level == 0}
        assert "evals" not in mods and "registry" not in mods
    src = (ROOT / "src/mechanisms/residual_terms.py").read_text(encoding="utf-8")
    assert "evals" not in ast.dump(ast.parse(src))


def test_check_B_equals_ID_bit_for_bit_in_both_problems():
    import run_piezo
    with tempfile.TemporaryDirectory() as td:
        r = run_piezo.check_id(td, steps=4, smoke_pool=16)
    assert r["pass"] and all(r[m]["pass"] for m in MODES)


# ------------------------------------------------------------------ artefactos (artifacts.py) y W&B offline
def test_run_artifacts_are_complete_and_do_not_change_the_trajectory():
    import run_piezo
    import artifacts as art
    for mode in MODES:
        r0 = run_piezo.run_one(mode, "E", 4001, 6, "cpu", "float64", pre_every=3, pre_n=32, smoke_pool=16, chunk=2, eval_every=3, quiet=True)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            r1 = run_piezo.run_one(mode, "E", 4001, 6, "cpu", "float64", pre_every=3, pre_n=32, smoke_pool=16, chunk=2, eval_every=3, quiet=True, out_dir=td, artifacts=True, wandb_offline=False)
            assert r1["score_final"] == r0["score_final"] and r1["loss_end"] == r0["loss_end"]          # solo lectura: misma trayectoria al bit
            with np.load(r1["artifacts"]["path"]) as z:
                keys = set(z.files)
                for need in ("s000000/state/layers.0.coefficients", "s000006/state/layers.0.pre_P", "s000006/fields/values", "s000006/resid_grid/equilibrium_y", "s000006/resid_pool/gauss",
                             "s000006/consistency_grid/discrepancy", "s000006/occupancy/layer0", "refresh00/layer0/trace", "refresh00/layer0/cond", "refresh00/layer0/P_sv", "hist/loss", "hist/gradnorm/layers.0.coefficients",
                             "settle/step000006/layer0", "meta_json"):
                    assert need in keys, need
                assert z["s000006/fields/values"].shape[1] == 8 and z["s000006/fields/values"].dtype == np.float32
                assert any(k.startswith("hist/term/") for k in keys)
                meta = json.loads(str(z["meta_json"]))
                assert meta["seed"] == 4001 and "torch" in meta["versions"] and meta["hashes"]["piezo_problem.py"]

            assert (Path(td) / f"fem_{mode}_reference.npz").exists()
            assert r1["artifacts"]["wandb"] == "disabled"


def test_save_run_artifacts_standalone():
    import artifacts as art
    prob, model, P = _small("direct", layer=8)
    with tempfile.TemporaryDirectory() as td:
        path = art.save_run_artifacts(model, prob, "unit", td, P=P, step=0, seed=1)
        with np.load(path) as z:
            assert "s000000/fields/values" in z.files and "s000000/resid_pool/bc_top_zero_net_charge" in z.files


def test_wandb_offline_writes_the_run_folder_in_an_isolated_subprocess():
    """W&B real (offline) en un subproceso con timeout: en Windows el servicio de wandb puede cerrar el proceso (violacion de acceso); entonces se omite y queda documentado."""
    import subprocess
    code = f"""
import sys, tempfile, os, json
sys.path.insert(0, r'{PZ}')
import run_piezo
td = tempfile.mkdtemp()
r = run_piezo.run_one('converse', 'B', 4001, 4, 'cpu', 'float64', smoke_pool=8, eval_every=2, quiet=True, out_dir=td, artifacts=True, wandb_offline=True)
w = os.path.join(td, 'wandb')
print(json.dumps(dict(w=r['artifacts']['wandb'], runs=[p for p in os.listdir(w) if p.startswith('offline-run')] if os.path.isdir(w) else [])))
"""
    env = dict(os.environ, X28_WANDB_FORCE="1")
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env, cwd=tempfile.gettempdir())
    except subprocess.TimeoutExpired:
        pytest.skip("wandb offline excede el timeout en este sistema: se verifica en Linux (Kaggle/Colab)")
    if p.returncode != 0:
        pytest.skip(f"el subproceso con wandb termino con codigo {p.returncode} (en Windows: violacion de acceso del servicio de wandb); W&B offline se verifica en Linux (Kaggle/Colab)")
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["w"].startswith("offline") and res["runs"], res


# ------------------------------------------------------------------ alineacion con el paper: semillas, brazo MLP-paper, tensiones P3
def test_manifest_uses_the_paper_seeds_and_has_a_low_priority_mlp_reference():
    kb = _load(PZ / "kaggle_block_piezo.py", "x28_kb")
    jobs = kb.all_jobs()
    seeds = sorted({s for _, s, a in jobs if a != "MLP"})
    assert seeds == [20260728, 20260729, 20260730, 20260731, 20260732]
    assert len(jobs) == 52 and jobs[-2:] == [("converse", 20260728, "MLP"), ("direct", 20260728, "MLP")]
    assert kb.block_jobs("k26") == jobs[-2:] and len(kb.block_jobs("k01")) == 2
    import run_piezo
    assert run_piezo.SEEDS_MAIN == tuple(seeds)


@pytest.mark.parametrize("mode", MODES)
def test_flux_score_against_the_refined_p3_fem(mode):
    prob = pp.PiezoProblem(mode)
    if not (PZ / "reference" / f"fem_flux_{mode}.npz").exists():
        pytest.skip("sin referencia P3")
    m = prob.build("rbf", 4, 11, 3)
    r = prob.score_flux(m)
    assert all(np.isfinite(r[k]["nrms_pct"]) for k in ("sigma_xx", "sigma_yy", "tau_xy", "D_x", "D_y"))
    ref_max = {"converse": (4416.0, 10.6244e-6), "direct": (59460.0, 0.3351e-6)}[mode]
    assert abs(r["sigma_ref_max"] / ref_max[0] - 1) < 0.01 and abs(r["D_y_ref_max"] / ref_max[1] - 1) < 0.02      # maximos de normalizacion del paper
    s = prob.score(m)
    assert all(k in s for k in ("rmse_u", "mae_v", "max_abs_phi"))
