"""Prueba (ii) del EXP-0016: el barrido en k = 1.00 reproduce el control bit a bit, y el entrenamiento nunca toca la referencia.

Usa el codigo de la campana de baseline (var/agent_runs/..., no versionado) y la referencia Cole-Hopf de Burgers; si faltan, las pruebas se saltan.
Todo corre en CPU con una red y un lote chicos, y L-BFGS limitado por iteraciones (no por reloj) para que la comparacion sea determinista.
"""
import contextlib
import importlib.util
import inspect
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff/sweep.py"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
CAMPAIGN = BL / "burgers-close-20261003/pikan_campaign.py"
REF = BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz"
AC_REF = BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat"
AVAILABLE = SWEEP.exists() and CAMPAIGN.exists() and REF.exists()
AC_AVAILABLE = AVAILABLE and AC_REF.exists()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(AVAILABLE, "faltan el codigo de la campana o la referencia de Burgers")
class SweepEquivalenceTests(unittest.TestCase):
    OVERRIDE = {}

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)
        cls.sw = _load(SWEEP, "exp0016_sweep")
        cls.pc = cls.sw.load_campaign(CAMPAIGN)
        cls.cfg = dict(problem="burgers", ref=str(REF), arm="rbf", width=8, bases=11, dtype="float32", device="cpu", adam_steps=40,
                       adam_seconds_cap=0.0, lbfgs_seconds=1e6, lbfgs_chunk=10, lbfgs_max_iter=60, n_col=256, snapshot_every=1e9,
                       grid=list(cls.sw.GRID), order=None, seed=301, record_every=10)
        cls.cfg.update(cls.OVERRIDE)
        cls.tmp = tempfile.TemporaryDirectory()
        # control: la implementacion original de la campana, sin tocar
        wargs = Namespace(**{**cls.cfg, "lbfgs_dtype": "same", "out": str(Path(cls.tmp.name) / "control")})
        with contextlib.redirect_stdout(io.StringIO()):
            cls.pc.run_worker(wargs)
        cls.control = Path(cls.tmp.name) / "control"
        cls.res, cls.states, cls.trace = cls.sw.run_seed(cls.pc, Namespace(**cls.cfg))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _eq(self, a, b):
        self.assertEqual(set(a), set(b))
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), f"difiere el tensor {k}")

    def test_adam_final_is_bit_identical(self):
        ck = torch.load(self.control / "checkpoint-adam-final.pt", map_location="cpu", weights_only=False)["state_dict"]
        self._eq(ck, self.states["adam_final"])

    def test_k_1_00_lbfgs_is_bit_identical_to_control(self):
        ck = torch.load(self.control / "checkpoint-lbfgs-final.pt", map_location="cpu", weights_only=False)["state_dict"]
        self._eq(ck, self.states[1.0])

    def test_k_1_00_status_iterations_and_scores_match_control(self):
        import json

        run = json.loads((self.control / "run.json").read_text())
        mine = self.res["runs"]["1.0"]
        self.assertEqual(mine["lbfgs_status"], run["lbfgs_status"])
        self.assertEqual(mine["lbfgs_iterations"], run["lbfgs_iterations"])
        self.assertEqual(mine["final_score"], run["score_lbfgs_final"])
        self.assertEqual(mine["adam_score_at_k"], run["score_adam_final"])

    def test_adam_loss_history_is_bit_identical(self):
        import json

        hist = json.loads((self.control / "history.json").read_text())
        self.assertEqual([h["loss"] for h in hist], [h["loss"] for h in self.res["hist"]])
        self.assertEqual([h["step"] for h in hist], [h["step"] for h in self.res["hist"]])

    def test_order_of_the_sweep_does_not_change_any_result(self):
        rev = self.sw.run_seed(self.pc, Namespace(**{**self.cfg, "order": sorted(self.cfg["grid"], reverse=True)}))[1]
        for f in self.cfg["grid"]:
            self._eq(self.states[f], rev[f])

    def test_handoff_runs_differ_from_each_other(self):
        # el barrido no es trivial: los traspasos distintos terminan en estados distintos
        hashes = {v["final_state_sha256"] for v in self.res["runs"].values()}
        self.assertEqual(len(hashes), len(self.cfg["grid"]))

    def test_caps_equalize_the_total_clock(self):
        t_total = self.res["t_total"]
        for f, r in self.res["runs"].items():
            self.assertAlmostEqual(r["cap"] + self.res["ckpt_seconds"][str(r["k"])], t_total, places=6)
        self.assertAlmostEqual(self.res["runs"]["1.0"]["cap"], self.cfg["lbfgs_seconds"], places=6)

    def test_trigger_is_computed_without_the_reference(self):
        self.assertEqual(set(self.res["trigger"]), {"0.05", "0.1", "0.2"})

    # ---- registro de la capa 0 y del residuo por punto (v2) ----
    def test_recording_does_not_change_the_training(self):
        off = self.sw.run_seed(self.pc, Namespace(**{**self.cfg, "record_every": 0}))
        self.assertIsNone(off[2])
        for key in self.states:
            self._eq(self.states[key], off[1][key])

    def test_trace_has_the_preregistered_layout(self):
        t = self.trace
        steps = self.cfg["adam_steps"]
        self.assertEqual(list(t["steps"]), list(range(0, steps + 1, 10)))
        T, w = len(t["steps"]), self.cfg["width"]
        self.assertEqual(t["coeffs"].shape, (T, w, 2, self.cfg["bases"]))
        self.assertEqual(t["base"].shape, (T, w, 2))
        self.assertEqual(t["resid2"].shape, (T, t["points"].shape[0]))
        self.assertEqual(t["points"].shape[1], 2)
        self.assertEqual(t["centers"].shape, (self.cfg["bases"],))
        self.assertTrue((t["resid2"] >= 0).all())

    def test_trace_coefficients_match_the_model_states(self):
        self._eq({"c": torch.as_tensor(self.trace["coeffs"][-1])},
                 {"c": self.states["adam_final"]["layers.0.coefficients"].cpu()})
        init = self.pc.make_problem(self.cfg["problem"], self.cfg["ref"]).build(
            self.cfg["arm"], self.cfg["width"], self.cfg["bases"], self.cfg["seed"])
        self._eq({"c": torch.as_tensor(self.trace["coeffs"][0])}, {"c": init.layers[0].coefficients.detach()})

    def test_resid2_mean_is_the_training_loss(self):
        prob = self.pc.make_problem(self.cfg["problem"], self.cfg["ref"])
        if self.cfg.get("n_col") and hasattr(prob, "n_col"):
            prob.n_col = self.cfg["n_col"]
        model = prob.build(self.cfg["arm"], self.cfg["width"], self.cfg["bases"], self.cfg["seed"]).to(dtype=torch.float32)
        P = prob.pools(self.cfg["seed"], torch.device("cpu"), torch.float32)
        model.load_state_dict(self.states["adam_final"])
        loss = float(prob.loss(model, P)[0].detach())
        self.assertAlmostEqual(float(self.trace["resid2"][-1].mean()), loss, delta=1e-5 * abs(loss))

    def test_warmup_does_not_change_any_result_and_uses_the_real_shapes(self):
        info = self.sw.warmup(self.pc, Namespace(**self.cfg), steps=10)
        again = self.sw.run_seed(self.pc, Namespace(**self.cfg))[1]
        for key in self.states:
            self._eq(self.states[key], again[key])
        prob = self.pc.make_problem(self.cfg["problem"], self.cfg["ref"])
        net = prob.build(self.cfg["arm"], self.cfg["width"], self.cfg["bases"], self.cfg["seed"])
        self.assertEqual(info["n_params"], sum(q.numel() for q in net.parameters()))
        self.assertEqual(info["pool_points"], self.trace["points"].shape[0])

    def test_settling_trigger_is_logged_without_the_reference(self):
        self.assertEqual(set(self.res["trigger_settling"]),
                         {"q0.8_rho0.10", "q0.6_rho0.10", "q0.9_rho0.10", "q0.8_rho0.05", "q0.8_rho0.20"})
        self.assertGreaterEqual(self.res["record_seconds"], 0.0)


@unittest.skipUnless(AC_AVAILABLE, "faltan el codigo de la campana o la referencia de Allen-Cahn")
class SweepEquivalenceAllenCahnTests(SweepEquivalenceTests):
    """Misma comprobacion con la otra clase de problema (Allen-Cahn, otro conjunto de puntos y otra transformacion dura)."""

    OVERRIDE = dict(problem="ac", ref=str(AC_REF), width=8, adam_steps=20, lbfgs_max_iter=30, seed=601)


@unittest.skipUnless(AVAILABLE, "faltan el codigo de la campana o la referencia de Burgers")
class SweepTrainingNeverTouchesTheReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.sw = _load(SWEEP, "exp0016_sweep_b")
        cls.pc = cls.sw.load_campaign(CAMPAIGN)
        prob = cls.pc.make_problem("burgers", str(REF))
        prob.n_col = 256
        cls.prob = prob

    def _stub(self):
        prob = self.prob

        class LossOnly:  # sin score, sin referencia: si una etapa los pidiera, fallaria
            loss = staticmethod(prob.loss)

        return LossOnly()

    def _model(self):
        prob = self.prob
        model = prob.build("rbf", 8, 11, 301).to(dtype=torch.float32)
        return model, prob.pools(301, torch.device("cpu"), torch.float32)

    def test_stages_run_with_an_object_that_only_has_loss(self):
        model, P = self._model()
        A = self.sw.adam_stage(self._stub(), model, P, steps=20, ckpt_steps=[10, 20])
        self.assertEqual(A["status"], "adam_complete")
        self.assertEqual(sorted(A["ckpts"]), [10, 20])
        L = self.sw.lbfgs_stage(self._stub(), model, P, seconds_cap=1e6, max_iter=20, chunk=10)
        self.assertGreater(L["iterations"], 0)

    def test_stage_sources_do_not_mention_scoring_or_reference(self):
        for fn in (self.sw.adam_stage, self.sw.lbfgs_stage):
            src = inspect.getsource(fn)
            for word in (".score", "score(", "ref_path", ".ref", "ref="):
                self.assertNotIn(word, src, f"{fn.__name__} menciona {word!r}")

    def test_time_cap_zero_returns_the_starting_state(self):
        model, P = self._model()
        start = self.sw.clone_state(model)
        L = self.sw.lbfgs_stage(self._stub(), model, P, seconds_cap=0.0)
        self.assertEqual(L["status"], "lbfgs_time_cap")
        self.assertEqual(L["iterations"], 0)
        for k in start:
            self.assertTrue(torch.equal(start[k], model.state_dict()[k]))

    def test_snapshots_are_ordered_and_end_at_the_final_state(self):
        model, P = self._model()
        tick = {"t": 0.0}

        def clock():
            tick["t"] += 5.0
            return tick["t"]

        L = self.sw.lbfgs_stage(self._stub(), model, P, seconds_cap=1e9, max_iter=40, chunk=10, snapshot_every=20.0, clock=clock)
        times = [s[0] for s in L["snaps"]]
        self.assertEqual(times, sorted(times))
        self.assertGreaterEqual(len(L["snaps"]), 2)
        for k, v in L["final_state"].items():
            self.assertTrue(torch.equal(v, L["snaps"][-1][2][k]))

    def test_grid_must_include_the_control(self):
        with self.assertRaises(ValueError):
            self.sw.run_seed(self.pc, Namespace(problem="burgers", ref=str(REF), arm="rbf", width=8, bases=11, dtype="float32",
                                                device="cpu", adam_steps=20, adam_seconds_cap=0.0, lbfgs_seconds=1.0,
                                                lbfgs_chunk=10, lbfgs_max_iter=10, n_col=256, snapshot_every=30.0,
                                                grid=[0.25, 0.5], order=None, seed=301))


if __name__ == "__main__":
    unittest.main()
