"""EXP-0018: ejecutor del techo oraculo (prueba minima de punta a punta en CPU y coincidencia de A* con el EXP-0017)."""
import importlib.util
import json
import math
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP18 = ROOT / "experiments/mechanisms/EXP-0018-responsibility-ceiling"
EXP16 = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff"
EXP17 = ROOT / "experiments/mechanisms/EXP-0017-representation-battery"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
CAMPAIGN = BL / "burgers-close-20261003/pikan_campaign.py"
BREF = BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz"
AREF = BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat"
STAGE_A = EXP16 / "results/stage-a/stageA"
AVAILABLE = (EXP18 / "run_ceiling.py").exists() and CAMPAIGN.exists() and BREF.exists()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(AVAILABLE, "faltan el ejecutor, la campana o la referencia")
class CeilingExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.rc = _load(EXP18 / "run_ceiling.py", "exp0018_run_ceiling")
        cls.pc = cls.rc.sw.load_campaign(CAMPAIGN)

    def tiny(self, **kw):
        base = dict(problem="burgers", ref=str(BREF), width=8, bases=11, adam_steps=40, lbfgs_seconds=1e9, lbfgs_chunk=10, lbfgs_max_iter=40,
                    n_col=256, device="cpu", state_dir="", arms=["B", "P", "H", "N", "O", "NH"])
        base.update(kw)
        return Namespace(**base)

    def test_tiny_end_to_end_has_every_arm_and_a_control_identical_to_plain_lbfgs(self):
        a = self.tiny()
        res = self.rc.run_job(self.pc, a, 301)
        self.assertEqual(set(res["arms"]), {"B", "P", "H", "N", "O", "NH"})
        self.assertEqual(res["label"], "oracle_ceiling")
        for arm, r in res["arms"].items():
            self.assertTrue(r["finite"], arm)
            self.assertTrue(math.isfinite(r["global_l2"]), arm)
            self.assertIsNotNone(r["critical_l2"], arm)
        # B es exactamente el L-BFGS de la receta desde el mismo estado
        prob = self.pc.make_problem("burgers", str(BREF))
        prob.n_col = 256
        P = prob.pools(301, torch.device("cpu"), torch.float32)
        m = prob.build("rbf", 8, 11, 301).to(dtype=torch.float32)
        self.rc.sw.adam_stage(prob, m, P, 40)
        self.rc.sw.lbfgs_stage(prob, m, P, seconds_cap=1e9, max_iter=40, chunk=10, snapshot_every=1e9)
        self.assertEqual(res["arms"]["B"]["state_sha256"], self.rc.sw.state_sha256(self.rc.sw.clone_state(m)))
        # las demas ramas son distintas del control (el foco hace algo)
        hashes = {arm: r["state_sha256"] for arm, r in res["arms"].items()}
        self.assertEqual(len(set(hashes.values())), len(hashes), hashes)

    def test_arm_transforms_have_the_documented_structure(self):
        rng = torch.Generator().manual_seed(0)
        A = [torch.rand(2, 11, generator=rng) + 0.1, torch.rand(6, 11, generator=rng) + 0.1, torch.rand(6, 11, generator=rng) + 0.1]
        O = [torch.rand(2, 11, generator=rng) + 0.1, torch.rand(6, 11, generator=rng) + 0.1, torch.rand(6, 11, generator=rng) + 0.1]
        self.assertEqual(self.rc.arm_transforms("B", A, O, 1), (None, None))
        sP, _ = self.rc.arm_transforms("P", A, O, 1)
        sN, _ = self.rc.arm_transforms("N", A, O, 1)
        for p, n in zip(sP, sN):
            self.assertEqual(sorted(p.reshape(-1).tolist()), sorted(n.reshape(-1).tolist()))
        _, mH = self.rc.arm_transforms("H", A, O, 1)
        _, mNH = self.rc.arm_transforms("NH", A, O, 1)
        self.assertEqual(sum(int(m.sum()) for m in mH), sum(int(m.sum()) for m in mNH))
        self.assertNotEqual([m.tolist() for m in mH], [m.tolist() for m in mNH])
        with self.assertRaises(ValueError):
            self.rc.arm_transforms("Z", A, O, 1)

    def test_critical_region_definitions(self):
        import numpy as np

        pts = np.array([[0.0, 0.5], [0.5, 0.5], [0.01, 0.1], [0.0, 0.2]])
        np.testing.assert_array_equal(self.rc.critical_mask("burgers", pts, np.zeros(4)), [True, False, False, False])
        np.testing.assert_array_equal(self.rc.critical_mask("ac", pts, np.array([0.0, 1.0, -0.89, -0.95])), [True, False, True, False])

    def test_aistar_matches_the_one_computed_in_exp0017_for_the_same_states(self):
        torch.set_num_threads(4)                                   # el EXP-0017 se calculo con 4 hilos: el orden de las sumas en float32 depende de ello
        self.addCleanup(torch.set_num_threads, 1)
        for problem, seed, f, ref, width, k in (("burgers", 301, "1.0", BREF, 48, 20000), ("ac", 601, "1.0", AREF, 16, 40000)):
            state_file = STAGE_A / f"{problem}-seed{seed}-adam-k{k}.pt"
            ref_json = EXP17 / f"results/states/{problem}-{seed}-{f}.json"
            if not (state_file.exists() and ref_json.exists() and Path(ref).exists()):
                self.skipTest("faltan los estados de la etapa A o los resultados del EXP-0017")
            prob = self.pc.make_problem(problem, str(ref))
            model = prob.build("rbf", width, 11, seed).to(dtype=torch.float32)
            model.load_state_dict(torch.load(state_file, map_location="cpu", weights_only=False)["state_dict"])
            pts, u_ref = self.rc.eval_grid(prob, problem)
            with torch.no_grad():
                err = abs(prob.transform(pts, model(pts)).numpy().reshape(u_ref.shape) - u_ref).reshape(-1)
            import numpy as np

            idx = np.sort(np.random.default_rng(self.rc.SUB_SEED).choice(pts.shape[0], size=min(self.rc.N_SUB, pts.shape[0]), replace=False))
            A, _ = self.rc.oracle_responsibility(model, pts[idx], err[idx])
            share = self.rc.sig.top_share(self.rc.sig.flat(A), 0.2)
            expected = json.loads(ref_json.read_text())["attr"]["top20_share_Astar"]
            self.assertAlmostEqual(share, expected, delta=1e-12, msg=problem)

    def test_oracle_error_tensor_follows_the_device_of_the_points(self):
        import inspect

        self.assertIn("device=xs.device", inspect.getsource(self.rc.oracle_responsibility))   # fallo real de la primera prueba en T4


if __name__ == "__main__":
    unittest.main()
