"""Logica de decision de la prueba previa en GPU del EXP-0016 y su humo en CPU (sin GPU)."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff"
SCRIPT = EXP / "kaggle/precheck/precheck_gpu.py"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"


def _load():
    spec = importlib.util.spec_from_file_location("exp0016_precheck", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _state(v):
    return {"a": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64) * v, "b": torch.tensor([[1.0, -1.0]], dtype=torch.float64) * v}


@unittest.skipUnless(SCRIPT.exists(), "falta el script de la prueba previa")
class DecisionLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = _load()

    def test_identical_runs_are_within_noise(self):
        r = self.p.compare([_state(1.0)] * 3, _state(1.0))
        self.assertEqual((r["noise"], r["deviation"]), (0.0, 0.0))
        self.assertTrue(r["within_noise"])

    def test_zero_noise_requires_zero_deviation(self):
        r = self.p.compare([_state(1.0)] * 3, _state(1.0 + 1e-9))
        self.assertEqual(r["noise"], 0.0)
        self.assertGreater(r["deviation"], 0.0)
        self.assertFalse(r["within_noise"])

    def test_deviation_inside_twice_the_noise_passes(self):
        controls = [_state(1.0), _state(1.0 + 1e-6), _state(1.0 - 1e-6)]
        self.assertTrue(self.p.compare(controls, _state(1.0 + 1e-6))["within_noise"])

    def test_deviation_beyond_twice_the_noise_fails(self):
        controls = [_state(1.0), _state(1.0 + 1e-6), _state(1.0 - 1e-6)]
        self.assertFalse(self.p.compare(controls, _state(1.0 + 1e-3))["within_noise"])

    def test_scalar_comparison(self):
        self.assertTrue(self.p.compare_scalar([0.5, 0.5, 0.5], 0.5)["within_noise"])
        self.assertFalse(self.p.compare_scalar([0.5, 0.5, 0.5], 0.5001)["within_noise"])
        self.assertTrue(self.p.compare_scalar([0.5, 0.5001, 0.4999], 0.50005)["within_noise"])

    def test_distance_is_relative_and_symmetric_in_zero(self):
        rel, mx = self.p.rel_dist(_state(2.0), _state(1.0))
        self.assertAlmostEqual(rel, 1.0, places=6)
        self.assertGreater(mx, 0)
        self.assertEqual(self.p.rel_dist(_state(1.0), _state(1.0)), (0.0, 0.0))


@unittest.skipUnless(SCRIPT.exists() and (BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz").exists(),
                     "faltan las referencias de la campana")
class TinyCpuSmokeTest(unittest.TestCase):
    def test_whole_precheck_runs_on_cpu_and_cpu_is_deterministic(self):
        bundle = Path(tempfile.mkdtemp())
        for src in (EXP / "sweep.py", ROOT / "src/mechanisms/stall_handoff.py", ROOT / "src/mechanisms/settling_focus.py",
                    BL / "burgers-close-20261003/pikan_campaign.py", BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz",
                    BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat"):
            (bundle / src.name).write_bytes(src.read_bytes())
        out = Path(tempfile.mkdtemp())
        env = {**os.environ, "PRECHECK_TINY": "1", "BUNDLE_DIR": str(bundle), "OUT_DIR": str(out), "OMP_NUM_THREADS": "1"}
        r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, cwd=str(bundle))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(set(summary["problems"]), {"burgers", "ac"})
        self.assertTrue(summary["all_within_noise"], "en CPU, un solo hilo, el barrido debe reproducir los controles")
        for p in summary["problems"].values():
            self.assertEqual(p["comparison"]["lbfgs_state"]["noise"], 0.0)
            self.assertEqual(len(p["timing"]["sweep_lbfgs"]), 5)
            self.assertIn("record_overhead_fraction", p["timing"])


WARMUP_SCRIPT = EXP / "kaggle/warmup_check/verify_warmup_gpu.py"


@unittest.skipUnless(WARMUP_SCRIPT.exists() and (BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz").exists(),
                     "faltan el script de verificacion o las referencias")
class WarmupCheckTinyCpuTest(unittest.TestCase):
    def test_warmup_check_runs_on_cpu_and_cpu_is_deterministic(self):
        bundle = Path(tempfile.mkdtemp())
        for src in (EXP / "sweep.py", ROOT / "src/mechanisms/stall_handoff.py", ROOT / "src/mechanisms/settling_focus.py",
                    BL / "burgers-close-20261003/pikan_campaign.py", BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz"):
            (bundle / src.name).write_bytes(src.read_bytes())
        out = Path(tempfile.mkdtemp())
        env = {**os.environ, "WARMUP_TINY": "1", "BUNDLE_DIR": str(bundle), "OUT_DIR": str(out), "OMP_NUM_THREADS": "1"}
        r = subprocess.run([sys.executable, str(WARMUP_SCRIPT)], env=env, capture_output=True, text=True, cwd=str(bundle))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        summary = json.loads((out / "summary.json").read_text())
        v = summary["verdict"]
        self.assertTrue(v["A_equals_B"] and v["A_equals_S"] and v["warmup_works"], v)
        self.assertEqual(set(summary["runs"]), {"A", "B", "S"})
        self.assertFalse(v["A_equals_prior_cold"])   # la comparacion con los hashes de GPU no puede coincidir en una red chica en CPU


if __name__ == "__main__":
    unittest.main()
