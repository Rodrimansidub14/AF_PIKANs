"""EXP-0018: kernel de la prueba 7 y orquestador del bloque (planificador y corrida minima de punta a punta en CPU)."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "experiments/mechanisms/EXP-0018-responsibility-ceiling/kaggle"
DATASET = K / "dataset"
PRECHECK = K / "precheck/precheck_ceiling_gpu.py"
CEILING = K / "ceiling/ceiling_gpu.py"
AVAILABLE = PRECHECK.exists() and CEILING.exists() and (DATASET / "run_ceiling.py").exists()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class FakeProc:
    def __init__(self, clock, duration, rc=0):
        self.clock, self.end, self.rc, self.killed = clock, clock() + duration, rc, False

    def poll(self):
        return -9 if self.killed else (self.rc if self.clock() >= self.end else None)

    def kill(self):
        self.killed = True


@unittest.skipUnless(AVAILABLE, "falta el bundle del EXP-0018")
class OrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load(CEILING, "exp0018_ceiling_gpu")

    def run_(self, jobs, durations, n=2, cutoff=1e9, hard=1e9, rcs=None):
        clock, started = Clock(), []

        def launch(job, slot):
            started.append((job, slot, clock()))
            return FakeProc(clock, durations[job], (rcs or {}).get(job, 0))

        return self.m.run_jobs(jobs, n, cutoff, hard, launch, poll=1.0, clock=clock, sleep=clock.sleep), started

    def test_default_queue_burgers_first_then_allen_cahn_five_seeds_each_and_no_heat_no_nh(self):
        jobs = self.m.JOBS
        self.assertEqual(jobs, [("burgers", s) for s in (301, 302, 303, 304, 305)] + [("ac", s) for s in (601, 602, 603, 604, 605)])
        self.assertEqual(self.m.ARMS, ["B", "P", "H", "N", "O"])

    def test_topes_del_spec(self):
        self.assertEqual(self.m.CUTOFF_S, 2.5 * 3600)
        self.assertEqual(self.m.HARD_CAP_S, 3.0 * 3600)

    def test_cutoff_hard_cap_failure_and_order(self):
        jobs = [("burgers", s) for s in range(5)]
        ledger, started = self.run_(jobs, {j: 100 for j in jobs}, cutoff=150)
        self.assertEqual([s[0] for s in started], jobs[:4])
        self.assertEqual({e["job"]: e["status"] for e in ledger}[jobs[4]], "not_started_cutoff")
        ledger, _ = self.run_(jobs[:3], {j: 1000 for j in jobs}, hard=300)
        by = {e["job"]: e["status"] for e in ledger}
        self.assertEqual([by[j] for j in jobs[:3]], ["killed_hard_cap", "killed_hard_cap", "not_started_cutoff"])
        ledger, _ = self.run_(jobs[:3], {j: 10 for j in jobs}, rcs={jobs[0]: 1})
        by = {e["job"]: e["status"] for e in ledger}
        self.assertEqual([by[j] for j in jobs[:3]], ["failed", "ok", "ok"])


@unittest.skipUnless(AVAILABLE, "falta el bundle del EXP-0018")
class TinyEndToEnd(unittest.TestCase):
    def _bundle_copy(self):
        d = Path(tempfile.mkdtemp())
        for p in DATASET.iterdir():
            if p.suffix != ".pt":                       # los estados reales no sirven para la red chica
                (d / p.name).write_bytes(p.read_bytes())
        return d

    def test_precheck_runs_in_tiny_cpu_mode_and_b_is_reproducible(self):
        bundle, out = self._bundle_copy(), Path(tempfile.mkdtemp())
        env = {**os.environ, "PRECHECK_TINY": "1", "BUNDLE_DIR": str(bundle), "OUT_DIR": str(out), "OMP_NUM_THREADS": "1"}
        r = subprocess.run([sys.executable, str(PRECHECK)], env=env, capture_output=True, text=True, cwd=str(bundle))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        summary = json.loads((out / "summary.json").read_text())
        self.assertTrue(summary["passed"], summary)
        self.assertTrue(summary["verdict"]["identical_twice"])

    def test_ceiling_orchestrator_tiny_produces_one_json_per_seed_with_all_arms(self):
        m = _load(CEILING, "exp0018_ceiling_tiny")
        bundle, out = self._bundle_copy(), Path(tempfile.mkdtemp())
        m.JOBS[:] = [("burgers", 301), ("ac", 604)]      # 604: sin estado guardado -> entrena su Adam aqui
        old = {k: os.environ.get(k) for k in ("CEILING_TINY", "BUNDLE_DIR", "OUT_DIR")}
        os.environ.update(CEILING_TINY="1", BUNDLE_DIR=str(bundle), OUT_DIR=str(out))
        try:
            m.main()
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        ledger = json.loads((out / "ledger.json").read_text())
        self.assertTrue(ledger["complete"], ledger)
        for f in ("burgers-seed301.json", "ac-seed604.json"):
            res = json.loads((out / f).read_text())
            self.assertEqual(set(res["arms"]), {"B", "P", "H", "N", "O"})
            self.assertEqual(res["label"], "oracle_ceiling")
            self.assertTrue(all(a["finite"] for a in res["arms"].values()))
        self.assertEqual(json.loads((out / "ac-seed604.json").read_text())["adam_state"], "trained_here")


if __name__ == "__main__":
    unittest.main()
