"""EXP-0024 T1: planificador del bloque y recorrido minimo en CPU de la prueba corta y del bloque."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "experiments/mechanisms/EXP-0024-wind-tunnel-protocol/kaggle"
DATASET, SHORT, BLOCK = K / "dataset", K / "shorttest/t1_shorttest_gpu.py", K / "t1block/t1_block_gpu.py"
AVAILABLE = SHORT.exists() and BLOCK.exists() and (DATASET / "t1_trainer.py").exists()


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


class Fake:
    def __init__(self, clock, duration, rc=0):
        self.clock, self.end, self.rc, self.killed = clock, clock() + duration, rc, False

    def poll(self):
        return -9 if self.killed else (self.rc if self.clock() >= self.end else None)

    def kill(self):
        self.killed = True


@unittest.skipUnless(AVAILABLE, "falta el bundle del EXP-0024")
class Scheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load(BLOCK, "exp0024_block")

    def run_(self, jobs, dur, cutoff=1e9, hard=1e9, rcs=None):
        clock, started = Clock(), []

        def launch(job, slot):
            started.append((job, slot))
            return Fake(clock, dur[job], (rcs or {}).get(job, 0))

        return self.m.run_jobs(jobs, 2, cutoff, hard, launch, poll=1.0, clock=clock, sleep=clock.sleep), started

    def test_queue_designs_and_caps(self):
        self.assertEqual(self.m.JOBS, [("burgers", s) for s in range(1, 6)] + [("ac", s) for s in range(1, 6)])
        self.assertEqual((self.m.CUTOFF_S, self.m.HARD_CAP_S), (4 * 3600, 5 * 3600))
        self.assertEqual(self.m.DESIGNS, "CTRL,D1,D2,D3")

    def test_cutoff_hard_cap_and_failure(self):
        jobs = [("burgers", s) for s in range(5)]
        ledger, started = self.run_(jobs, {j: 100 for j in jobs}, cutoff=150)
        self.assertEqual([s[0] for s in started], jobs[:4])
        self.assertEqual({e["job"]: e["status"] for e in ledger}[jobs[4]], "not_started_cutoff")
        ledger, _ = self.run_(jobs[:3], {j: 1000 for j in jobs}, hard=300)
        by = {e["job"]: e["status"] for e in ledger}
        self.assertEqual([by[j] for j in jobs[:3]], ["killed_hard_cap", "killed_hard_cap", "not_started_cutoff"])
        ledger, _ = self.run_(jobs[:3], {j: 10 for j in jobs}, rcs={jobs[0]: 1})
        self.assertEqual(sorted(e["status"] for e in ledger), ["failed", "ok", "ok"])


@unittest.skipUnless(AVAILABLE, "falta el bundle del EXP-0024")
class TinyEndToEnd(unittest.TestCase):
    def _bundle(self):
        d = Path(tempfile.mkdtemp())
        for p in DATASET.iterdir():
            (d / p.name).write_bytes(p.read_bytes())
        return d

    def test_shorttest_tiny(self):
        bundle, out = self._bundle(), Path(tempfile.mkdtemp())
        env = {**os.environ, "SHORTTEST_TINY": "1", "BUNDLE_DIR": str(bundle), "OUT_DIR": str(out)}
        r = subprocess.run([sys.executable, str(SHORT)], env=env, capture_output=True, text=True, cwd=str(bundle))
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        summary = json.loads((out / "summary.json").read_text())
        self.assertTrue(all(v == 0 for v in summary["returncodes"].values()), summary["returncodes"])
        self.assertTrue(summary["finite"])
        self.assertIsNotNone(summary["estimate"])
        self.assertEqual(set(summary["report"]), {"burgers", "ac"})

    def test_block_tiny_allen_cahn_one_seed(self):
        m = _load(BLOCK, "exp0024_block_tiny")
        bundle, out = self._bundle(), Path(tempfile.mkdtemp())
        m.JOBS[:] = [("ac", 1)]
        old = {k: os.environ.get(k) for k in ("T1BLOCK_TINY", "BUNDLE_DIR", "OUT_DIR")}
        os.environ.update(T1BLOCK_TINY="1", BUNDLE_DIR=str(bundle), OUT_DIR=str(out))
        try:
            m.main()
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        ledger = json.loads((out / "ledger.json").read_text())
        self.assertTrue(ledger["complete"], ledger)
        files = sorted(p.name for p in out.glob("ac-*-seed1-*.json"))
        self.assertEqual(sorted(f.split("-")[1] for f in files), ["CTRL", "D1", "D2", "D3"])
        self.assertTrue((out / "designs.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
