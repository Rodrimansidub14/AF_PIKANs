"""Planificador de la etapa A del EXP-0016 (topes, orden, fallos) y una corrida minima de punta a punta en CPU."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff"
SCRIPT = EXP / "kaggle/stage_a/stage_a_gpu.py"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"


def _load():
    spec = importlib.util.spec_from_file_location("exp0016_stage_a", str(SCRIPT))
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
        if self.killed:
            return -9
        return self.rc if self.clock() >= self.end else None

    def kill(self):
        self.killed = True


@unittest.skipUnless(SCRIPT.exists(), "falta el orquestador")
class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def _run(self, jobs, durations, n_workers=2, cutoff=1e9, hard=1e9, rcs=None):
        clock = Clock()
        started = []

        def launch(job, slot):
            started.append((job, slot, clock()))
            return FakeProc(clock, durations[job], (rcs or {}).get(job, 0))

        ledger = self.m.run_jobs(jobs, n_workers, cutoff, hard, launch, poll=1.0, clock=clock, sleep=clock.sleep)
        return ledger, started

    def test_default_queue_is_burgers_first_then_allen_cahn_and_has_no_heat(self):
        jobs = self.m.JOBS
        self.assertEqual([j for j in jobs if j[0] == "burgers"], [("burgers", s) for s in (301, 302, 303, 304, 305)])
        self.assertEqual([j for j in jobs if j[0] == "ac"], [("ac", s) for s in (601, 602, 603)])
        self.assertEqual({j[0] for j in jobs}, {"burgers", "ac"})
        self.assertEqual([j[0] for j in jobs], ["burgers"] * 5 + ["ac"] * 3)

    def test_jobs_start_in_queue_order_on_two_slots(self):
        jobs = [("burgers", s) for s in (1, 2, 3, 4)]
        ledger, started = self._run(jobs, {j: 10 + i for i, j in enumerate(jobs)})
        self.assertEqual([s[0] for s in started], jobs)
        self.assertEqual({s[1] for s in started}, {0, 1})
        self.assertTrue(all(e["status"] == "ok" for e in ledger))
        self.assertEqual(len(ledger), 4)

    def test_no_new_job_after_the_cutoff_but_running_ones_finish(self):
        jobs = [("burgers", s) for s in (1, 2, 3, 4, 5)]
        ledger, started = self._run(jobs, {j: 100 for j in jobs}, cutoff=150)
        by = {e["job"]: e["status"] for e in ledger}
        self.assertEqual([s[0] for s in started], jobs[:4])         # 2 al inicio, 2 cuando terminan (t = 100 < 150)
        self.assertEqual(by[jobs[4]], "not_started_cutoff")
        self.assertTrue(all(by[j] == "ok" for j in jobs[:4]))

    def test_hard_cap_kills_running_jobs_and_skips_the_rest(self):
        jobs = [("burgers", s) for s in (1, 2, 3)]
        ledger, _ = self._run(jobs, {j: 1000 for j in jobs}, hard=300)
        by = {e["job"]: e["status"] for e in ledger}
        self.assertEqual(by[jobs[0]], "killed_hard_cap")
        self.assertEqual(by[jobs[1]], "killed_hard_cap")
        self.assertEqual(by[jobs[2]], "not_started_cutoff")

    def test_a_failed_job_is_recorded_and_does_not_stop_the_others(self):
        jobs = [("burgers", s) for s in (1, 2, 3)]
        ledger, started = self._run(jobs, {j: 10 for j in jobs}, rcs={jobs[0]: 1})
        by = {e["job"]: e["status"] for e in ledger}
        self.assertEqual(by[jobs[0]], "failed")
        self.assertEqual(by[jobs[1]], "ok")
        self.assertEqual(by[jobs[2]], "ok")

    def test_single_worker_runs_jobs_sequentially(self):
        jobs = [("ac", s) for s in (1, 2, 3)]
        ledger, started = self._run(jobs, {j: 10 for j in jobs}, n_workers=1)
        self.assertEqual([s[1] for s in started], [0, 0, 0])
        starts = [s[2] for s in started]
        self.assertEqual(starts, sorted(starts))
        self.assertGreaterEqual(starts[1], starts[0] + 10)

    def test_every_job_gets_exactly_one_ledger_entry(self):
        jobs = [("burgers", s) for s in range(6)]
        ledger, _ = self._run(jobs, {j: 50 for j in jobs}, cutoff=60)
        self.assertEqual(sorted(e["job"] for e in ledger), sorted(jobs))


@unittest.skipUnless(SCRIPT.exists() and (BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz").exists(),
                     "faltan las referencias de la campana")
class TinyEndToEndTest(unittest.TestCase):
    def test_tiny_run_produces_one_json_per_seed_and_a_complete_ledger(self):
        m = _load()
        bundle = Path(tempfile.mkdtemp())
        for src in (EXP / "sweep.py", ROOT / "src/mechanisms/stall_handoff.py", ROOT / "src/mechanisms/settling_focus.py",
                    BL / "burgers-close-20261003/pikan_campaign.py", BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz",
                    BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat"):
            (bundle / src.name).write_bytes(src.read_bytes())
        out = Path(tempfile.mkdtemp())
        m.JOBS[:] = [("burgers", 301), ("burgers", 302), ("ac", 601)]
        old = {k: os.environ.get(k) for k in ("STAGEA_TINY", "BUNDLE_DIR", "OUT_DIR")}
        os.environ.update(STAGEA_TINY="1", BUNDLE_DIR=str(bundle), OUT_DIR=str(out))
        try:
            m.main()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        ledger = json.loads((out / "ledger.json").read_text())
        self.assertTrue(ledger["complete"], ledger)
        self.assertEqual(sorted(ledger["finished_seeds"]), ["ac-seed601.json", "burgers-seed301.json", "burgers-seed302.json"])
        res = json.loads((out / "burgers-seed301.json").read_text())
        self.assertEqual(sorted(res["runs"]), ["0.25", "0.4", "0.6", "0.8", "1.0"])
        self.assertTrue((out / "burgers-seed301-adam-trace.npz").exists())
        self.assertTrue((out / "logs/burgers-301.log").exists())


if __name__ == "__main__":
    unittest.main()
