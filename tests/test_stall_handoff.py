import ast
import unittest
from pathlib import Path

import numpy as np

from src.mechanisms import stall_handoff as sh

MODULE = Path(sh.__file__)


def curve(n_steps=20000, every=100, plateau_at=None, rate=1e-3, floor=1e-9, noise=0.0, seed=0, spikes=()):
    """log L decays at `rate` per step from 1; after `plateau_at` it decays 100x slower."""
    rng = np.random.default_rng(seed)
    steps = np.arange(every, n_steps + 1, every)
    logl = np.zeros(steps.size)
    for i, s in enumerate(steps):
        t = s if plateau_at is None else min(s, plateau_at) + 0.01 * max(0, s - plateau_at)
        logl[i] = -rate * t
    logl = np.maximum(logl, np.log(floor))
    logl = logl + noise * rng.normal(size=steps.size)
    for s, size in spikes:
        logl[np.argmin(np.abs(steps - s))] += size
    return steps, np.exp(logl)


class StallHandoffTests(unittest.TestCase):
    def test_fires_shortly_after_a_stall(self):
        steps, loss = curve(plateau_at=8000, rate=1e-3)
        out = sh.handoff_step(steps, loss, k_max=20000)
        self.assertTrue(out['fired'])
        self.assertEqual(out['reason'], 'stall')
        # a stall at 8000 needs at least `persistence` blocks after it to confirm
        self.assertGreaterEqual(out['k_t'], 8000 + 2 * 500)
        self.assertLessEqual(out['k_t'], 8000 + 5 * 500)

    def test_steady_decay_never_fires_and_falls_back_to_k_max(self):
        steps, loss = curve(plateau_at=None)
        out = sh.handoff_step(steps, loss, k_max=20000)
        self.assertFalse(out['fired'])
        self.assertEqual(out['k_t'], 20000)
        self.assertEqual(out['reason'], 'no_stall')

    def test_never_later_than_k_max(self):
        steps, loss = curve(plateau_at=18000)
        for k_max in (9000, 15000, 20000):
            out = sh.handoff_step(steps, loss, k_max=k_max)
            self.assertLessEqual(out['k_t'], k_max)

    def test_respects_k_min(self):
        # stalled from the very start of the allowed window: still cannot fire before k_min
        steps, loss = curve(plateau_at=2500)
        out = sh.handoff_step(steps, loss, k_max=20000)
        self.assertTrue(out['fired'])
        self.assertGreaterEqual(out['k_t'], 4000 + 2 * 500)

    def test_single_spike_does_not_fire(self):
        steps, loss = curve(plateau_at=None, spikes=[(9000, 3.0)])
        out = sh.handoff_step(steps, loss, k_max=20000)
        self.assertFalse(out['fired'])

    def test_persistence_is_required(self):
        # a stall shorter than three blocks must not fire; a longer one must
        steps = np.arange(100, 20001, 100)
        logl = -1e-3 * steps.astype(float)
        short = logl.copy()
        short[(steps > 9000) & (steps <= 10000)] = short[steps == 9000][0]
        out = sh.handoff_step(steps, np.exp(short), k_max=20000)
        self.assertFalse(out['fired'])
        longer = logl.copy()
        mask = (steps > 9000) & (steps <= 11000)
        longer[mask] = longer[steps == 9000][0]
        out = sh.handoff_step(steps, np.exp(longer), k_max=20000)
        self.assertTrue(out['fired'])

    def test_robust_to_moderate_noise(self):
        fired = 0
        for seed in range(20):
            steps, loss = curve(plateau_at=None, noise=0.03, seed=seed)
            fired += sh.handoff_step(steps, loss, k_max=20000)['fired']
        self.assertEqual(fired, 0, 'steady decay with 3% log-noise should not trigger')

    def test_scale_invariant(self):
        steps, loss = curve(plateau_at=8000, noise=0.01)
        a = sh.handoff_step(steps, loss, k_max=20000)['k_t']
        b = sh.handoff_step(steps, loss * 1e7, k_max=20000)['k_t']
        self.assertEqual(a, b)

    def test_causal_no_look_ahead(self):
        steps, loss = curve(plateau_at=8000, noise=0.01, seed=3)
        full = sh.handoff_step(steps, loss, k_max=20000)
        self.assertTrue(full['fired'])
        cut = steps <= full['k_t']
        prefix = sh.handoff_step(steps[cut], loss[cut], k_max=20000)
        self.assertEqual(prefix['k_t'], full['k_t'])
        # changing the future must not move the decision
        loss2 = loss.copy()
        loss2[~cut] *= 100.0
        self.assertEqual(sh.handoff_step(steps, loss2, k_max=20000)['k_t'], full['k_t'])

    def test_sensitivity_is_monotone_in_rho_star(self):
        steps, loss = curve(plateau_at=8000, noise=0.01, seed=1)
        ks = [sh.handoff_step(steps, loss, k_max=20000, rho_star=r)['k_t'] for r in (0.05, 0.10, 0.20, 0.50)]
        self.assertEqual(ks, sorted(ks, reverse=True), 'a larger rho_star must not fire later')

    def test_no_reference_rate_falls_back(self):
        steps = np.arange(100, 20001, 100)
        out = sh.handoff_step(steps, np.ones(steps.size), k_max=20000)
        self.assertFalse(out['fired'])
        self.assertEqual(out['k_t'], 20000)
        self.assertEqual(out['reason'], 'no_reference_rate')

    def test_short_history_falls_back(self):
        steps = np.arange(100, 2501, 100)
        out = sh.handoff_step(steps, np.exp(-1e-3 * steps), k_max=20000)
        self.assertEqual(out['reason'], 'history_too_short')
        self.assertEqual(out['k_t'], 20000)

    def test_block_rates_use_complete_blocks_only(self):
        steps = np.arange(100, 2301, 100)
        ends, r = sh.block_rates(steps, np.exp(-1e-3 * steps), block=500)
        self.assertEqual(list(ends), [500, 1000, 1500, 2000])
        self.assertTrue(np.isnan(r[0]))
        np.testing.assert_allclose(r[1:], 1e-3, rtol=1e-9)

    def test_rejects_bad_inputs(self):
        steps = np.arange(100, 20001, 100)
        good = np.exp(-1e-3 * steps)
        with self.assertRaises(ValueError):
            sh.handoff_step(steps, -good, k_max=20000)
        with self.assertRaises(ValueError):
            sh.handoff_step(steps[::-1], good, k_max=20000)
        with self.assertRaises(ValueError):
            sh.handoff_step(steps, good, k_max=3000)
        with self.assertRaises(ValueError):
            sh.handoff_step(steps, good, k_max=20000, rho_star=1.5)
        bad = good.copy()
        bad[10] = np.nan
        with self.assertRaises(ValueError):
            sh.handoff_step(steps, bad, k_max=20000)

    def test_deterministic(self):
        steps, loss = curve(plateau_at=8000, noise=0.02, seed=5)
        a = sh.handoff_step(steps, loss, k_max=20000)
        b = sh.handoff_step(steps, loss, k_max=20000)
        self.assertEqual(a['k_t'], b['k_t'])
        np.testing.assert_array_equal(a['rho'], b['rho'])


class StallHandoffIsolationTests(unittest.TestCase):
    def test_imports_only_numpy_and_stdlib(self):
        tree = ast.parse(MODULE.read_text(encoding='utf-8'))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split('.')[0])
        self.assertLessEqual(mods, {'math', 'numpy'})
        self.assertNotIn('evals', mods)

    def test_signature_takes_losses_not_solutions(self):
        import inspect

        params = list(inspect.signature(sh.handoff_step).parameters)
        self.assertEqual(params[:3], ['steps', 'losses', 'k_max'])


if __name__ == '__main__':
    unittest.main()
