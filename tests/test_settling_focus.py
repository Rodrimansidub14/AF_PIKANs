import ast
import unittest
from pathlib import Path

import numpy as np

from src.mechanisms import settling_focus as sf

MODULE = Path(sf.__file__)
STEPS = np.arange(0, 20001, 500)


def trajectory(taus, steps=STEPS, outputs=3, seed=0, oscillate=(), dead=(), scale=1.0):
    """c[t, o, d, g] = c_inf + (c0 - c_inf) exp(-t / tau_dg); `oscillate` bases keep a constant-amplitude wobble; `dead` bases are zero."""
    rng = np.random.default_rng(seed)
    taus = np.asarray(taus, dtype=float)
    D, G = taus.shape
    c_inf = rng.normal(size=(outputs, D, G)) + 2.0
    c0 = c_inf + rng.normal(size=(outputs, D, G))
    t = steps.reshape(-1, 1, 1, 1).astype(float)
    c = c_inf + (c0 - c_inf) * np.exp(-t / taus.reshape(1, 1, D, G))
    for (d, g) in oscillate:
        c[:, :, d, g] = c_inf[:, d, g] + 0.5 * np.sin(2 * np.pi * steps.reshape(-1, 1) / 3000.0 + np.arange(outputs).reshape(1, -1))
    for (d, g) in dead:
        c[:, :, d, g] = 0.0
    return steps, c * scale


class SettlingSignalTests(unittest.TestCase):
    def test_fast_bases_settle_before_slow_ones_and_very_slow_never(self):
        taus = np.array([[300.0, 1500.0, 6000.0, 1e7]])
        steps, c = trajectory(taus)
        ends, S, rho = sf.settling_rates(steps, c)
        settled = sf.settled_mask(rho)
        first = [int(ends[np.argmax(settled[:, 0, g])]) if settled[:, 0, g].any() else None for g in range(4)]
        self.assertIsNotNone(first[0])
        self.assertIsNotNone(first[1])
        self.assertLess(first[0], first[1])
        self.assertIsNone(first[3], 'a base that never decays must not settle')

    def test_oscillating_base_never_settles(self):
        steps, c = trajectory(np.full((1, 3), 300.0), oscillate=[(0, 1)])
        _, _, rho = sf.settling_rates(steps, c)
        settled = sf.settled_mask(rho)
        self.assertFalse(settled[:, 0, 1].any())
        self.assertTrue(settled[-1, 0, 0] and settled[-1, 0, 2])

    def test_dead_base_has_rho_zero_and_counts_as_settled(self):
        steps, c = trajectory(np.full((1, 3), 300.0), dead=[(0, 0)])
        _, S, rho = sf.settling_rates(steps, c)
        self.assertTrue((S[:, 0, 0] == 0).all())
        self.assertTrue((rho[:, 0, 0] == 0).all())
        self.assertTrue(sf.settled_mask(rho)[-1, 0, 0])

    def test_scale_invariant_in_the_coefficients(self):
        taus = np.array([[300.0, 1500.0, 6000.0]])
        _, c = trajectory(taus)
        _, c2 = trajectory(taus, scale=1e3)
        np.testing.assert_allclose(sf.settling_rates(STEPS, c)[2], sf.settling_rates(STEPS, c2)[2], rtol=1e-9, atol=1e-9)  # atol: rho ~ 1e-13 is rounding noise

    def test_persistence_is_required(self):
        rho = np.ones((8, 1, 1))
        rho[3, 0, 0] = 0.01                                   # one isolated dip
        self.assertFalse(sf.settled_mask(rho, persistence=3).any())
        rho[3:6, 0, 0] = 0.01                                 # three in a row
        self.assertTrue(sf.settled_mask(rho, persistence=3)[5, 0, 0])
        self.assertFalse(sf.settled_mask(rho, persistence=3)[4, 0, 0])

    def test_unequal_snapshot_gaps_are_rejected(self):
        steps, c = trajectory(np.full((1, 2), 500.0))
        with self.assertRaises(ValueError):
            sf.settling_rates(np.r_[steps[:-1], steps[-1] + 1], c)

    def test_bad_inputs(self):
        steps, c = trajectory(np.full((1, 2), 500.0))
        with self.assertRaises(ValueError):
            sf.settling_rates(steps, c[..., 0])               # wrong rank
        bad = c.copy()
        bad[3, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            sf.settling_rates(steps, bad)
        with self.assertRaises(ValueError):
            sf.settling_step(steps, c, k_max=3000)            # k_max < k_min
        with self.assertRaises(ValueError):
            sf.settling_step(steps, c, k_max=20000, k_min=3000)   # k_min < ref_hi
        with self.assertRaises(ValueError):
            sf.settling_step(steps, c, k_max=20000, q_star=1.5)


class SettlingTriggerTests(unittest.TestCase):
    def _taus(self, n_fast, n_slow):
        return np.array([[400.0] * n_fast + [1e7] * n_slow])

    def test_fires_when_the_fraction_reaches_q_star(self):
        steps, c = trajectory(self._taus(8, 2))
        out = sf.settling_step(steps, c, k_max=20000)
        self.assertTrue(out['fired'])
        self.assertEqual(out['reason'], 'settled')
        self.assertGreaterEqual(out['k_t'], 4000)
        self.assertAlmostEqual(out['fraction'][list(out['ends']).index(out['k_t'])], 0.8)

    def test_does_not_fire_below_q_star_and_falls_back_to_k_max(self):
        steps, c = trajectory(self._taus(7, 3))
        out = sf.settling_step(steps, c, k_max=20000)
        self.assertFalse(out['fired'])
        self.assertEqual(out['k_t'], 20000)
        self.assertEqual(out['reason'], 'not_settled')
        self.assertTrue(sf.settling_step(steps, c, k_max=20000, q_star=0.6)['fired'])

    def test_never_later_than_k_max_and_never_before_k_min(self):
        steps, c = trajectory(self._taus(10, 0))
        out = sf.settling_step(steps, c, k_max=20000)
        self.assertGreaterEqual(out['k_t'], 4000)
        for k_max in (9000, 15000, 20000):
            self.assertLessEqual(sf.settling_step(steps, c, k_max=k_max)['k_t'], k_max)

    def test_causal_no_look_ahead(self):
        steps, c = trajectory(np.array([[400.0, 400.0, 900.0, 1500.0, 1e7]]))
        full = sf.settling_step(steps, c, k_max=20000, q_star=0.6)
        self.assertTrue(full['fired'])
        cut = steps <= full['k_t']
        self.assertEqual(sf.settling_step(steps[cut], c[cut], k_max=20000, q_star=0.6)['k_t'], full['k_t'])
        c2 = c.copy()
        c2[~cut] *= 50.0                                       # change the future
        self.assertEqual(sf.settling_step(steps, c2, k_max=20000, q_star=0.6)['k_t'], full['k_t'])

    def test_short_history_falls_back(self):
        steps = np.arange(0, 2501, 500)   # only one window ends inside the reference range
        _, c = trajectory(self._taus(4, 0), steps=steps)
        out = sf.settling_step(steps, c, k_max=20000)
        self.assertEqual(out['reason'], 'history_too_short')
        self.assertEqual(out['k_t'], 20000)

    def test_sensitivity_to_q_star_is_monotone(self):
        steps, c = trajectory(np.array([[300.0, 500.0, 800.0, 1200.0, 2000.0, 3500.0, 6000.0, 1e7]]))
        ks = [sf.settling_step(steps, c, k_max=20000, q_star=q)['k_t'] for q in (0.3, 0.5, 0.6, 0.8)]
        self.assertEqual(ks, sorted(ks), 'a larger q_star must not fire earlier')

    def test_deterministic(self):
        steps, c = trajectory(self._taus(8, 2), seed=3)
        a, b = sf.settling_step(steps, c, k_max=20000), sf.settling_step(steps, c, k_max=20000)
        self.assertEqual(a['k_t'], b['k_t'])
        np.testing.assert_array_equal(a['rho'], b['rho'])


class UnsettledMapTests(unittest.TestCase):
    centers = np.linspace(-1.0, 1.0, 5)
    width = 1.75 * 0.5

    def test_constant_rho_gives_a_constant_map(self):
        x = np.linspace(-1, 1, 40).reshape(-1, 1)
        m = sf.unsettled_map(x, np.full((1, 5), 0.37), self.centers, self.width)
        np.testing.assert_allclose(m, 0.37, rtol=1e-12)

    def test_map_is_peaked_near_the_unsettled_basis(self):
        rho = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
        x = np.array([[-1.0], [0.0], [1.0]])
        m = sf.unsettled_map(x, rho, self.centers, self.width)
        self.assertGreater(m[1], m[0])
        self.assertGreater(m[1], m[2])
        self.assertAlmostEqual(m[0], m[2], places=12)          # symmetric geometry

    def test_map_is_a_convex_combination_of_rho(self):
        rng = np.random.default_rng(0)
        rho = rng.uniform(0.0, 2.0, size=(2, 5))
        x = rng.uniform(-1, 1, size=(300, 2))
        m = sf.unsettled_map(x, rho, self.centers, self.width)
        self.assertGreaterEqual(m.min(), rho.min() - 1e-12)
        self.assertLessEqual(m.max(), rho.max() + 1e-12)

    def test_each_dimension_marks_its_own_strip(self):
        # only the t-dimension basis at the centre is unsettled: the map must vary with t and (much less) with x
        rho = np.zeros((2, 5))
        rho[1, 2] = 1.0
        t = np.linspace(-1, 1, 21)
        along_t = sf.unsettled_map(np.stack([np.zeros_like(t), t], 1), rho, self.centers, self.width)
        along_x = sf.unsettled_map(np.stack([t, np.zeros_like(t)], 1), rho, self.centers, self.width)
        self.assertGreater(np.ptp(along_t), 0.1)
        self.assertLess(np.ptp(along_x), np.ptp(along_t))

    def test_shape_and_finiteness_checks(self):
        with self.assertRaises(ValueError):
            sf.unsettled_map(np.zeros((3, 2)), np.zeros((1, 5)), self.centers, self.width)
        with self.assertRaises(ValueError):
            sf.unsettled_map(np.zeros((3, 1)), np.full((1, 5), np.nan), self.centers, self.width)
        with self.assertRaises(ValueError):
            sf.unsettled_map(np.zeros((3, 1)), np.zeros((1, 5)), self.centers, 0.0)


class FocusWeightTests(unittest.TestCase):
    def test_weights_are_positive_with_mean_one_and_follow_the_signal(self):
        v = np.array([0.0, 1.0, 2.0, 5.0])
        w = sf.focus_weights(v, alpha=1.0)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=12)
        self.assertTrue((w > 0).all())
        self.assertEqual(list(np.argsort(w)), [0, 1, 2, 3])

    def test_constant_zero_and_alpha_zero_give_uniform_weights(self):
        np.testing.assert_allclose(sf.focus_weights(np.full(10, 0.4)), 1.0)
        np.testing.assert_allclose(sf.focus_weights(np.zeros(10)), 1.0)
        np.testing.assert_allclose(sf.focus_weights(np.arange(10.0), alpha=0.0), 1.0)

    def test_unnormalised_form_matches_the_preregistered_formula(self):
        v = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(sf.focus_weights(v, alpha=1.0, normalize=False), 1.0 + v / v.mean())

    def test_scale_invariant_in_the_signal(self):
        v = np.array([0.2, 1.0, 3.0])
        np.testing.assert_allclose(sf.focus_weights(v), sf.focus_weights(v * 17.0), rtol=1e-12)

    def test_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            sf.focus_weights(np.array([1.0, -1.0]))
        with self.assertRaises(ValueError):
            sf.focus_weights(np.array([1.0, np.nan]))
        with self.assertRaises(ValueError):
            sf.focus_weights(np.ones(3), alpha=-1.0)


class SettlingFocusIsolationTests(unittest.TestCase):
    def test_imports_only_numpy_and_math(self):
        tree = ast.parse(MODULE.read_text(encoding='utf-8'))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split('.')[0])
        self.assertLessEqual(mods, {'math', 'numpy'})
        self.assertNotIn('evals', mods)


if __name__ == '__main__':
    unittest.main()
