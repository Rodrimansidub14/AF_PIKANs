import unittest

import numpy as np

from src.mechanisms import self_gated_sampler as sg


class SelfGatedSamplerTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.n = 2000
        self.s = np.abs(self.rng.normal(size=self.n)) + 0.01

    def test_probabilities_normalized_with_uniform_floor(self):
        r = self.s * (1 + 0.1 * self.rng.normal(size=self.n))
        for mode in ('gated', 'structure', 'residual'):
            p, lam = sg.probabilities(r, self.s, mode=mode)
            self.assertAlmostEqual(float(p.sum()), 1.0, places=12)
            self.assertTrue((p >= 0.5 / self.n - 1e-15).all())
            self.assertTrue(0.0 <= lam <= 1.0)

    def test_gate_follows_alignment(self):
        aligned = sg.gate(self.s * 3.0, self.s)
        anti = sg.gate(-np.log(self.s), self.s)
        self.assertGreater(aligned, 0.99)
        self.assertEqual(anti, 0.0)
        p_anti, lam = sg.probabilities(-np.log(self.s), self.s, mode='gated')
        p_str, _ = sg.probabilities(-np.log(self.s), self.s, mode='structure')
        self.assertEqual(lam, 0.0)
        np.testing.assert_allclose(p_anti, p_str)

    def test_mode_forcing(self):
        r = self.rng.random(self.n)
        self.assertEqual(sg.probabilities(r, self.s, mode='structure')[1], 0.0)
        self.assertEqual(sg.probabilities(r, self.s, mode='residual')[1], 1.0)

    def test_structure_mode_concentrates_where_structure_is_large(self):
        pts = np.stack([np.linspace(0, 1, self.n), np.zeros(self.n)], axis=1)
        out = sg.sample(pts, np.ones(self.n), self.s, 20000, np.random.default_rng(1), mode='structure')
        top = np.argsort(self.s)[-self.n // 20:]
        share = np.isin(out['indices'], top).mean()
        self.assertGreater(share, 0.05 * 1.5)

    def test_deterministic_given_generator(self):
        pts = self.rng.random((self.n, 2))
        a = sg.sample(pts, self.s, self.s, 500, np.random.default_rng(7))
        b = sg.sample(pts, self.s, self.s, 500, np.random.default_rng(7))
        np.testing.assert_array_equal(a['indices'], b['indices'])
        self.assertEqual(a['points'].shape, (500, 2))

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            sg.probabilities([1.0, 2.0], [1.0], mode='gated')
        with self.assertRaises(ValueError):
            sg.probabilities([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            sg.probabilities(self.s, self.s, mode='unknown')
        with self.assertRaises(ValueError):
            sg.sample(np.zeros((self.n, 2)), self.s, self.s, 0, np.random.default_rng(0))

    def test_constant_cues_fall_back_to_uniform(self):
        p, lam = sg.probabilities(np.zeros(100), np.zeros(100))
        np.testing.assert_allclose(p, np.full(100, 0.01))
        self.assertEqual(lam, 0.0)


if __name__ == '__main__':
    unittest.main()
