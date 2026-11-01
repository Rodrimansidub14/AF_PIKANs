import unittest
import numpy as np
from src.mechanisms.rad import residual_probabilities, sample_rad


class RADTests(unittest.TestCase):
    def test_zero_residual_is_uniform(self):
        np.testing.assert_allclose(residual_probabilities(np.zeros(32)), 1/32)

    def test_quadratic_distribution_and_uniform_floor(self):
        p = residual_probabilities(np.array([0., 1., 2.]), power=2, uniform_fraction=.5)
        np.testing.assert_allclose(p, .5/3 + .5*np.array([0.,1.,4.])/5)
        np.testing.assert_allclose(residual_probabilities(np.array([0.,1.,2.])*1e200), p)

    def test_invalid_residual_rejected(self):
        for r in [np.array([]), np.array([np.nan]), np.array([np.inf])]:
            with self.assertRaises(ValueError): residual_probabilities(r)
        for mix in [-.1, 1.1]:
            with self.assertRaises(ValueError): residual_probabilities(np.ones(3), uniform_fraction=mix)

    def test_reproducible_fixed_size_and_support(self):
        candidates=np.column_stack((np.linspace(-1,1,64),np.linspace(0,1,64)))
        residual=np.arange(64.)
        a=sample_rad(candidates,residual,128,np.random.default_rng(5))
        b=sample_rad(candidates,residual,128,np.random.default_rng(5))
        np.testing.assert_array_equal(a['indices'],b['indices'])
        self.assertEqual(a['points'].shape,(128,2))
        np.testing.assert_array_equal(a['points'],candidates[a['indices']])
        self.assertTrue(np.isfinite(a['probabilities']).all())

    def test_sampling_concentrates_on_hotspots(self):
        points=np.column_stack((np.arange(100.),np.zeros(100)))
        residual=np.r_[np.zeros(90),np.ones(10)]
        result=sample_rad(points,residual,20000,np.random.default_rng(7))
        self.assertAlmostEqual(np.mean(result['indices']>=90),.55,delta=.015)

