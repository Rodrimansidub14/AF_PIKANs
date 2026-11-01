import unittest

import numpy as np

from evals.problems.burgers_pinnacle_1d import reference


class CanonicalBurgersReferenceTests(unittest.TestCase):
    def test_initial_and_boundary_conditions(self):
        x = np.linspace(-1.0, 1.0, 401)
        np.testing.assert_allclose(reference.exact(x, np.zeros_like(x)), -np.sin(np.pi * x), atol=1e-14)
        t = np.linspace(0.0, 1.0, 101)
        np.testing.assert_allclose(reference.exact(np.full_like(t, -1.0), t), 0.0, atol=1e-14)
        np.testing.assert_allclose(reference.exact(np.full_like(t, 1.0), t), 0.0, atol=1e-14)

    def test_cole_hopf_quadrature_converges_and_preserves_odd_symmetry(self):
        x, t = np.meshgrid(np.linspace(-1.0, 1.0, 101), np.linspace(0.0, 1.0, 41), indexing='ij')
        coarse = reference.exact(x, t, quadrature_order=64)
        fine = reference.exact(x, t, quadrature_order=128)
        self.assertLess(float(np.max(np.abs(coarse - fine))), 2e-10)
        np.testing.assert_allclose(fine, -fine[::-1], atol=2e-13, rtol=0)
        self.assertTrue(np.isfinite(fine).all())
        self.assertLessEqual(float(np.max(np.abs(fine))), 1.0 + 1e-12)

    def test_reference_satisfies_viscous_burgers_under_refined_differences(self):
        x, t = np.meshgrid(np.linspace(-0.9, 0.9, 31), np.linspace(0.05, 0.95, 13), indexing='ij')
        h = 5e-5
        u = reference.exact(x, t)
        ux = (reference.exact(x + h, t) - reference.exact(x - h, t)) / (2.0 * h)
        uxx = (reference.exact(x + h, t) - 2.0 * u + reference.exact(x - h, t)) / (h * h)
        ut = (reference.exact(x, t + h) - reference.exact(x, t - h)) / (2.0 * h)
        residual = reference.residual(u, ut, ux, uxx)
        self.assertLess(float(np.max(np.abs(residual))), 5e-5)


if __name__ == '__main__':
    unittest.main()
