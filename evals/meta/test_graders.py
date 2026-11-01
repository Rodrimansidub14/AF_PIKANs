"""Meta-evals: verifican que los graders midan lo que dicen medir.

Un grader roto es peor que no tener grader: produce evidencia con apariencia
de rigor. Estas pruebas son Tier 0 y corren antes que cualquier experimento.

Se apoyan en el metodo de soluciones manufacturadas (MMS): se conoce la
solucion exacta, asi que se conoce de antemano el valor que cada grader debe
devolver.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.graders.core import (  # noqa: E402
    boundary_error_max,
    critical_region_residual_mean,
    grade_all,
    max_abs_error,
    physics_residual_mean,
    relative_l2,
    stability_score,
    wall_time_sec,
)
from evals.problems.heat_1d.reference import (  # noqa: E402
    ALPHA_DEFAULT,
    critical_mask,
    eval_grid,
    exact,
)


class TestReferenceSolution(unittest.TestCase):
    """La solucion de referencia debe satisfacer la PDE que dice resolver."""

    def test_initial_condition(self) -> None:
        x = np.linspace(0.0, 1.0, 101)
        t = np.zeros_like(x)
        np.testing.assert_allclose(exact(x, t), np.sin(np.pi * x), atol=1e-12)

    def test_dirichlet_boundaries_are_zero(self) -> None:
        t = np.linspace(0.0, 1.0, 51)
        for x_value in (0.0, 1.0):
            x = np.full_like(t, x_value)
            self.assertLess(float(np.abs(exact(x, t)).max()), 1e-12)

    def test_satisfies_pde_by_finite_differences(self) -> None:
        """MMS: el residuo de la solucion exacta debe ser numericamente cero.

        Valida el operador, no el modelo. Si esta prueba falla, cualquier
        `physics_residual_mean` calculado mas adelante carece de sentido.
        """
        h = 1e-4
        x = np.linspace(0.2, 0.8, 41)
        t = np.full_like(x, 0.3)
        u_t = (exact(x, t + h) - exact(x, t - h)) / (2 * h)
        u_xx = (exact(x + h, t) - 2 * exact(x, t) + exact(x - h, t)) / h**2
        residual = u_t - ALPHA_DEFAULT * u_xx
        self.assertLess(float(np.abs(residual).max()), 1e-6)


class TestGradersAreCorrect(unittest.TestCase):
    """Un grader debe dar 0 en la solucion exacta y crecer con el error."""

    def setUp(self) -> None:
        self.x, self.t = eval_grid(64, 32)
        self.reference = exact(self.x, self.t)

    def test_exact_prediction_scores_zero(self) -> None:
        self.assertAlmostEqual(relative_l2(self.reference, self.reference).value, 0.0, places=12)
        self.assertAlmostEqual(max_abs_error(self.reference, self.reference).value, 0.0, places=12)

    def test_error_is_monotone_in_perturbation(self) -> None:
        previous = -1.0
        for scale in (1e-4, 1e-3, 1e-2, 1e-1):
            prediction = self.reference + scale
            value = relative_l2(prediction, self.reference).value
            self.assertGreater(value, previous)
            previous = value

    def test_global_metric_hides_local_failure(self) -> None:
        """El caso que justifica tener una metrica de region critica.

        Un candidato con error concentrado en la frontera puede verse bien en
        el residuo global y ser inaceptable donde importa.
        """
        mask = critical_mask(self.x, self.t, "boundary_neighborhood")
        residual = np.zeros_like(self.reference)
        residual[mask] = 1.0

        global_score = physics_residual_mean(residual).value
        local_score = critical_region_residual_mean(
            residual, mask, region="boundary_neighborhood"
        ).value

        self.assertLess(global_score, 0.2)
        self.assertAlmostEqual(local_score, 1.0, places=12)
        self.assertGreater(local_score, global_score * 5)

    def test_boundary_error_uses_only_boundary_points(self) -> None:
        mask = (self.x < 1e-9) | (self.x > 1.0 - 1e-9)
        prediction = self.reference.copy()
        prediction[~mask] += 10.0
        self.assertAlmostEqual(
            boundary_error_max(prediction, self.reference, mask).value, 0.0, places=12
        )


class TestGradersFailLoudly(unittest.TestCase):
    """Ausencia de medicion no es cero. Invariante de la ontologia."""

    def test_nan_input_is_invalid_not_zero(self) -> None:
        nan = np.full(10, np.nan)
        score = relative_l2(nan, nan)
        self.assertIsNone(score.value)
        self.assertEqual(score.reason, "no_finite_points")

    def test_empty_critical_region_is_invalid_not_zero(self) -> None:
        residual = np.ones(10)
        score = critical_region_residual_mean(residual, np.zeros(10, dtype=bool))
        self.assertIsNone(score.value)
        self.assertEqual(score.reason, "empty_region")

    def test_mismatched_mask_is_invalid(self) -> None:
        score = critical_region_residual_mean(np.ones(10), np.ones(5, dtype=bool))
        self.assertIsNone(score.value)
        self.assertEqual(score.reason, "mask_shape_mismatch")

    def test_unmeasured_cost_is_invalid_not_zero(self) -> None:
        score = wall_time_sec(None)
        self.assertIsNone(score.value)
        self.assertEqual(score.reason, "not_measured")

    def test_zero_reference_norm_is_invalid(self) -> None:
        zeros = np.zeros(10)
        score = relative_l2(zeros, zeros)
        self.assertIsNone(score.value)
        self.assertEqual(score.reason, "reference_norm_zero")

    def test_unknown_grader_is_reported(self) -> None:
        card = grade_all(["no_such_metric"], prediction=np.ones(3), reference=np.ones(3))
        self.assertIsNone(card["no_such_metric"]["value"])
        self.assertEqual(card["no_such_metric"]["reason"], "unknown_grader")

    def test_missing_input_is_reported(self) -> None:
        card = grade_all(["relative_l2_error"])
        self.assertIsNone(card["relative_l2_error"]["value"])
        self.assertTrue(card["relative_l2_error"]["reason"].startswith("missing_input"))


class TestStabilityScore(unittest.TestCase):
    def test_clean_monotone_training_scores_high(self) -> None:
        history = np.geomspace(1.0, 1e-4, 50)
        self.assertGreater(stability_score(history).value, 0.99)

    def test_divergent_training_scores_low(self) -> None:
        history = np.concatenate([np.geomspace(1.0, 1e-4, 40), np.full(10, 1e3)])
        self.assertLess(stability_score(history).value, 0.01)

    def test_nan_epochs_reduce_the_score(self) -> None:
        history = np.geomspace(1.0, 1e-4, 50).copy()
        history[25:] = np.nan
        score = stability_score(history).value
        self.assertLess(score, 0.6)
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
