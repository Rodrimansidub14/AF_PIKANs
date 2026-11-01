"""El proposer no puede ver al verificador.

Estas pruebas hacen ejecutable la regla central del loop: un mecanismo que
puede leer su propio grader o la malla de evaluación puede optimizar la
métrica en vez del problema, y toda la evidencia derivada queda sin valor.

El chequeo es estático (AST) a propósito: no depende de que el import se
ejecute, así que detecta también imports dentro de funciones o condicionales.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MECHANISMS = ROOT / "src" / "mechanisms"

FORBIDDEN_ROOTS = {"evals"}


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


class TestMechanismsCannotImportEvals(unittest.TestCase):
    def test_no_mechanism_imports_the_verifier(self) -> None:
        offenders: list[str] = []
        for path in sorted(MECHANISMS.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for module in _imported_modules(tree):
                if module.split(".")[0] in FORBIDDEN_ROOTS:
                    offenders.append(f"{path.relative_to(ROOT)} importa '{module}'")
        self.assertEqual(
            offenders,
            [],
            "Un mecanismo no puede importar el verificador:\n" + "\n".join(offenders),
        )

    def test_guard_detects_a_planted_violation(self) -> None:
        """El guardia sirve solo si falla cuando debe."""
        tree = ast.parse("def f():\n    from evals.graders import core\n")
        modules = _imported_modules(tree)
        self.assertIn("evals.graders", modules)


class TestBaselineNone(unittest.TestCase):
    def setUp(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT))
        from src.mechanisms import BaselineNone, TrainingState

        self.BaselineNone = BaselineNone
        self.TrainingState = TrainingState

    def test_samples_inside_the_domain(self) -> None:
        mech = self.BaselineNone()
        out = mech.propose(500, self.TrainingState(epoch=0), np.random.default_rng(11))
        self.assertEqual(out.collocation.shape, (500, 2))
        self.assertGreaterEqual(out.collocation.min(), 0.0)
        self.assertLessEqual(out.collocation.max(), 1.0)

    def test_is_deterministic_given_a_seed(self) -> None:
        a = self.BaselineNone().propose(64, self.TrainingState(epoch=0), np.random.default_rng(11))
        b = self.BaselineNone().propose(64, self.TrainingState(epoch=0), np.random.default_rng(11))
        np.testing.assert_array_equal(a.collocation, b.collocation)

    def test_different_seeds_give_different_points(self) -> None:
        a = self.BaselineNone().propose(64, self.TrainingState(epoch=0), np.random.default_rng(11))
        b = self.BaselineNone().propose(64, self.TrainingState(epoch=0), np.random.default_rng(17))
        self.assertFalse(np.array_equal(a.collocation, b.collocation))

    def test_fixed_grid_is_reused_across_epochs(self) -> None:
        mech = self.BaselineNone(resample_every=0)
        rng = np.random.default_rng(11)
        first = mech.propose(32, self.TrainingState(epoch=0), rng)
        later = mech.propose(32, self.TrainingState(epoch=50), rng)
        np.testing.assert_array_equal(first.collocation, later.collocation)
        self.assertEqual(later.diagnostics["resampled"], 0.0)

    def test_out_of_domain_sampling_raises(self) -> None:
        from src.mechanisms.base import validate_in_domain

        bad = np.array([[0.5, 1.5]])
        with self.assertRaises(ValueError) as ctx:
            validate_in_domain(bad, {"x": (0.0, 1.0), "t": (0.0, 1.0)})
        self.assertIn("fuera del dominio", str(ctx.exception))

    def test_non_finite_sampling_raises(self) -> None:
        from src.mechanisms.base import validate_in_domain

        with self.assertRaises(ValueError):
            validate_in_domain(
                np.array([[0.5, np.nan]]), {"x": (0.0, 1.0), "t": (0.0, 1.0)}
            )


if __name__ == "__main__":
    unittest.main()
