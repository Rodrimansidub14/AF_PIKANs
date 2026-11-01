"""Behavior checks for the isolated Allen-Cahn candidate reference builder."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "allen_cahn_bdf_reference.py"


class AllenCahnReferenceChecks(unittest.TestCase):
    def run_builder(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=HERE,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )

    def test_operator_sign_matches_hand_derived_values(self) -> None:
        completed = self.run_builder("--self-check")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        checks = json.loads(completed.stdout)
        self.assertAlmostEqual(checks["quadratic_laplacian_center"], 2.0, places=10)
        self.assertAlmostEqual(checks["diffusion_rhs_for_quadratic"], 0.002, places=12)
        self.assertAlmostEqual(checks["reaction_rhs_at_quarter"], 1.171875, places=12)
        self.assertTrue(checks["operator_signs_ok"])

    def test_smoke_solution_succeeds_and_preserves_traces(self) -> None:
        with tempfile.TemporaryDirectory(prefix="_smoke-", dir=HERE) as tmp:
            completed = self.run_builder("--smoke", "--output", tmp)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["study_status"], "candidate_only")
        checks = metrics["sanity_checks"]
        self.assertTrue(checks["all_solvers_succeeded"])
        self.assertTrue(checks["all_values_finite"])
        self.assertEqual(checks["max_boundary_abs_error"], 0.0)
        self.assertLessEqual(checks["max_initial_abs_error"], 1e-14)
        self.assertTrue(checks["energy_monotone_within_tolerance"])


if __name__ == "__main__":
    unittest.main()
