from __future__ import annotations

import unittest

from src.research_pipeline.pipeline import validate_manifest


class ManifestContractTests(unittest.TestCase):
    def test_requires_ontology_references_for_experiment_instances(self) -> None:
        manifest = {
            "project": {"ontology": "pikan-research"},
            "iteration": {
                "id": "H001",
                "protocol_phase": "phase_2_baseline_environment",
                "domain_level": "canonical_pde",
                "hypothesis_id": "H-CANON-BASE-001",
            },
            "hypothesis": {
                "statement": "Test",
                "expected_effect": "Test",
                "falsification_condition": "Test",
            },
            "problem": {},
            "model": {},
            "training": {},
            "focalization": {},
            "acceptance_gates": {"metrics": {"relative_l2_error": {"required": True}}},
            "artifacts": {},
        }

        errors = validate_manifest(manifest)

        self.assertIn("Missing iteration.experiment_id", errors)
        self.assertIn("Missing iteration.problem_id", errors)
        self.assertIn("Missing iteration.model_id", errors)
        self.assertIn("Missing iteration.focalization_id", errors)

    def test_current_manifest_uses_the_research_ontology(self) -> None:
        from pathlib import Path
        import yaml

        manifest = yaml.safe_load(
            Path("src/harness_loop/experiment_manifest.yaml").read_text(encoding="utf-8"),
        )

        self.assertEqual(validate_manifest(manifest), [])


if __name__ == "__main__":
    unittest.main()
