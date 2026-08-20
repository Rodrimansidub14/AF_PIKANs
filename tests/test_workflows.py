from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.graph_engineering.workflows import (
    project_workflow_registry,
    validate_workflow_registry,
)
from src.graph_engineering.contracts import load_contracts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WorkflowRegistryTests(unittest.TestCase):
    def test_projects_workflows_and_references_existing_roles(self) -> None:
        contracts = load_contracts(PROJECT_ROOT)
        projection = project_workflow_registry(PROJECT_ROOT, contracts["agent"])

        self.assertEqual(projection["default"], "literature")
        self.assertEqual(projection["ambiguity_policy"], "conservative")
        self.assertEqual(
            {item["id"] for item in projection["items"]},
            {"literature", "hypothesis", "notebook", "run", "evaluation", "synthesis"},
        )
        self.assertEqual(
            {item["role"] for item in projection["items"]},
            {role["id"] for role in contracts["agent"]["role"]},
        )
        self.assertNotIn(str(PROJECT_ROOT), json.dumps(projection))

    def test_rejects_workflow_with_unknown_role(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / "workflow"
            registry.mkdir()
            (registry / "index.toml").write_text(
                '[meta]\nschema_version = 1\ndefault = "bad"\nambiguity_policy = "conservative"\n'
                '\n[[workflow]]\nid = "bad"\nrole = "missing_role"\npriority = 1\n'
                'triggers = ["bad"]\nexcludes = []\nstages = ["inspect"]\ngate_ids = []\n',
                encoding="utf-8",
            )

            result = validate_workflow_registry(
                root,
                {"role": [{"id": "literature_scout"}]},
            )

            self.assertTrue(any("missing_role" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
