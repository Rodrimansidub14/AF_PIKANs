from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.graph_engineering import collect_vault, extract_wikilinks, parse_frontmatter, resolve_links
from src.graph_engineering.build import build_graph, render_report
from src.graph_engineering.contracts import (
    audit_notebooks,
    load_contracts,
    validate_contracts,
    validate_notebook_metadata,
)
from src.graph_engineering.context import build_context_pack


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GraphEngineeringTests(unittest.TestCase):
    def test_parses_tags_and_frontmatter(self) -> None:
        text = """---
type: paper
title: \"PINNs\"
tags: #found #pinn #sciml
---

# PINNs
"""

        result = parse_frontmatter(text)

        self.assertEqual(result["type"], "paper")
        self.assertEqual(result["title"], "PINNs")
        self.assertEqual(result["tags"], ["found", "pinn", "sciml"])

    def test_extracts_wikilinks_without_alias_or_heading(self) -> None:
        text = "Relaciona [[Raissi_2019_PINNs|PINNs]] con [[Spectral_Bias#Notas]]."

        self.assertEqual(
            extract_wikilinks(text),
            ["Raissi_2019_PINNs", "Spectral_Bias"],
        )

    def test_resolves_wikilink_by_unique_vault_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = Path(temp_dir)
            (vault / "concepts").mkdir()
            (vault / "concepts" / "PINN.md").write_text(
                "---\ntype: concept\n---\n# PINN\n",
                encoding="utf-8",
            )
            source = vault / "index.md"
            source.write_text("---\ntype: index\n---\n[[PINN]]\n", encoding="utf-8")

            graph = resolve_links(collect_vault(vault))

            self.assertEqual(len(graph["edges"]), 1)
            self.assertEqual(graph["missing_links"], [])
            self.assertEqual(graph["ambiguous_links"], [])
            self.assertEqual(graph["edges"][0]["relation"], "links_to")

    def test_infers_vault_type_from_controlled_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = Path(temp_dir)
            (vault / "papers").mkdir()
            (vault / "papers" / "Paper.md").write_text(
                "---\ntitle: Paper\n---\n# Paper\n",
                encoding="utf-8",
            )

            nodes = collect_vault(vault)

            self.assertEqual(nodes[0]["type"], "paper")

    def test_reports_missing_and_ambiguous_wikilinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = Path(temp_dir)
            (vault / "one").mkdir()
            (vault / "two").mkdir()
            (vault / "one" / "Shared.md").write_text("# Shared\n", encoding="utf-8")
            (vault / "two" / "Shared.md").write_text("# Shared\n", encoding="utf-8")
            (vault / "index.md").write_text(
                "# Index\n[[Shared]]\n[[MissingConcept]]\n",
                encoding="utf-8",
            )

            graph = resolve_links(collect_vault(vault))

            self.assertEqual(graph["edges"], [])
            self.assertEqual(len(graph["missing_links"]), 1)
            self.assertEqual(len(graph["ambiguous_links"]), 1)

    def test_contracts_have_no_structural_errors(self) -> None:
        result = validate_contracts(PROJECT_ROOT)

        self.assertEqual(result["errors"], [])

    def test_rejects_invalid_notebook_role(self) -> None:
        contracts = load_contracts(PROJECT_ROOT)

        errors = validate_notebook_metadata(
            {"notebook_id": "nb.invalid", "role": "unknown"},
            contracts["notebook"],
        )

        self.assertTrue(any("role" in error for error in errors))

    def test_audits_missing_research_metadata_without_editing_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            notebooks = Path(temp_dir)
            notebook = notebooks / "legacy.ipynb"
            original = {
                "cells": [],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            notebook.write_text(json.dumps(original), encoding="utf-8")
            before = notebook.read_bytes()
            contract = {
                "meta": {"metadata_key": "research"},
                "metadata": {
                    "required_for_all": ["notebook_id", "role"],
                    "allowed_roles": ["exploration", "baseline"],
                    "required_for_experiment": [],
                    "allowed_source_of_truth": ["manifest", "src", "vault", "notebook"],
                },
                "role": [
                    {"id": "exploration", "required_metadata": ["notebook_id", "role"]},
                    {"id": "baseline", "required_metadata": ["notebook_id", "role"]},
                ],
            }

            result = audit_notebooks(notebooks, contract)

            self.assertEqual(result["count"], 1)
            self.assertEqual(result["items"][0]["status"], "unclassified")
            self.assertIn("notebook_id", result["items"][0]["missing_fields"])
            self.assertEqual(notebook.read_bytes(), before)

    def test_build_graph_includes_vault_notebooks_and_manifest(self) -> None:
        graph = build_graph(PROJECT_ROOT)
        node_kinds = {node["kind"] for node in graph["nodes"]}
        report = render_report(graph, load_contracts(PROJECT_ROOT))

        self.assertIn("vault_note", node_kinds)
        self.assertIn("notebook", node_kinds)
        self.assertIn("manifest", node_kinds)
        self.assertGreater(len(graph["edges"]), 0)
        self.assertIn("Graph Engineering Report", report)

    def test_context_pack_projects_contracts_and_graph_without_absolute_paths(self) -> None:
        contracts = load_contracts(PROJECT_ROOT)
        graph = build_graph(PROJECT_ROOT)

        pack = build_context_pack(PROJECT_ROOT, graph, contracts)
        serialized = json.dumps(pack, ensure_ascii=False)

        self.assertEqual(pack["schema_version"], 1)
        self.assertEqual(pack["ontology"], "pikan-research")
        self.assertEqual(
            {role["id"] for role in pack["roles"]},
            {
                "literature_scout",
                "hypothesis_planner",
                "mechanism_proposer",
                "notebook_engineer",
                "experiment_runner",
                "evaluator",
                "research_synthesizer",
            },
        )
        self.assertEqual(
            {gate["id"] for gate in pack["gates"]},
            {"scope", "hypothesis", "code_surface", "eval_integrity", "scientific_claim"},
        )
        self.assertEqual(pack["graph"]["nodes"], len(graph["nodes"]))
        self.assertEqual(pack["graph"]["edges"], len(graph["edges"]))
        self.assertEqual(pack["notebooks"]["count"], graph["notebook_audit"]["count"])
        self.assertEqual(pack["workflow"]["default"], "literature")
        self.assertEqual(len(pack["workflow"]["items"]), 8)
        self.assertNotIn(str(PROJECT_ROOT), serialized)


if __name__ == "__main__":
    unittest.main()
