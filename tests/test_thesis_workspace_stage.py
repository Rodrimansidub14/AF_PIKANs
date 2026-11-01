from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest


SOURCE_NAMES = (
    "a-prefacio.tex",
    "b-resumen.tex",
    "c-abstract.tex",
    "d-introduccion.tex",
    "e-antecedentes.tex",
    "f-justificacion.tex",
    "g-objetivos.tex",
    "h-alcance.tex",
    "i-marco_teorico.tex",
    "j-capitulos.tex",
    "k-conclusiones.tex",
    "l-recomendaciones.tex",
    "n-anexos.tex",
    "o-glosario.tex",
)


class ThesisWorkspaceStageTests(unittest.TestCase):
    def _fixture_root(self) -> Path:
        temp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: self._remove_tree(temp))
        for name in SOURCE_NAMES:
            content = rf"\section{{{name}}}\nTexto de prueba para {name}.\n"
            if name == "j-capitulos.tex":
                content = rf"\chapter{{Metodología}}\n\section{{{name}}}\nTexto de prueba para {name}.\n"
            (temp / name).write_text(content, encoding="utf-8")
        return temp

    @staticmethod
    def _remove_tree(path: Path) -> None:
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()

    @staticmethod
    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_stage_writes_drafts_and_diffs_without_overwriting_active_tex(self) -> None:
        from src.thesis_workspace.bootstrap import bootstrap_workspace
        from src.thesis_workspace.stage import stage_workspace

        root = self._fixture_root()
        bootstrap = root / "var" / "agent_runs" / "obsidian-bootstrap" / "Tesis-Studio"
        bootstrap_workspace(root, bootstrap)
        before = {name: self._hash(root / name) for name in SOURCE_NAMES}

        output = root / "var" / "agent_runs" / "latex-drafts" / "run-001"
        report = stage_workspace(root, bootstrap, output)

        self.assertEqual(report.section_count, 14)
        self.assertEqual({name: self._hash(root / name) for name in SOURCE_NAMES}, before)
        self.assertEqual(len(list(output.glob("*.tex"))), 14)
        self.assertEqual(len(list(output.glob("*.diff"))), 14)
        self.assertTrue((output / "manifest.json").is_file())
        self.assertTrue(
            all(not diff.read_text(encoding="utf-8") for diff in output.glob("*.diff"))
        )

    def test_stage_refuses_changed_active_source(self) -> None:
        from src.thesis_workspace.bootstrap import bootstrap_workspace
        from src.thesis_workspace.stage import SourceConflictError, stage_workspace

        root = self._fixture_root()
        bootstrap = root / "var" / "agent_runs" / "obsidian-bootstrap" / "Tesis-Studio"
        bootstrap_workspace(root, bootstrap)
        (root / "i-marco_teorico.tex").write_text("changed\n", encoding="utf-8")

        with self.assertRaises(SourceConflictError) as context:
            stage_workspace(
                root,
                bootstrap,
                root / "var" / "agent_runs" / "latex-drafts" / "run-002",
            )

        self.assertIn("i-marco_teorico.tex", str(context.exception))

    def test_stage_refuses_note_with_mismatched_source_hash(self) -> None:
        from src.thesis_workspace.bootstrap import bootstrap_workspace
        from src.thesis_workspace.stage import SourceConflictError, stage_workspace

        root = self._fixture_root()
        bootstrap = root / "var" / "agent_runs" / "obsidian-bootstrap" / "Tesis-Studio"
        bootstrap_workspace(root, bootstrap)
        note = bootstrap / "01_Secciones" / "60_Marco_Teorico.md"
        note.write_text(
            note.read_text(encoding="utf-8").replace(
                "source_sha256: ",
                "source_sha256: tampered-",
                1,
            ),
            encoding="utf-8",
        )

        with self.assertRaises(SourceConflictError):
            stage_workspace(
                root,
                bootstrap,
                root / "var" / "agent_runs" / "latex-drafts" / "run-003",
            )

    def test_edited_note_produces_real_draft_and_diff(self) -> None:
        from src.thesis_workspace.bootstrap import bootstrap_workspace
        from src.thesis_workspace.stage import stage_workspace

        root = self._fixture_root()
        bootstrap = root / "var" / "agent_runs" / "obsidian-bootstrap" / "Tesis-Studio"
        bootstrap_workspace(root, bootstrap)
        note = bootstrap / "01_Secciones" / "60_Marco_Teorico.md"
        note.write_text(
            note.read_text(encoding="utf-8") + "\nTexto editado por el investigador.\n",
            encoding="utf-8",
        )

        output = root / "var" / "agent_runs" / "latex-drafts" / "run-edited"
        stage_workspace(root, bootstrap, output)

        self.assertIn(
            "Texto editado por el investigador.",
            (output / "i-marco_teorico.tex").read_text(encoding="utf-8"),
        )
        self.assertTrue((output / "i-marco_teorico.diff").read_text(encoding="utf-8"))

    def test_stage_refuses_output_outside_latex_draft_root(self) -> None:
        from src.thesis_workspace.bootstrap import bootstrap_workspace
        from src.thesis_workspace.stage import stage_workspace

        root = self._fixture_root()
        bootstrap = root / "var" / "agent_runs" / "obsidian-bootstrap" / "Tesis-Studio"
        bootstrap_workspace(root, bootstrap)

        with self.assertRaises(ValueError):
            stage_workspace(root, bootstrap, root / "not-allowed")


if __name__ == "__main__":
    unittest.main()
