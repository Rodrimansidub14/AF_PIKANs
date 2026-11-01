from __future__ import annotations

import hashlib
import json
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


class ThesisWorkspaceBootstrapTests(unittest.TestCase):
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
    def _hashes(root: Path) -> dict[str, str]:
        return {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_NAMES
        }

    def test_bootstrap_writes_fourteen_notes_and_manifest_without_mutating_sources(self) -> None:
        from src.thesis_workspace.bootstrap import bootstrap_workspace

        root = self._fixture_root()
        before = self._hashes(root)
        output = root / "var" / "agent_runs" / "obsidian-bootstrap" / "Tesis-Studio"

        report = bootstrap_workspace(root, output)

        self.assertEqual(report.note_count, 14)
        self.assertEqual(self._hashes(root), before)
        self.assertTrue((output / "00_Dashboard.md").is_file())
        self.assertTrue((output / "bootstrap-manifest.json").is_file())
        for directory in ("01_Secciones", "02_Revision", "03_Decisiones", "04_Evidencia"):
            self.assertTrue((output / directory).is_dir())
        self.assertTrue(
            (output / "02_Revision" / "Capitulo_6_Attack_Plan.md").is_file()
        )
        self.assertTrue(
            (output / "03_Decisiones" / "DEC-001_Punto_de_inyeccion.md").is_file()
        )
        self.assertTrue(
            (output / "04_Evidencia" / "Figuras_requeridas.md").is_file()
        )

        notes = sorted((output / "01_Secciones").glob("*.md"))
        self.assertEqual(len(notes), 14)
        dashboard = (output / "00_Dashboard.md").read_text(encoding="utf-8")
        self.assertIn("[[01_Secciones/60_Marco_Teorico]]", dashboard)
        self.assertIn("14 notas", dashboard)

        manifest = json.loads((output / "bootstrap-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["conversion_version"], "latex-to-markdown-v2")
        self.assertEqual(len(manifest["sections"]), 14)
        self.assertEqual(manifest["sections"][0]["source_sha256"], before[SOURCE_NAMES[0]])
        first_note = output / manifest["sections"][0]["note_path"]
        self.assertEqual(
            manifest["sections"][0]["note_sha256"],
            hashlib.sha256(first_note.read_bytes()).hexdigest(),
        )

    def test_bootstrap_writes_bibliography_index_when_bib_exists(self) -> None:
        from src.thesis_workspace.bootstrap import bootstrap_workspace

        root = self._fixture_root()
        (root / "m-bibliografia.bib").write_text(
            r"""@article{fuente2026,
  author = {Fuente, Autora},
  title = {Una fuente},
  date = {2026}
}
""",
            encoding="utf-8",
        )
        output = root / "var" / "agent_runs" / "obsidian-bootstrap" / "Tesis-Studio"

        bootstrap_workspace(root, output)

        bibliography = output / "99_Bibliografia.md"
        self.assertTrue(bibliography.is_file())
        self.assertIn("^cite-fuente2026", bibliography.read_text(encoding="utf-8"))
        self.assertEqual(
            (output / "m-bibliografia.bib").read_bytes(),
            (root / "m-bibliografia.bib").read_bytes(),
        )
        self.assertIn("[[99_Bibliografia]]", (output / "00_Dashboard.md").read_text(encoding="utf-8"))

    def test_bootstrap_refuses_output_outside_staging_root(self) -> None:
        from src.thesis_workspace.bootstrap import bootstrap_workspace

        root = self._fixture_root()
        outside = root / "not-allowed"

        with self.assertRaises(ValueError):
            bootstrap_workspace(root, outside)


if __name__ == "__main__":
    unittest.main()
