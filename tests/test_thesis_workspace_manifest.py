from __future__ import annotations

import unittest
from pathlib import Path

from src.thesis_workspace.model import SectionSpec, load_section_specs
from src.thesis_workspace.paths import thesis_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = thesis_root()


@unittest.skipIf(
    THESIS_ROOT is None,
    "El repositorio de escritura no está disponible; define THESIS_ROOT para incluir estas pruebas.",
)
class ThesisWorkspaceManifestTests(unittest.TestCase):
    def test_active_narrative_modules_have_stable_markdown_targets(self) -> None:
        specs = load_section_specs(THESIS_ROOT)

        self.assertEqual(len(specs), 14)
        self.assertEqual(
            [spec.source_tex.name for spec in specs],
            [
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
            ],
        )
        self.assertEqual(
            [spec.note_path.as_posix() for spec in specs],
            [
                "01_Secciones/00_Prefacio.md",
                "01_Secciones/01_Resumen.md",
                "01_Secciones/02_Abstract.md",
                "01_Secciones/10_Introduccion.md",
                "01_Secciones/20_Antecedentes.md",
                "01_Secciones/30_Justificacion.md",
                "01_Secciones/40_Objetivos.md",
                "01_Secciones/50_Alcance.md",
                "01_Secciones/60_Marco_Teorico.md",
                "01_Secciones/70_Metodologia_y_Plan_de_Trabajo.md",
                "01_Secciones/80_Conclusiones.md",
                "01_Secciones/90_Recomendaciones.md",
                "01_Secciones/95_Anexos.md",
                "01_Secciones/99_Glosario.md",
            ],
        )

    def test_manifest_keeps_chapter_titles_that_live_in_z_main(self) -> None:
        specs = load_section_specs(THESIS_ROOT)

        by_name = {spec.source_tex.name: spec for spec in specs}
        self.assertEqual(by_name["d-introduccion.tex"].chapter_title, "Introducción")
        self.assertEqual(by_name["i-marco_teorico.tex"].chapter_title, "Marco teórico")
        self.assertEqual(
            by_name["j-capitulos.tex"].chapter_title,
            "Metodología y Plan de trabajo",
        )
        self.assertTrue(by_name["j-capitulos.tex"].contains_own_chapter)
        self.assertFalse(by_name["i-marco_teorico.tex"].contains_own_chapter)

    def test_manifest_rejects_duplicate_or_unsafe_targets(self) -> None:
        self.assertTrue(issubclass(SectionSpec, object))
        with self.assertRaises((FileNotFoundError, ValueError)):
            load_section_specs(PROJECT_ROOT / "does-not-exist")


if __name__ == "__main__":
    unittest.main()
