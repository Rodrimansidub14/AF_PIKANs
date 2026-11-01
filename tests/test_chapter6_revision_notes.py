from __future__ import annotations

from pathlib import Path
import re
import unittest


TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "thesis_workspace"
    / "templates"
)


class Chapter6RevisionNotesTests(unittest.TestCase):
    def _read(self, relative: str) -> str:
        return (TEMPLATE_ROOT / relative).read_text(encoding="utf-8")

    def test_gab_ledger_contains_each_of_29_annotations_once(self) -> None:
        ledger = self._read("02_Revision/GAB_29_Anotaciones.md")
        identifiers = re.findall(r"^### GAB-(\d{2})$", ledger, flags=re.MULTILINE)

        self.assertEqual(identifiers, [f"{number:02d}" for number in range(1, 30)])
        for number in range(1, 30):
            block = re.search(
                rf"^### GAB-{number:02d}\n(.*?)(?=^### GAB-|\Z)",
                ledger,
                flags=re.MULTILINE | re.DOTALL,
            )
            self.assertIsNotNone(block)
            self.assertRegex(block.group(1), r"status: (open|addressed|superseded|needs_adviser)")

    def test_consensus_note_keeps_pedagogy_and_argumentation_together(self) -> None:
        consensus = self._read("02_Revision/Capitulo_6_Consenso.md")

        for phrase in (
            "tesis de mecanismo",
            "pedagogía",
            "argumentación",
            "entrada y cierre",
            "precedente clásico",
            "qué redistribuye",
            "6.4",
            "6.7",
            "PIKAN",
        ):
            self.assertIn(phrase, consensus)

    def test_attack_plan_leaves_scientific_redaction_to_researcher(self) -> None:
        plan = self._read("02_Revision/Capitulo_6_Attack_Plan.md")

        self.assertIn("redacción científica la hace el investigador", plan)
        self.assertIn("DEC-001", plan)
        self.assertIn("punto de inyección", plan)
        self.assertIn("presupuesto de páginas", plan)
        self.assertIn("promoción", plan)

    def test_decision_packets_expose_adviser_gates(self) -> None:
        injection = self._read("03_Decisiones/DEC-001_Punto_de_inyeccion.md")
        mechanism = self._read("03_Decisiones/DEC-002_Mecanismo_vs_aplicacion.md")
        signal = self._read("03_Decisiones/DEC-003_Senal_y_residual.md")
        notation = self._read("03_Decisiones/DEC-004_Notacion_del_residual.md")

        for option in ("A", "B", "C", "decisión del asesor"):
            self.assertIn(option, injection)
        self.assertIn("piezoeléctrico", mechanism)
        for phrase in ("propiedad del problema", "observable", "colinealidad"):
            self.assertIn(phrase, signal)
        self.assertIn("N[u](x) - f(x)", notation)
        self.assertIn("decisión del asesor", notation)


if __name__ == "__main__":
    unittest.main()
