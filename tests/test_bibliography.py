from __future__ import annotations

import unittest

from src.thesis_workspace.bibliography import parse_bibtex, render_bibliography_note


class BibliographyTests(unittest.TestCase):
    def test_parses_nested_values_and_formats_author_year_label(self) -> None:
        entries = parse_bibtex(
            r"""% comentario
@article{raissi2019,
  author = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  title = {{Physics-informed} neural networks},
  date = {2019},
  doi = {10.1016/j.jcp.2018.10.045}
}
""",
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].key, "raissi2019")
        self.assertEqual(entries[0].fields["date"], "2019")
        self.assertEqual(entries[0].citation_label, "Raissi et al. (2019)")

    def test_renders_stable_block_for_each_bibtex_key(self) -> None:
        entries = parse_bibtex(
            r"""@book{evans1998pde,
  author = {Evans, Lawrence C.},
  title = {Partial Differential Equations},
  date = {1998}
}
""",
        )

        note = render_bibliography_note(entries, "m-bibliografia.bib")

        self.assertIn("kind: thesis_bibliography", note)
        self.assertIn("source_bib: \"m-bibliografia.bib\"", note)
        self.assertIn("^cite-evans1998pde", note)
        self.assertIn("## Evans (1998) — Partial Differential Equations", note)


if __name__ == "__main__":
    unittest.main()
