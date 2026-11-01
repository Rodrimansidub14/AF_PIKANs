from __future__ import annotations

import unittest
from pathlib import Path

from src.thesis_workspace.bibliography import parse_bibtex
from src.thesis_workspace.latex_to_markdown import convert_latex_to_markdown
from src.thesis_workspace.model import SectionSpec


class LatexToMarkdownTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = SectionSpec(
            source_tex=Path("C:/tesis/i-marco_teorico.tex"),
            note_path=Path("01_Secciones/60_Marco_Teorico.md"),
            chapter_title="Marco teórico",
            contains_own_chapter=False,
        )

    def test_converts_section_syntax_without_losing_research_keys(self) -> None:
        source = r"""
\label{sec:marco-teorico}
\section{Aproximación local}
Texto con \textit{énfasis}, \textbf{importante}, \cite{raissi2019} y
\eqref{eq:residual}. La macro \dom permanece visible.

\begin{equation}
r(x;\theta) = \mathcal{N}[u_\theta](x) - f(x)
\label{eq:residual}
\end{equation}

\begin{itemize}
  \item Primer criterio.
  \item Segundo criterio.
\end{itemize}

\pendiente{Verificar la independencia de la señal.}
""".strip()

        result = convert_latex_to_markdown(self.spec, source, "abc123")

        self.assertIn("kind: thesis_section", result)
        self.assertIn("source_sha256: abc123", result)
        self.assertIn("# Marco teórico", result)
        self.assertIn("## Aproximación local", result)
        self.assertIn("*énfasis*", result)
        self.assertIn("**importante**", result)
        self.assertIn(
            r"<!-- latex-citation: \cite{raissi2019} -->[[99_Bibliografia#^cite-raissi2019|raissi2019]]",
            result,
        )
        self.assertIn("[[#^eq-residual]]", result)
        self.assertNotIn(r"\eqref{eq:residual}", result)
        self.assertIn("<!-- latex-label: sec:marco-teorico -->", result)
        self.assertIn("<!-- latex-label: eq:residual -->", result)
        self.assertIn("$$", result)
        self.assertIn("- Primer criterio.", result)
        self.assertIn("> [!todo] Pendiente", result)
        self.assertIn("Verificar la independencia de la señal.", result)
        self.assertIn(r"<!-- latex-pendiente: \pendiente{Verificar la independencia de la señal.} -->", result)
        self.assertIn(r"\dom", result)

    def test_preserves_table_as_raw_latex_and_does_not_duplicate_own_chapter(self) -> None:
        own_chapter = SectionSpec(
            source_tex=Path("C:/tesis/j-capitulos.tex"),
            note_path=Path("01_Secciones/70_Metodologia_y_Plan_de_Trabajo.md"),
            chapter_title="Metodología y Plan de trabajo",
            contains_own_chapter=True,
        )
        source = r"""\chapter{Metodología}
\begin{table}[htbp]
\caption{Estrategias}
\begin{tabular}{ll}
A & B \\
\end{tabular}
\end{table}"""

        result = convert_latex_to_markdown(own_chapter, source, "def456")

        self.assertIn("# Metodología", result)
        self.assertNotIn("# Metodología y Plan de trabajo\n\n# Metodología", result)
        self.assertIn("<!-- latex-env: table -->", result)
        self.assertIn("```latex", result)
        self.assertIn(r"\begin{tabular}{ll}", result)

    def test_exposes_raw_block_citations_as_links_without_editing_the_block(self) -> None:
        bibliography = {
            entry.key: entry
            for entry in parse_bibtex(
                r"""@article{raissi2019,
  author = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  title = {Physics-informed neural networks},
  date = {2019}
}""",
            )
        }
        source = r"""\begin{table}
RAR \cite{raissi2019} \\
\end{table}"""

        result = convert_latex_to_markdown(
            self.spec,
            source,
            "rawcite123",
            bibliography=bibliography,
        )

        raw_start = result.index("```latex")
        raw_end = result.index("```", raw_start + len("```latex"))
        raw_block = result[raw_start:raw_end]
        self.assertIn(r"\cite{raissi2019}", raw_block)
        self.assertIn("<!-- latex-raw-citations -->", result)
        self.assertIn(
            "[[99_Bibliografia#^cite-raissi2019|Raissi et al. (2019)]]",
            result,
        )

    def test_rejects_unbalanced_supported_environment(self) -> None:
        with self.assertRaises(ValueError):
            convert_latex_to_markdown(
                self.spec,
                r"\begin{equation} x = 1",
                "ghi789",
            )

    def test_converts_multiline_pending_marker(self) -> None:
        source = r"""\section{Pendientes}
\pendiente{Agregar una explicación con
\cite{fuente2026} y conservar su contexto.}
"""

        result = convert_latex_to_markdown(self.spec, source, "jkl012")

        self.assertIn("> [!todo] Pendiente", result)
        self.assertIn("fuente2026", result)
        self.assertIn("<!-- latex-pendiente:", result)

    def test_hides_leading_preamble_and_preserves_it_for_latex(self) -> None:
        source = r"""% Comentario del capítulo.
\newcommand{\dom}{\Omega}
\newcommand{\pde}{\mathcal{N}}

\label{sec:test}
\section{Prueba}
Texto."""

        result = convert_latex_to_markdown(self.spec, source, "pre123")

        self.assertIn("<!-- latex-preamble", result)
        self.assertIn("latex-preamble-end -->", result)
        self.assertIn(r"\newcommand{\dom}{\Omega}", result)
        self.assertIn(r"\newcommand{\pde}{\mathcal{N}}", result)

    def test_converts_references_and_moves_equation_labels_outside_math(self) -> None:
        source = r"""\section{Referencias}
Texto con \eqref{eq:residual} y \ref{tab:resultados}.

\begin{equation}
x = 1
\label{eq:residual}
\end{equation}

\begin{table}
\caption{Resultados}
\label{tab:resultados}
\end{table}"""

        result = convert_latex_to_markdown(self.spec, source, "ref123")

        self.assertIn("[[#^eq-residual]]", result)
        self.assertIn("[[#^tab-resultados]]", result)
        self.assertIn("^eq-residual", result)
        self.assertIn("^tab-resultados", result)
        equation_start = result.index("<!-- latex-env: equation -->")
        equation_open = result.index("$$", equation_start)
        equation_close = result.index("$$", equation_open + 2)
        label_position = result.index("<!-- latex-label: eq:residual -->")
        self.assertGreater(label_position, equation_close)

    def test_converts_citation_group_to_bibliography_links(self) -> None:
        bibliography = {
            entry.key: entry
            for entry in parse_bibtex(
                r"""@article{raissi2019,
  author = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  title = {Physics-informed neural networks},
  date = {2019}
}
@article{karniadakis2021,
  author = {Karniadakis, George Em and Kevrekidis, Ioannis G.},
  title = {Physics-informed machine learning},
  date = {2021}
}""",
            )
        }
        result = convert_latex_to_markdown(
            self.spec,
            r"Texto con \cite{raissi2019,karniadakis2021}.",
            "cite123",
            bibliography=bibliography,
        )

        self.assertIn(
            r"<!-- latex-citation: \cite{raissi2019,karniadakis2021} -->"
            "[[99_Bibliografia#^cite-raissi2019|Raissi et al. (2019)]], "
            "[[99_Bibliografia#^cite-karniadakis2021|Karniadakis & Kevrekidis (2021)]].",
            result,
        )

    def test_separates_inline_pending_callouts_from_surrounding_prose(self) -> None:
        source = r"Texto previo. \pendiente{Revisar esta afirmación.} Texto posterior."

        result = convert_latex_to_markdown(self.spec, source, "todo123")
        lines = result.splitlines()

        self.assertIn("Texto previo.", lines)
        self.assertIn("> [!todo] Pendiente", lines)
        self.assertIn("Texto posterior.", lines)
        self.assertFalse(
            any(
                "> [!todo] Pendiente" in line
                and ("Texto previo" in line or "Texto posterior" in line)
                for line in lines
            )
        )

    def test_converts_inline_pending_marker_without_swallowing_paragraph_text(self) -> None:
        source = r"Texto previo. \pendiente{Revisar esta afirmación.} Texto posterior."

        result = convert_latex_to_markdown(self.spec, source, "mno345")

        self.assertIn("Texto previo.", result)
        self.assertIn("> [!todo] Pendiente", result)
        self.assertIn("Texto posterior.", result)


if __name__ == "__main__":
    unittest.main()
