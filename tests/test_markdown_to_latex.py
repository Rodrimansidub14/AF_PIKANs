from __future__ import annotations

from pathlib import Path
import unittest

from src.thesis_workspace.latex_to_markdown import convert_latex_to_markdown
from src.thesis_workspace.markdown_to_latex import convert_markdown_to_latex
from src.thesis_workspace.model import SectionSpec


class MarkdownToLatexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = SectionSpec(
            source_tex=Path("C:/tesis/i-marco_teorico.tex"),
            note_path=Path("01_Secciones/60_Marco_Teorico.md"),
            chapter_title="Marco teórico",
            contains_own_chapter=False,
        )

    def test_round_trip_restores_supported_constructs(self) -> None:
        source = r"""\section{Aproximación local}
Texto con \textit{énfasis}, \textbf{importante}, \cite{raissi2019} y
\eqref{eq:residual}.

\begin{equation}
r(x;\theta) = \mathcal{N}[u_\theta](x) - f(x)
\label{eq:residual}
\end{equation}

\begin{itemize}
  \item Primer criterio.
\end{itemize}

\pendiente{Verificar la señal.}

\begin{table}[htbp]
\caption{Estrategias}
\end{table}"""

        markdown = convert_latex_to_markdown(self.spec, source, "abc123")
        restored = convert_markdown_to_latex(markdown)

        self.assertNotIn(r"\chapter{Marco teórico}", restored)
        self.assertIn(r"\section{Aproximación local}", restored)
        self.assertIn(r"\textit{énfasis}", restored)
        self.assertIn(r"\textbf{importante}", restored)
        self.assertIn(r"\cite{raissi2019}", restored)
        self.assertIn(r"\eqref{eq:residual}", restored)
        self.assertIn(r"\label{eq:residual}", restored)
        self.assertIn(r"\begin{equation}", restored)
        self.assertIn(r"\end{equation}", restored)
        self.assertIn(r"\begin{table}[htbp]", restored)
        self.assertIn(r"\pendiente{Verificar la señal.}", restored)
        self.assertIn(r"\item Primer criterio.", restored)

    def test_unknown_markdown_text_is_not_silently_deleted(self) -> None:
        markdown = """---
kind: thesis_section
contains_own_chapter: true
---

# Metodología

Texto con una macro \\dom y una cita \\cite{fuente}.
"""

        restored = convert_markdown_to_latex(markdown)

        self.assertIn(r"\chapter{Metodología}", restored)
        self.assertIn(r"\dom", restored)
        self.assertIn(r"\cite{fuente}", restored)

    def test_round_trip_restores_multiline_and_inline_todos(self) -> None:
        source = r"""Texto previo. \pendiente{Revisar una afirmación con
\cite{fuente2026}.} Texto posterior."""

        markdown = convert_latex_to_markdown(self.spec, source, "pqr678")
        restored = convert_markdown_to_latex(markdown)

        self.assertIn(r"\pendiente{Revisar una afirmación con \cite{fuente2026}.}", restored)
        self.assertIn("Texto previo.", restored)
        self.assertIn("Texto posterior.", restored)

    def test_restores_preamble_links_and_external_equation_label(self) -> None:
        markdown = r"""---
kind: thesis_section
contains_own_chapter: true
---

<!-- latex-preamble
% Comentario del capítulo.
\newcommand{\dom}{\Omega}
latex-preamble-end -->

# Metodología

Texto con [[#^eq-residual]] y [[#^tab-resultados]].

<!-- latex-env: equation -->
$$
x = 1
$$
^eq-residual
<!-- latex-label: eq:residual -->
<!-- /latex-env -->

<!-- latex-env: table -->
```latex
\begin{table}
\caption{Resultados}
\label{tab:resultados}
\end{table}
```
^tab-resultados
<!-- /latex-env -->
"""

        restored = convert_markdown_to_latex(markdown)

        self.assertIn(r"\newcommand{\dom}{\Omega}", restored)
        self.assertIn(r"\eqref{eq:residual}", restored)
        self.assertIn(r"\ref{tab:resultados}", restored)
        self.assertIn(r"\label{eq:residual}", restored)
        self.assertIn(r"\label{tab:resultados}", restored)
        self.assertNotIn("[[#^", restored)
        self.assertNotIn("^eq-residual", restored)

    def test_restores_grouped_bibliography_links_to_original_citation(self) -> None:
        markdown = r"""---
kind: thesis_section
contains_own_chapter: true
---

# Metodología

Texto con <!-- latex-citation: \cite{raissi2019,karniadakis2021} -->[[99_Bibliografia#^cite-raissi2019|Raissi et al. (2019)]], [[99_Bibliografia#^cite-karniadakis2021|Karniadakis & Kevrekidis (2021)]].
"""

        restored = convert_markdown_to_latex(markdown)

        self.assertIn(r"\cite{raissi2019,karniadakis2021}", restored)
        self.assertNotIn("latex-citation", restored)
        self.assertNotIn("[[99_Bibliografia", restored)

    def test_discards_raw_block_citation_summary_when_restoring_table(self) -> None:
        markdown = r"""---
kind: thesis_section
contains_own_chapter: true
---

# Metodología

<!-- latex-env: table -->
```latex
\begin{table}
RAR \cite{raissi2019} \\
\end{table}
```
<!-- latex-raw-citations -->
- <!-- latex-citation: \cite{raissi2019} -->[[99_Bibliografia#^cite-raissi2019|Raissi et al. (2019)]]
<!-- /latex-raw-citations -->
<!-- /latex-env -->
"""

        restored = convert_markdown_to_latex(markdown)

        self.assertIn(r"RAR \cite{raissi2019}", restored)
        self.assertNotIn("latex-raw-citations", restored)
        self.assertNotIn("[[99_Bibliografia", restored)

    def test_skips_hidden_latex_agent_prompt(self) -> None:
        markdown = r"""---
kind: thesis_section
contains_own_chapter: true
---

# Metodología

Texto que sí debe llegar a LaTeX.

<!-- latex-agent-prompt
# PROMPT PARA AGENTE LATEX

Implementa el cambio en el archivo asociado.
Conserva \cite{fuente} y no inventes referencias.
latex-agent-prompt-end -->
"""

        restored = convert_markdown_to_latex(markdown)

        self.assertIn("Texto que sí debe llegar a LaTeX.", restored)
        self.assertNotIn("latex-agent-prompt", restored)
        self.assertNotIn(r"\cite{fuente}", restored)


if __name__ == "__main__":
    unittest.main()
