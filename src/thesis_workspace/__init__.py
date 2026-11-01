"""Safe, one-way workspace bridge for thesis section drafts."""

from .bootstrap import BootstrapReport, bootstrap_workspace
from .markdown_to_latex import convert_markdown_to_latex
from .stage import SourceConflictError, StageReport, stage_workspace
from .model import SectionSpec, load_section_specs

__all__ = [
    "BootstrapReport",
    "SectionSpec",
    "SourceConflictError",
    "StageReport",
    "bootstrap_workspace",
    "convert_markdown_to_latex",
    "load_section_specs",
    "stage_workspace",
]
