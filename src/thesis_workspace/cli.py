"""Command-line interface for the thesis workspace bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bootstrap import bootstrap_workspace
from .stage import stage_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thesis-workspace")
    commands = parser.add_subparsers(dest="command", required=True)

    bootstrap = commands.add_parser(
        "bootstrap",
        help="generate Markdown notes under the repository staging area",
    )
    bootstrap.add_argument("--root", type=Path, default=Path("."))
    bootstrap.add_argument(
        "--out",
        type=Path,
        default=Path("var/agent_runs/obsidian-bootstrap/Tesis-Studio"),
    )

    stage = commands.add_parser(
        "stage",
        help="stage reviewed Markdown as LaTeX drafts and diffs",
    )
    stage.add_argument("--root", type=Path, default=Path("."))
    stage.add_argument("--vault", type=Path, required=True)
    stage.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "bootstrap":
        report = bootstrap_workspace(args.root, args.out)
        print(
            f"Generated {report.note_count} notes in "
            f"{report.output.as_posix()}"
        )
        print(f"Manifest: {report.manifest_path.as_posix()}")
        return 0
    if args.command == "stage":
        report = stage_workspace(args.root, args.vault, args.out)
        print(
            f"Staged {report.section_count} sections in "
            f"{report.output.as_posix()}"
        )
        print(f"Manifest: {report.manifest_path.as_posix()}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")
