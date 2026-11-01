"""CLI de validación, auditoría y graph engineering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build_graph, render_report
from .contracts import audit_notebooks, load_contracts, validate_contracts
from .context import build_context_pack


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research automation graph engineering")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("validate", "audit-notebooks"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", default=".")

    build = subparsers.add_parser("build")
    build.add_argument("--root", default=".")
    build.add_argument("--out", default="var/context")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root).resolve()

    if args.command == "validate":
        result = validate_contracts(root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result["errors"] else 0

    contracts = load_contracts(root)
    if args.command == "audit-notebooks":
        result = audit_notebooks(root / contracts["ontology"]["paths"]["notebooks"], contracts["notebook"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    graph = build_graph(root)
    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "research-graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "notebook-audit.json").write_text(
        json.dumps(graph["notebook_audit"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "validation.json").write_text(
        json.dumps(graph["diagnostics"]["validation"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "GRAPH_REPORT.md").write_text(
        render_report(graph, contracts),
        encoding="utf-8",
    )
    (out / "senpi-context.json").write_text(
        json.dumps(
            build_context_pack(root, graph, contracts),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": out.relative_to(root).as_posix(),
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
                "missing_links": len(graph["diagnostics"]["missing_links"]),
                "ambiguous_links": len(graph["diagnostics"]["ambiguous_links"]),
                "notebooks": graph["notebook_audit"]["count"],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return 1 if graph["diagnostics"]["validation"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
