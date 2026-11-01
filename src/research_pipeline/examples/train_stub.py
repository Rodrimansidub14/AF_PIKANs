"""Example non-notebook training entrypoint.

Replace this file with a real Heat/Burgers/PIKAN training script. It shows the
contract expected by the harness: read HARNESS_MANIFEST, write metrics and
artifacts into HARNESS_RUN_DIR.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    run_dir = Path(os.environ["HARNESS_RUN_DIR"])
    manifest_path = Path(os.environ["HARNESS_MANIFEST"])

    metrics = {
        "relative_l2_error": 0.25,
        "physics_residual_mean": 0.18,
        "critical_region_residual_mean": 0.31,
        "stability_score": 0.90,
        "wall_time_sec": 1.0,
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "decision_note.md").write_text(
        "\n".join(
            [
                "# Stub run",
                "",
                f"Manifest: `{manifest_path}`",
                "",
                "This is a placeholder script that validates the production pipeline contract.",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

