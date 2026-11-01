"""Package the already approved DFG FEM/scoring source for Kaggle; no execution."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
RUN = Path(__file__).resolve().parent
SLUG = "rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929"
SESSION_CAP = 1281.024
SOURCES = [
    "rodrimandub/af-pikans-dfg-rbf-refinement-20260928",
    "rodrimandub/af-pikans-dfg-lbfgs-refinement-20260927",
]
EVAL_CONTRACT = {
    "grid": {
        "x": {"start": 0.0, "stop": 2.2, "count": 221, "endpoint": True},
        "y": {"start": 0.0, "stop": 0.41, "count": 42, "endpoint": True},
        "ordering": "numpy.meshgrid(indexing='xy'); row-major flattened",
        "mask": "exclude points with (x-0.2)^2+(y-0.2)^2 <= 0.05^2",
    },
    "pressure_gauge": {
        "method": "subtract pressure at fixed outlet point from FEM and each checkpoint",
        "outlet_anchor_xy": [2.2, 0.2],
    },
    "metrics": ["u_relative_l2", "v_relative_l2", "p_relative_l2_gauge_fixed_outlet"],
    "cohorts": {
        "current_rbf": [301, 302, 303, 304, 305],
        "refinement_rbf_and_tanh_mlp": [501, 502, 503, 504, 505],
    },
    "reference_level": "DFG-L3 (third of three requested mesh levels)",
    "retraining": False,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_wrapper = ROOT / "var/agent_runs/baseline-competitive-2026-09-27/s1b-retry2/closure_s1b_retry.py"
    solver_path = ROOT / "var/agent_runs/baseline-competitive-2026-09-27/s1/solve_dfg.py"
    current_model_path = ROOT / "var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-rbf-current-cohort-refinement/dfg-current-cohort-refinement/run-1790606331368753261/vendor/src/research_pipeline/sciml/dfg.py"
    wrapper = source_wrapper.read_text(encoding="utf-8")
    solver = solver_path.read_text(encoding="utf-8")
    current_model_source = current_model_path.read_text(encoding="utf-8")

    old_probe = '"gmsh": getattr(__import__("gmsh"), "__version__", "unknown"),'
    new_probe = '"gmsh_binary": subprocess.check_output([args.gmsh, "--version"], text=True).strip(),'
    if old_probe not in solver:
        raise SystemExit("Expected failed gmsh Python import probe was not found in solver source.")
    solver = solver.replace(old_probe, new_probe, 1)
    ufl_probe = '"ufl": getattr(ufl, "__version__", "unknown"),'
    if ufl_probe not in solver:
        raise SystemExit("Expected UFL version field was not found before adding the Python Gmsh version key.")
    solver = solver.replace(
        ufl_probe,
        ufl_probe + '\n        "gmsh_python": getattr(__import__("gmsh"), "__version__", "unknown"),',
        1,
    )
    fem_import = "from dolfinx import fem, geometry, io"
    if fem_import not in solver:
        raise SystemExit("Expected DOLFINx FEM import was not found before importing the optional PETSc module.")
    solver = solver.replace(
        fem_import,
        fem_import + "\nfrom dolfinx.fem import petsc as fem_petsc",
        1,
    )
    old_problem = "fem.petsc.NonlinearProblem("
    if old_problem not in solver:
        raise SystemExit("Expected DOLFINx PETSc nonlinear-problem reference was not found.")
    solver = solver.replace(old_problem, "fem_petsc.NonlinearProblem(", 1)

    tree = ast.parse(wrapper)
    asset_assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_EMBEDDED_DFG_ASSETS" for target in node.targets)
    )
    assets = ast.literal_eval(asset_assignment.value)
    assets["solve_dfg.py"] = base64.b64encode(solver.encode("utf-8")).decode("ascii")
    assets["dfg_model.py"] = base64.b64encode(current_model_source.encode("utf-8")).decode("ascii")
    encoded_assets = "_EMBEDDED_DFG_ASSETS = " + repr(assets)
    wrapper_lines = wrapper.splitlines(keepends=True)
    wrapper = (
        "".join(wrapper_lines[:asset_assignment.lineno - 1])
        + encoded_assets + "\n"
        + "".join(wrapper_lines[asset_assignment.end_lineno:])
    )

    replacements = {
        'SCRIPT_DIR = Path("/kaggle/working/closure-s1b-source")':
            'SCRIPT_DIR = Path("/kaggle/working/dfg-fem-field-followup-retry5-source")',
        'OUT = Path("/kaggle/working/closure-s1b-retry2")':
            'OUT = Path("/kaggle/working/dfg-fem-field-followup-retry5")',
        'MAMBA_TMP = Path("/tmp/af-pikans-closure-s1-micromamba")':
            'MAMBA_TMP = Path("/tmp/af-pikans-dfg-fem-followup-retry5-micromamba")',
        "SESSION_CAP = 1800.0": f"SESSION_CAP = {SESSION_CAP}",
        "WATCHDOG = 1680.0": "WATCHDOG = 1220.0",
        "DFG_CAP = 1450.0": "DFG_CAP = 1100.0",
        'FEATFLOW = {"Cd": 5.57953523384, "Cl": 0.010618948146, "delta_p": 0.11752016697}':
            'FEATFLOW = {"Cd": 5.57953523384, "Cl": 0.010618948146, "delta_p": 0.11752016697}\n'
            f'EVAL_CONTRACT = {EVAL_CONTRACT!r}',
        '"session": "S1b-infrastructure-retry"': '"session": "DFG-FEM-field-followup-retry5-2026-09-29"',
        '"kernel_slug": "rodrimandub/af-pikans-competitive-baseline-s1b-retry2-20260928"':
            f'"kernel_slug": "{SLUG}"',
        '"rodrimandub/af-pikans-allen-cahn-t4x2-20260927",\n'
        '            "rodrimandub/af-pikans-allen-cahn-score-only-20260927",\n'
        '            "rodrimandub/allen-cahn-deepxde-reference-20260928",\n'
        '            "rodrimandub/af-pikans-dfg-rbf-refinement-20260928",\n'
        '            "rodrimandub/af-pikans-dfg-lbfgs-refinement-20260927",':
            '"rodrimandub/af-pikans-dfg-rbf-refinement-20260928",\n'
            '            "rodrimandub/af-pikans-dfg-lbfgs-refinement-20260927",',
        '"fixed_field_grid": "DFG-L2 fields_on_common_grid.npz (221x42 requested points, cylinder interior excluded)",':
            '"fixed_field_scoring_contract": EVAL_CONTRACT,',
        'reference_path = OUT / "dfg" / "DFG-L2" / "fields_on_common_grid.npz"':
            'reference_path = OUT / "dfg" / "DFG-L3" / "fields_on_common_grid.npz"',
        '"pressure_gauge": {"method": "subtract pressure at outflow anchor", "xy": anchor.tolist()},':
            '"pressure_gauge": EVAL_CONTRACT["pressure_gauge"],',
        'anchor = np.asarray([2.2, 0.2], dtype=np.float64)':
            'anchor = np.asarray(EVAL_CONTRACT["pressure_gauge"]["outlet_anchor_xy"], dtype=np.float64)',
        'current_sources = [p for p in roots[0].rglob("dfg.py") if "dfg-rbf-refinement" in str(p).lower()]':
            'current_sources = [SCRIPT_DIR / "dfg_model.py"] if (SCRIPT_DIR / "dfg_model.py").is_file() else []',
        'pred_p = pred[:, 2] - pred[anchor_index]':
            'pred_p = pred[:, 2] - pred[anchor_index, 2]',
        '    module = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(module)':
            '    module = importlib.util.module_from_spec(spec)\n    sys.modules[name] = module\n    spec.loader.exec_module(module)',
        '            "--max-levels", "3", "--wall-cap-seconds", str(min(DFG_CAP - 120.0, remaining)),\n':
            '            "--max-levels", "3", "--start-level", "1",\n'
            '            "--wall-cap-seconds", str(min(DFG_CAP - 120.0, remaining)),\n',
    }
    for before, after in replacements.items():
        if before not in wrapper:
            raise SystemExit(f"Expected source fragment not found: {before}")
        wrapper = wrapper.replace(before, after, 1)
    env_marker = '    env = MAMBA_TMP / "dfg-fem-env"\n'
    if wrapper.count(env_marker) != 1:
        raise SystemExit("Could not add a shared timeout deadline to the DOLFINx environment setup.")
    wrapper = wrapper.replace(env_marker, env_marker + '    deadline = time.perf_counter() + timeout\n', 1)
    create_timeout = 'timeout=max(30.0, timeout),\n'
    if wrapper.count(create_timeout) != 1:
        raise SystemExit("Could not bound micromamba environment creation by its remaining worker allowance.")
    wrapper = wrapper.replace(create_timeout, 'timeout=max(1.0, deadline - time.perf_counter()),\n', 1)
    return_marker = '    return exe, env\n\n\ndef dfg_worker() -> int:\n'
    pip_install = (
        '    with log_path.open("a", encoding="utf-8") as log:\n'
        '        subprocess.run(\n'
        '            [str(exe), "run", "--root-prefix", str(root), "--prefix", str(env),\n'
        '             "python", "-m", "pip", "install", "--disable-pip-version-check", "gmsh==4.15.2"],\n'
        '            stdout=log, stderr=subprocess.STDOUT, check=True,\n'
        '            timeout=max(1.0, deadline - time.perf_counter()),\n'
        '        )\n'
    )
    if wrapper.count(return_marker) != 1:
        raise SystemExit("Could not insert the Gmsh Python wheel install before FEM execution.")
    wrapper = wrapper.replace(return_marker, pip_install + return_marker, 1)
    watchdog_line = '        "internal_watchdog_seconds": WATCHDOG,\n'
    with_contract = watchdog_line + '        "fixed_field_scoring_contract": EVAL_CONTRACT,\n'
    if wrapper.count(watchdog_line) != 1:
        raise SystemExit("Could not preserve the fixed-grid declaration in the final manifest.")
    wrapper = wrapper.replace(watchdog_line, with_contract, 1)

    scoring_loop = '    for seed, arm in schedule:\n        checkpoint = find_checkpoint(seed, arm, roots, np)\n'
    bounded_scoring_loop = (
        '    write_json(OUT / "dfg" / "checkpoint_field_scores.partial.json", {\n'
        '        "status": "scoring", "fixed_field_scoring_contract": EVAL_CONTRACT,\n'
        '        "per_seed": [], "reference_accepted": False, "scientific_accepted": False,\n'
        '    })\n'
        '    for schedule_index, (seed, arm) in enumerate(schedule):\n'
        '        if time.perf_counter() >= STARTED_MONOTONIC + SESSION_CAP - 15.0:\n'
        '            rows.extend({"seed": later_seed, "arm": later_arm, "status": "session_time_cap_not_started"}\n'
        '                        for later_seed, later_arm in schedule[schedule_index:])\n'
        '            break\n'
        '        checkpoint = find_checkpoint(seed, arm, roots, np)\n'
    )
    if wrapper.count(scoring_loop) != 1:
        raise SystemExit("Could not install DFG field-scoring deadline/partial checkpoint.")
    wrapper = wrapper.replace(scoring_loop, bounded_scoring_loop, 1)
    row_write = '            rows.append(row)\n            del model, payload, parts, pred\n'
    partial_write = (
        '            rows.append(row)\n'
        '            write_json(OUT / "dfg" / "checkpoint_field_scores.partial.json", {\n'
        '                "status": "scoring", "fixed_field_scoring_contract": EVAL_CONTRACT,\n'
        '                "per_seed": rows, "reference_accepted": False, "scientific_accepted": False,\n'
        '            })\n'
        '            del model, payload, parts, pred\n'
    )
    if wrapper.count(row_write) != 1:
        raise SystemExit("Could not add per-checkpoint partial score persistence.")
    wrapper = wrapper.replace(row_write, partial_write, 1)
    error_write = (
        '            rows.append({"seed": seed, "arm": arm, "status": "scoring_error",\n'
        '                         "checkpoint": str(checkpoint), "error": f"{type(exc).__name__}: {exc}"})\n'
    )
    error_partial_write = error_write + (
        '            write_json(OUT / "dfg" / "checkpoint_field_scores.partial.json", {\n'
        '                "status": "scoring", "fixed_field_scoring_contract": EVAL_CONTRACT,\n'
        '                "per_seed": rows, "reference_accepted": False, "scientific_accepted": False,\n'
        '            })\n'
    )
    if wrapper.count(error_write) != 1:
        raise SystemExit("Could not add partial score persistence after scoring errors.")
    wrapper = wrapper.replace(error_write, error_partial_write, 1)

    marker = 'def s1b_retry_main() -> int:\n    OUT.mkdir(parents=True, exist_ok=True)\n'
    initial_manifest = (
        'def s1b_retry_main() -> int:\n'
        '    OUT.mkdir(parents=True, exist_ok=True)\n'
        '    write_json(OUT / "manifest.json", {\n'
        '        "schema_version": 1, "session": "DFG-FEM-field-followup-retry5-2026-09-29",\n'
        f'        "kernel_slug": "{SLUG}", "status": "starting",\n'
        '        "push_cap_seconds": SESSION_CAP, "fixed_field_scoring_contract": EVAL_CONTRACT,\n'
        '        "kernel_sources": ' + repr(SOURCES) + ',\n'
        '        "reference_accepted": False, "scientific_accepted": False,\n'
        '    })\n'
    )
    if marker not in wrapper:
        raise SystemExit("Could not add pre-scoring manifest declaration.")
    wrapper = wrapper.replace(marker, initial_manifest, 1)

    wrapper = wrapper.replace("DFG-FEM-field-followup-retry2-2026-09-29", "DFG-FEM-field-followup-retry5-2026-09-29")
    wrapper = wrapper.replace("DFG-FEM-field-followup-retry4-2026-09-29", "DFG-FEM-field-followup-retry5-2026-09-29")
    wrapper = wrapper.replace("af-pikans-dfg-fem-field-followup-retry2-20260929", SLUG.split("/", 1)[1])
    wrapper = wrapper.replace("af-pikans-dfg-fem-field-follow-up-retry-4-20260929", SLUG.split("/", 1)[1])
    wrapper = wrapper.replace("dfg-fem-field-followup-retry2", "dfg-fem-field-followup-retry5")
    wrapper = wrapper.replace("dfg-fem-field-followup-retry4", "dfg-fem-field-followup-retry5")
    wrapper = wrapper.replace("retry2-micromamba", "retry5-micromamba")
    wrapper = wrapper.replace("retry4-micromamba", "retry5-micromamba")
    (RUN / "closure_dfg_followup_retry5.py").write_text(wrapper, encoding="utf-8", newline="\n")
    (RUN / "solve_dfg.py").write_text(solver, encoding="utf-8", newline="\n")
    metadata = {
        "id": SLUG,
        "title": "AF PIKANs DFG FEM Field Follow-up Retry 5 20260929",
        "code_file": "closure_dfg_followup_retry5.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": SOURCES,
        "model_sources": [],
    }
    (RUN / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "session": "DFG-FEM-field-followup-retry5-2026-09-29",
        "kernel_slug": SLUG,
        "private": True,
        "accelerator": "NvidiaTeslaT4 x2 required and checked by entrypoint",
        "session_cap_seconds": SESSION_CAP,
        "worker_cap_seconds": 1100,
        "watchdog_seconds": 1220,
        "kernel_sources": SOURCES,
        "gmsh_python_package": "gmsh==4.15.2 installed into the micromamba DOLFINx interpreter from PyPI",
        "fixed_field_scoring_contract_declared_before_scoring": EVAL_CONTRACT,
        "features": "P2/P1 Taylor-Hood, Newton, natural do-nothing outlet, three Gmsh-refined levels; no retraining",
        "reference_accepted": False,
        "scientific_accepted": False,
        "source_hashes": {
            "entrypoint_sha256": digest(RUN / "closure_dfg_followup_retry5.py"),
            "solver_sha256": digest(RUN / "solve_dfg.py"),
            "mesh_geo_sha256": digest(ROOT / "var/agent_runs/baseline-competitive-2026-09-27/s1/make_mesh.geo"),
        },
    }
    (RUN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
