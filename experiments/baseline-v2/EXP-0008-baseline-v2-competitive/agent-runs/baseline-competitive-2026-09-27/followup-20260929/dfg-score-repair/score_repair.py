#!/usr/bin/env python3
"""Reconstruct the saved DFG vector field and score frozen checkpoints on Kaggle."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

STARTED = time.perf_counter()
STARTED_UTC = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
SESSION_CAP = 351.814
SLUG = "rodrimandub/af-pikans-dfg-field-score-repair-retry-7-20260929"
OUT = Path("/kaggle/working/dfg-field-score-repair")
CONTRACT = {
    "grid": {
        "source": "saved FEM DFG-L3 fields_on_common_grid.npz; declared before checkpoint scoring",
        "x": {"start": 0.0, "stop": 2.2, "count": 221, "endpoint": True},
        "y": {"start": 0.0, "stop": 0.41, "count": 42, "endpoint": True},
        "ordering": "numpy.meshgrid(indexing='xy'); row-major flattened",
        "mask": "exclude points with (x-0.2)^2+(y-0.2)^2 <= 0.05^2",
    },
    "pressure_gauge": {"method": "subtract pressure at fixed outlet grid point", "outlet_anchor_xy": [2.2, 0.2]},
    "metrics": ["u_relative_l2", "v_relative_l2", "p_relative_l2_gauge_fixed_outlet"],
    "cohorts": {"current_rbf": [301, 302, 303, 304, 305], "refinement_rbf_and_tanh_mlp": [501, 502, 503, 504, 505]},
    "reference_level": "DFG-L3 (third of three requested mesh levels)",
    "retraining": False,
}


def elapsed() -> float:
    return time.perf_counter() - STARTED


def check_cap(reserve: float = 10.0) -> None:
    if elapsed() >= SESSION_CAP - reserve:
        raise TimeoutError(f"DFG score-repair cap reached with {reserve:.1f}s finalization reserve")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def status(name: str, state: str, **extra) -> None:
    write_json(OUT / f"{name}.json", {
        "part": name, "status": state, "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_elapsed_seconds": round(elapsed(), 3), "scientific_accepted": False,
        "reference_accepted": False, **extra,
    })


def find_one(pattern: str, *, must_contain: str, level: str | None = None) -> Path:
    check_cap()
    level_fragment = None if level is None else level.lower()
    matches = []
    for path in Path("/kaggle/input").rglob(pattern):
        normalized = str(path).replace("\\", "/").lower()
        if must_contain.lower() in normalized and (level_fragment is None or level_fragment in normalized):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"No input matched {pattern!r} containing {must_contain!r}")
    return sorted(matches, key=lambda p: (len(str(p)), str(p)))[0]


def coordinate_map(mesh_xy, dof_xyz, np, cKDTree, tolerance=1.0e-6):
    tree = cKDTree(mesh_xy)
    distances, mapped = tree.query(dof_xyz[:, :2], k=1, workers=2)
    mapped = np.asarray(mapped, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    unique_count = int(np.unique(mapped).size)
    diagnostics = {
        "count": int(len(mapped)),
        "unique_mesh_nodes": unique_count,
        "max_nearest_distance": float(distances.max(initial=0.0)),
        "p99_nearest_distance": float(np.quantile(distances, 0.99)) if len(distances) else 0.0,
        "distance_tolerance": float(tolerance),
    }
    if diagnostics["max_nearest_distance"] > diagnostics["distance_tolerance"]:
        raise ValueError(f"FEM DOF coordinates are not Gmsh nodes within tolerance: {diagnostics}")
    if unique_count != len(mapped):
        raise ValueError(f"Nearest-node map is not one-to-one: {diagnostics}")
    return mapped, diagnostics


def robust_coordinate_map(mesh_xy, mesh_geometry, dof_xyz, np, cKDTree):
    try:
        dof_to_geometry, dof_geometry_diag = coordinate_map(
            mesh_geometry[:, :2], dof_xyz, np, cKDTree, tolerance=1.0e-6
        )
        geometry_to_mesh, geometry_mesh_diag = coordinate_map(
            mesh_xy, mesh_geometry[:, :2], np, cKDTree, tolerance=1.0e-6
        )
        return geometry_to_mesh[dof_to_geometry], {
            "route": "saved_DOLFINx_geometry",
            "dof_to_geometry": dof_geometry_diag,
            "geometry_to_mesh": geometry_mesh_diag,
        }
    except ValueError as exact_route_error:
        mapped, diagnostics = coordinate_map(
            mesh_xy, dof_xyz, np, cKDTree, tolerance=1.6e-3
        )
        diagnostics["route"] = "nearest_mesh_node_with_saved_trace_check"
        diagnostics["exact_geometry_route_error"] = str(exact_route_error)
        return mapped, diagnostics


def p2_basis(r: float, s: float, np):
    l1, l2, l3 = 1.0 - r - s, r, s
    values = np.asarray([
        l1 * (2.0 * l1 - 1.0), l2 * (2.0 * l2 - 1.0), l3 * (2.0 * l3 - 1.0),
        4.0 * l1 * l2, 4.0 * l2 * l3, 4.0 * l3 * l1,
    ])
    dr = np.asarray([1.0 - 4.0 * l1, 4.0 * l2 - 1.0, 0.0, 4.0 * (l1 - l2), 4.0 * l3, -4.0 * l3])
    ds = np.asarray([1.0 - 4.0 * l1, 0.0, 4.0 * l3 - 1.0, -4.0 * l2, 4.0 * l2, 4.0 * (l1 - l3)])
    return values, dr, ds


def barycentric(point, triangle, np):
    matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
    try:
        r, s = np.linalg.solve(matrix, point - triangle[0])
    except np.linalg.LinAlgError:
        return None
    weights = np.asarray([1.0 - r - s, r, s])
    if weights.min() < -1.0e-8 or weights.max() > 1.0 + 1.0e-8:
        return None
    return float(r), float(s)


def evaluate_p2_vector(mesh_xy, tri6, dof_values, query_xy, np, cKDTree):
    corners = mesh_xy[tri6[:, :3]]
    centers = corners.mean(axis=1)
    tree = cKDTree(centers)
    k = min(64, len(tri6))
    _, nearest = tree.query(query_xy, k=k, workers=2)
    if k == 1:
        nearest = nearest[:, None]
    output = np.empty((len(query_xy), dof_values.shape[1]), dtype=np.float64)
    for point_index, point in enumerate(query_xy):
        if point_index % 256 == 0:
            check_cap()
        found = None
        for element_index in np.atleast_1d(nearest[point_index]):
            initial = barycentric(point, corners[int(element_index)], np)
            if initial is not None:
                found = int(element_index), initial
                break
        if found is None:
            _, extra = tree.query(point, k=min(256, len(tri6)), workers=2)
            for element_index in np.atleast_1d(extra):
                initial = barycentric(point, corners[int(element_index)], np)
                if initial is not None:
                    found = int(element_index), initial
                    break
        if found is None:
            raise ValueError(f"No quadratic triangle contains fixed grid point {point.tolist()}")
        element_index, (r, s) = found
        element_nodes = tri6[element_index]
        element_xy = mesh_xy[element_nodes]
        for _ in range(5):
            basis, dr, ds = p2_basis(r, s, np)
            mapped = basis @ element_xy
            jacobian = np.column_stack((dr @ element_xy, ds @ element_xy))
            delta = np.linalg.solve(jacobian, point - mapped)
            r += float(delta[0])
            s += float(delta[1])
            if max(abs(delta[0]), abs(delta[1])) < 1.0e-13:
                break
        basis, _, _ = p2_basis(r, s, np)
        output[point_index] = basis @ dof_values[element_nodes]
    return output


def evaluate_p1_scalar(mesh_xy, tri6, values_by_node, query_xy, refs, np):
    out = np.empty(len(query_xy), dtype=np.float64)
    for i, (element_index, r, s) in enumerate(refs):
        nodes = tri6[element_index, :3]
        out[i] = (1.0 - r - s) * values_by_node[nodes[0]] + r * values_by_node[nodes[1]] + s * values_by_node[nodes[2]]
    return out


def locate_reference_points(mesh_xy, tri6, query_xy, np, cKDTree):
    corners = mesh_xy[tri6[:, :3]]
    tree = cKDTree(corners.mean(axis=1))
    _, nearest = tree.query(query_xy, k=min(64, len(tri6)), workers=2)
    if nearest.ndim == 1:
        nearest = nearest[:, None]
    refs = []
    for point_index, point in enumerate(query_xy):
        if point_index % 256 == 0:
            check_cap()
        found = None
        for element_index in nearest[point_index]:
            element_index = int(element_index)
            initial = barycentric(point, corners[element_index], np)
            if initial is not None:
                found = (element_index, *initial)
                break
        if found is None:
            _, extra = tree.query(point, k=min(256, len(tri6)), workers=2)
            for element_index in np.atleast_1d(extra):
                element_index = int(element_index)
                initial = barycentric(point, corners[element_index], np)
                if initial is not None:
                    found = (element_index, *initial)
                    break
        if found is None:
            raise ValueError(f"No triangle contains fixed grid point {point.tolist()}")
        element_index, r, s = found
        element_xy = mesh_xy[tri6[element_index]]
        for _ in range(5):
            basis, dr, ds = p2_basis(r, s, np)
            mapped = basis @ element_xy
            jacobian = np.column_stack((dr @ element_xy, ds @ element_xy))
            delta = np.linalg.solve(jacobian, point - mapped)
            r += float(delta[0])
            s += float(delta[1])
            if max(abs(delta[0]), abs(delta[1])) < 1.0e-13:
                break
        refs.append((element_index, r, s))
    return refs


def summary_stats(values, np):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None, "n": int(len(values))}


def load_model(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def locate_checkpoint(seed, arm, roots, np):
    tag = "DFG-RBF-PIKAN" if arm == "rbf" else "DFG-Tanh-MLP"
    candidates = []
    for root in roots:
        for path in root.rglob("checkpoint-lbfgs.pt"):
            value = str(path).lower()
            if f"seed-{seed}" not in value:
                continue
            if seed <= 305:
                if arm == "rbf" and "dfg-rbf-refinement" in value:
                    candidates.append(path)
            elif "dfg-lbfgs-refinement" in value and tag.lower() in value:
                candidates.append(path)
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0] if candidates else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contract_record = {
        "schema_version": 1, "session": "DFG-field-score-repair-retry-7-2026-09-29", "kernel_slug": SLUG,
        "status": "starting", "push_cap_seconds": SESSION_CAP, "fixed_field_scoring_contract": CONTRACT,
        "source_session": "rodrimandub/af-pikans-dfg-fem-field-follow-up-retry-5-20260929",
        "method": "reconstruct both P2 velocity components from saved DOLFINx coefficients and the saved L3 Gmsh mesh; score frozen checkpoints only",
        "retraining": False, "reference_accepted": False, "scientific_accepted": False,
    }
    write_json(OUT / "manifest.json", contract_record)
    write_json(OUT / "checkpoint_field_scores.partial.json", {
        "status": "starting", "fixed_field_scoring_contract": CONTRACT, "per_seed": [],
        "scientific_accepted": False, "reference_accepted": False,
    })
    try:
        import numpy as np
        import torch
        from scipy.spatial import cKDTree

        if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("Kaggle T4x2 is required")
        devices = [torch.cuda.get_device_name(i) for i in range(2)]
        if any("T4" not in name.upper() for name in devices):
            raise RuntimeError(f"T4x2 required; visible devices={devices}")
        runtime = {"gpu_names": devices, "visible_gpu_count": torch.cuda.device_count(), "torch": torch.__version__, "numpy": np.__version__}
        if importlib.util.find_spec("meshio") is None:
            remaining = SESSION_CAP - elapsed() - 30.0
            if remaining <= 5:
                raise TimeoutError("No time remains for meshio setup")
            subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "meshio"],
                           check=True, timeout=min(75.0, remaining), stdout=(OUT / "setup.log").open("w"), stderr=subprocess.STDOUT)
        import meshio

        fields_path = find_one("fields_on_common_grid.npz", must_contain="dfg-fem-field-follow-up-retry-5-20260929", level="DFG-L3")
        coeff_path = find_one("fem_coefficients.npz", must_contain="dfg-fem-field-follow-up-retry-5-20260929", level="DFG-L3")
        mesh_path = find_one("dfg_re20_DFG-L3.msh", must_contain="dfg-fem-field-follow-up-retry-5-20260929", level="DFG-L3")
        fem_summary_path = find_one("summary.json", must_contain="dfg-fem-field-follow-up-retry-5-20260929")
        with np.load(fields_path, allow_pickle=False) as z:
            xy = np.asarray(z["xy"], dtype=np.float64)
            saved_velocity_component = np.asarray(z["velocity"], dtype=np.float64).reshape(-1)
            target_p = np.asarray(z["pressure"], dtype=np.float64).reshape(-1)
        with np.load(coeff_path, allow_pickle=False) as z:
            mesh_geometry = np.asarray(z["mesh_geometry"], dtype=np.float64)
            velocity_dof_xyz = np.asarray(z["velocity_dof_coordinates"], dtype=np.float64)
            velocity_coefficients = np.asarray(z["velocity_coefficients"], dtype=np.float64).reshape(-1)
            pressure_dof_xyz = np.asarray(z["pressure_dof_coordinates"], dtype=np.float64)
            pressure_coefficients = np.asarray(z["pressure_coefficients"], dtype=np.float64).reshape(-1)
        mesh = meshio.read(mesh_path)
        mesh_xy = np.asarray(mesh.points[:, :2], dtype=np.float64)
        tri6 = np.asarray(mesh.cells_dict["triangle6"], dtype=np.int64)
        if len(xy) != len(target_p) or len(saved_velocity_component) != len(xy):
            raise ValueError("Saved DFG-L3 point and scalar velocity arrays are inconsistent")
        if velocity_coefficients.size != 2 * len(velocity_dof_xyz):
            raise ValueError("P2 velocity coefficients do not match two components per saved dof")
        if pressure_coefficients.size != len(pressure_dof_xyz):
            raise ValueError("P1 pressure coefficients do not match saved dofs")
        if len(tri6) == 0 or tri6.shape[1] != 6:
            raise ValueError(f"Expected Gmsh quadratic triangles, got shape {tri6.shape}")
        source_shapes = {
            "meshio_point_count": int(len(mesh_xy)),
            "dolfinx_geometry_count": int(len(mesh_geometry)),
            "velocity_dof_coordinate_count": int(len(velocity_dof_xyz)),
            "velocity_coefficient_count": int(velocity_coefficients.size),
            "pressure_dof_coordinate_count": int(len(pressure_dof_xyz)),
            "pressure_coefficient_count": int(pressure_coefficients.size),
        }
        if len(mesh_geometry) != len(mesh_xy) or len(velocity_dof_xyz) != len(mesh_xy):
            raise ValueError(
                f"L3 geometry/velocity coordinate counts do not match the selected mesh: "
                f"{source_shapes}; fields={fields_path}; coefficients={coeff_path}; mesh={mesh_path}"
            )

        velocity_node_map, velocity_mapping_diagnostics = robust_coordinate_map(
            mesh_xy, mesh_geometry, velocity_dof_xyz, np, cKDTree
        )
        pressure_node_map, pressure_mapping_diagnostics = robust_coordinate_map(
            mesh_xy, mesh_geometry, pressure_dof_xyz, np, cKDTree
        )
        n_velocity = len(velocity_dof_xyz)
        layouts = {
            "interleaved_by_node": velocity_coefficients.reshape(n_velocity, 2),
            "component_blocks": np.column_stack((velocity_coefficients[:n_velocity], velocity_coefficients[n_velocity:])),
        }
        best = None
        candidates = []
        for edge_order in itertools.permutations((3, 4, 5)):
            candidate_tri6 = tri6[:, (0, 1, 2, *edge_order)]
            for layout_name, dof_values in layouts.items():
                values_by_node = np.full((len(mesh_xy), 2), np.nan, dtype=np.float64)
                values_by_node[velocity_node_map] = dof_values
                grid_values = evaluate_p2_vector(mesh_xy, candidate_tri6, values_by_node, xy, np, cKDTree)
                for component in range(2):
                    relative_match = float(np.linalg.norm(grid_values[:, component] - saved_velocity_component) /
                                           max(np.linalg.norm(saved_velocity_component), 1e-30))
                    candidate = {
                        "relative_match": relative_match if math.isfinite(relative_match) else None,
                        "non_finite_prediction_values": int(np.size(grid_values) - np.isfinite(grid_values).sum()),
                        "coefficient_layout": layout_name,
                        "saved_scalar_velocity_component": component,
                        "gmsh_edge_node_order": list(edge_order),
                    }
                    candidates.append(candidate)
                    if math.isfinite(relative_match):
                        record = (relative_match, layout_name, component, grid_values, edge_order, candidate_tri6)
                        if best is None or record[0] < best[0]:
                            best = record
        if best is None or best[0] > 5.0e-3:
            diagnostics = {
                "status": "field_reconstruction_failed",
                "fixed_field_scoring_contract": CONTRACT,
                "velocity_mapping": velocity_mapping_diagnostics,
                "pressure_mapping": pressure_mapping_diagnostics,
                "candidate_reconstruction_errors": sorted(
                    candidates,
                    key=lambda item: math.inf if item["relative_match"] is None else item["relative_match"],
                ),
                "mesh_point_count": int(len(mesh_xy)),
                "triangle6_cell_count": int(len(tri6)),
                "triangle6_unique_node_count": int(np.unique(tri6).size),
                "velocity_mapped_node_count": int(len(velocity_node_map)),
                "pressure_mapped_node_count": int(len(pressure_node_map)),
                "scientific_accepted": False,
                "reference_accepted": False,
            }
            write_json(OUT / "checkpoint_field_scores.partial.json", diagnostics)
            raise RuntimeError(f"P2 vector reconstruction did not reproduce the saved scalar trace: {None if best is None else best[0]}")
        _, selected_layout, saved_component, reconstructed_velocity, selected_edge_order, tri6 = best
        target_uv = reconstructed_velocity[:, [saved_component, 1 - saved_component]]

        pressure_by_node = np.full(len(mesh_xy), np.nan, dtype=np.float64)
        pressure_by_node[pressure_node_map] = pressure_coefficients
        refs = locate_reference_points(mesh_xy, tri6, xy, np, cKDTree)
        pressure_reconstructed = evaluate_p1_scalar(mesh_xy, tri6, pressure_by_node, xy, refs, np)
        pressure_match = float(np.linalg.norm(pressure_reconstructed - target_p) / max(np.linalg.norm(target_p), 1e-30))
        if pressure_match > 5.0e-3:
            raise RuntimeError(f"P1 pressure reconstruction differs from saved trace by {pressure_match}")
        target_p = pressure_reconstructed
        anchor_xy = np.asarray(CONTRACT["pressure_gauge"]["outlet_anchor_xy"], dtype=np.float64)
        anchor_index = int(np.argmin(np.linalg.norm(xy - anchor_xy[None, :], axis=1)))
        target_p = target_p - target_p[anchor_index]
        reconstruction = {
            "method": "P2 isoparametric interpolation from saved vector coefficients and MSH triangle6 cells; P1 isoparametric pressure interpolation",
            "source_artifact_paths": {
                "fields": str(fields_path),
                "coefficients": str(coeff_path),
                "mesh": str(mesh_path),
            },
            "velocity_coefficient_layout": selected_layout,
            "saved_scalar_velocity_component": int(saved_component),
            "gmsh_edge_node_order": list(selected_edge_order),
            "saved_scalar_trace_relative_reconstruction_error": float(best[0]),
            "pressure_trace_relative_reconstruction_error": pressure_match,
            "coordinate_mapping": {
                "velocity": velocity_mapping_diagnostics,
                "pressure": pressure_mapping_diagnostics,
            },
            "mesh_file_sha256": digest(mesh_path),
            "coefficient_file_sha256": digest(coeff_path),
            "source_field_file_sha256": digest(fields_path),
        }
        np.savez_compressed(OUT / "DFG-L3-fields-reconstructed.npz", xy=xy, velocity=target_uv, pressure_gauge_fixed=target_p)
        write_json(OUT / "checkpoint_field_scores.partial.json", {
            "status": "scoring", "fixed_field_scoring_contract": CONTRACT, "reconstruction": reconstruction,
            "per_seed": [], "scientific_accepted": False, "reference_accepted": False,
        })

        roots = [Path("/kaggle/input")]
        current_source = find_one("dfg.py", must_contain="af-pikans-dfg-rbf-refinement-20260928")
        old_source = find_one("run_campaign_source.py", must_contain="af-pikans-dfg-lbfgs-refinement-20260927")
        current_module = load_model(current_source, "closure_dfg_current_repair")
        old_module = load_model(old_source, "closure_dfg_old_repair")
        points = torch.as_tensor(xy, dtype=torch.float32, device="cuda:0")
        rows = []
        schedule = [(seed, "rbf") for seed in (301, 302, 303, 304, 305)]
        schedule += [(seed, arm) for seed in (501, 502, 503, 504, 505) for arm in ("rbf", "mlp")]
        for index, (seed, arm) in enumerate(schedule):
            if elapsed() >= SESSION_CAP - 15.0:
                rows.extend({"seed": later_seed, "arm": later_arm, "status": "session_time_cap_not_started"}
                            for later_seed, later_arm in schedule[index:])
                break
            checkpoint = locate_checkpoint(seed, arm, roots, np)
            if checkpoint is None:
                rows.append({"seed": seed, "arm": arm, "status": "missing_checkpoint"})
            else:
                try:
                    if seed <= 305:
                        model = current_module.DFGRBFPIKAN(seed=seed).to("cuda:0")
                    else:
                        name = "DFG-RBF-PIKAN" if arm == "rbf" else "DFG-Tanh-MLP"
                        model = old_module.build_model(name, seed).to("cuda:0")
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    model.load_state_dict(payload["state_dict"], strict=True)
                    model.eval()
                    parts = []
                    with torch.no_grad():
                        for start in range(0, len(points), 4096):
                            parts.append(model(points[start:start + 4096]).detach().cpu().numpy())
                    pred = np.concatenate(parts, axis=0)
                    if pred.ndim != 2 or pred.shape[1] != 3:
                        raise ValueError(f"Expected checkpoint output (n,3), got {pred.shape}")
                    pred_p = pred[:, 2] - pred[anchor_index, 2]
                    row = {
                        "seed": seed, "arm": arm, "status": "scored", "checkpoint": str(checkpoint),
                        "checkpoint_sha256": digest(checkpoint),
                        "u_relative_l2": float(np.linalg.norm(pred[:, 0] - target_uv[:, 0]) / max(np.linalg.norm(target_uv[:, 0]), 1e-30)),
                        "v_relative_l2": float(np.linalg.norm(pred[:, 1] - target_uv[:, 1]) / max(np.linalg.norm(target_uv[:, 1]), 1e-30)),
                        "p_relative_l2_gauge_fixed_outlet": float(np.linalg.norm(pred_p - target_p) / max(np.linalg.norm(target_p), 1e-30)),
                    }
                    rows.append(row)
                    del model, payload, parts, pred
                    torch.cuda.empty_cache()
                except Exception as exc:
                    rows.append({"seed": seed, "arm": arm, "status": "scoring_error",
                                 "checkpoint": str(checkpoint), "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()[-2500:]})
            write_json(OUT / "checkpoint_field_scores.partial.json", {
                "status": "scoring", "fixed_field_scoring_contract": CONTRACT, "reconstruction": reconstruction,
                "per_seed": rows, "scientific_accepted": False, "reference_accepted": False,
            })

        metrics = {}
        for cohort, seeds, arms in (("current_301_305", (301, 302, 303, 304, 305), ("rbf",)),
                                    ("older_501_505", (501, 502, 503, 504, 505), ("rbf", "mlp"))):
            metrics[cohort] = {}
            for arm in arms:
                for key in ("u_relative_l2", "v_relative_l2", "p_relative_l2_gauge_fixed_outlet"):
                    vals = [r[key] for r in rows if r.get("seed") in seeds and r.get("arm") == arm and r.get("status") == "scored"]
                    metrics[cohort].setdefault(arm, {})[key] = summary_stats(vals, np) if vals else {"mean": None, "sample_sd": None, "n": 0}
        lookup = {(r["arm"], r["seed"]): r for r in rows if r.get("status") == "scored" and r.get("seed") in (501, 502, 503, 504, 505)}
        paired = []
        for seed in (501, 502, 503, 504, 505):
            if ("rbf", seed) in lookup and ("mlp", seed) in lookup:
                paired.append({"seed": seed, **{key: lookup[("rbf", seed)][key] - lookup[("mlp", seed)][key]
                                                for key in ("u_relative_l2", "v_relative_l2", "p_relative_l2_gauge_fixed_outlet")}})
        result = {
            "status": "complete" if sum(r.get("status") == "scored" for r in rows) == 15 else "partial",
            "fixed_field_scoring_contract": CONTRACT, "reconstruction": reconstruction,
            "metrics": metrics, "paired_seed_differences_RBF_minus_MLP_501_505": paired, "per_seed": rows,
            "reference_accepted": False, "scientific_accepted": False,
        }
        write_json(OUT / "checkpoint_field_scores.json", result)
        with (OUT / "checkpoint_field_scores.csv").open("w", newline="", encoding="utf-8") as stream:
            columns = ["seed", "arm", "status", "u_relative_l2", "v_relative_l2", "p_relative_l2_gauge_fixed_outlet", "checkpoint_sha256", "checkpoint", "error"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        source_manifest = json.loads(fem_summary_path.read_text(encoding="utf-8"))
        status("score_repair", result["status"], actual_session_seconds=round(elapsed(), 3),
               scored_count=sum(r.get("status") == "scored" for r in rows), scientific_accepted=False)
        artifacts = []
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                artifacts.append({"path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size, "sha256": digest(path)})
        contract_record.update({
            "status": result["status"], "actual_session_seconds": round(elapsed(), 3),
            "started_at_utc": STARTED_UTC,
            "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_code_sha256": digest(Path(__file__)), "artifacts": artifacts,
            "runtime": runtime, "source_fem_summary": {
                "status": source_manifest.get("status"),
                "evidence_ready_for_human_gate": source_manifest.get("evidence_ready_for_human_gate"),
                "reference_qoi_errors_vs_featflow": source_manifest.get("reference_qoi_errors_vs_featflow"),
                "two_finest_mesh_relative_changes": source_manifest.get("two_finest_mesh_relative_changes"),
            },
            "reconstruction": reconstruction, "metrics": metrics,
            "paired_seed_differences_RBF_minus_MLP_501_505": paired,
            "reference_accepted": False, "scientific_accepted": False,
        })
        write_json(OUT / "manifest.json", contract_record)
    except Exception as exc:
        partial_path = OUT / "checkpoint_field_scores.partial.json"
        partial_record = {}
        if partial_path.is_file():
            try:
                partial_record = json.loads(partial_path.read_text(encoding="utf-8"))
            except Exception:
                partial_record = {}
        partial_record.update({
            "status": "failed_partial", "fixed_field_scoring_contract": CONTRACT,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-6000:],
            "scientific_accepted": False, "reference_accepted": False,
        })
        write_json(partial_path, partial_record)
        contract_record.update({"status": "failed_partial", "actual_session_seconds": round(elapsed(), 3),
                                "started_at_utc": STARTED_UTC, "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "source_code_sha256": digest(Path(__file__)),
                                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-6000:]})
        status("score_repair", "failed_partial", actual_session_seconds=round(elapsed(), 3),
               error=f"{type(exc).__name__}: {exc}", scientific_accepted=False)
        artifacts = []
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                artifacts.append({"path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size, "sha256": digest(path)})
        contract_record["artifacts"] = artifacts
        write_json(OUT / "manifest.json", contract_record)


if __name__ == "__main__":
    main()
