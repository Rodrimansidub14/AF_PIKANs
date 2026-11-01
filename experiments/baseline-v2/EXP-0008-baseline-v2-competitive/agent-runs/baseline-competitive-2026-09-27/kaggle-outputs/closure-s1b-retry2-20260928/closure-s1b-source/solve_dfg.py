#!/usr/bin/env python3
"""DRAFT_UNVERIFIED stationary Taylor–Hood solver for DFG 2D-1 (Re=20).

This source is a protocol-aligned draft. It has not been run against a resolved
DOLFINx/Gmsh/PETSc environment and cannot produce an accepted reference until
the validation gaps in README.md are closed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from basix.ufl import element, mixed_element
from dolfinx import fem, geometry, io
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from petsc4py import PETSc
import ufl

STATUS = "DRAFT_UNVERIFIED"
ROOT = Path(__file__).resolve().parent
GEO = ROOT / "make_mesh.geo"
L, H = 2.2, 0.41
CX, CY, R = 0.2, 0.2, 0.05
RHO, NU = 1.0, 0.001
UMAX, UMEAN, DIAMETER = 0.3, 0.2, 0.1
TAGS = {"inlet": 1, "walls": 2, "cylinder": 3, "outlet": 4, "fluid": 5}
LEVELS = [
    {"id": "DFG-L0", "bulk": 0.040, "cylinder": 0.0060},
    {"id": "DFG-L1", "bulk": 0.020, "cylinder": 0.0030},
    {"id": "DFG-L2", "bulk": 0.010, "cylinder": 0.0015},
    {"id": "DFG-L3", "bulk": 0.005, "cylinder": 0.00075},
    {"id": "DFG-L4", "bulk": 0.0025, "cylinder": 0.000375},
    {"id": "DFG-L5", "bulk": 0.00125, "cylinder": 0.0001875},
]
FEATFLOW = {"Cd": 5.57953523384, "Cl": 0.010618948146, "delta_p": 0.11752016697}


def make_mesh(level: dict, mesh_dir: Path, gmsh_bin: str) -> Path:
    """Generate one curved, quadratic Gmsh mesh from the shared CAD source."""
    mesh_dir.mkdir(parents=True, exist_ok=True)
    msh = mesh_dir / f"dfg_re20_{level['id']}.msh"
    command = [
        gmsh_bin, str(GEO), "-2",
        "-setnumber", "lc_bulk", str(level["bulk"]),
        "-setnumber", "lc_cyl", str(level["cylinder"]),
        "-o", str(msh),
    ]
    subprocess.run(command, check=True)
    return msh


def velocity_boundary_conditions(domain, facet_tags, W):
    """Impose the FEATFLOW parabolic inflow and zero velocity on solid walls."""
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    V, _ = W.sub(0).collapse()

    inlet = fem.Function(V, name="inlet_velocity")

    def profile(x):
        values = np.zeros((2, x.shape[1]), dtype=domain.geometry.x.dtype)
        y = x[1]
        values[0] = 4.0 * UMAX * y * (H - y) / (H * H)
        return values

    inlet.interpolate(profile)
    inlet.x.scatter_forward()
    zero = fem.Function(V, name="zero_velocity")
    zero.x.array[:] = 0.0
    zero.x.scatter_forward()

    def dofs(tag):
        return fem.locate_dofs_topological(
            (W.sub(0), V), fdim, facet_tags.find(TAGS[tag])
        )

    return [
        fem.dirichletbc(inlet, dofs("inlet"), W.sub(0)),
        fem.dirichletbc(zero, dofs("walls"), W.sub(0)),
        fem.dirichletbc(zero, dofs("cylinder"), W.sub(0)),
    ]


def solve_stationary(domain, facet_tags):
    """Solve Stokes first, then continue the convective coefficient to one."""
    P2 = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    P1 = element("Lagrange", domain.basix_cell(), 1)
    W = fem.functionspace(domain, mixed_element([P2, P1]))
    state = fem.Function(W, name="mixed_velocity_pressure")
    tests = ufl.TestFunctions(W)
    velocity, pressure = ufl.split(state)
    test_velocity, test_pressure = tests
    alpha = fem.Constant(domain, PETSc.ScalarType(0.0))
    viscosity = fem.Constant(domain, PETSc.ScalarType(NU))
    dx = ufl.Measure("dx", domain=domain)

    # This is the weak form fixed in reference-protocol-ac-dfg.md. There is no
    # outlet boundary term or outlet Dirichlet condition: the natural do-nothing
    # condition is the free traction on the open downstream boundary.
    residual = (
        RHO * alpha * ufl.inner(ufl.dot(ufl.grad(velocity), velocity), test_velocity)
        + viscosity * ufl.inner(ufl.grad(velocity), ufl.grad(test_velocity))
        - pressure * ufl.div(test_velocity)
        + test_pressure * ufl.div(velocity)
    ) * dx
    bcs = velocity_boundary_conditions(domain, facet_tags, W)
    options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-10,
        "snes_atol": 1.0e-12,
        "snes_stol": 1.0e-12,
        "snes_max_it": 80,
        "snes_error_if_not_converged": True,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }
    problem = fem.petsc.NonlinearProblem(
        residual,
        state,
        bcs=bcs,
        petsc_options_prefix="dfg_re20_",
        petsc_options=options,
    )

    # alpha=0 is the Stokes initialisation. Failed continuation steps are logged
    # and retried at half step; do not silently advance past a failed stage.
    stage_log = []
    last_alpha = 0.0
    step = 0.25
    state.x.array[:] = 0.0
    state.x.scatter_forward()

    def attempt(target: float) -> bool:
        nonlocal state
        previous = state.x.array.copy()
        alpha.value = PETSc.ScalarType(target)
        try:
            problem.solve()
            state.x.scatter_forward()
            reason = int(problem.solver.getConvergedReason())
            snes_iterations = int(problem.solver.getIterationNumber())
            nonlinear_norm = float(problem.solver.getFunctionNorm())
            ksp = problem.solver.getKSP()
            linear_reason = int(ksp.getConvergedReason())
            linear_norm = float(ksp.getResidualNorm())
            success = reason > 0 and np.isfinite(nonlinear_norm) and linear_reason > 0
            stage_log.append({
                "alpha": target,
                "accepted_by_solver": success,
                "snes_reason": reason,
                "snes_iterations": snes_iterations,
                "snes_final_norm": nonlinear_norm,
                "ksp_reason": linear_reason,
                "ksp_final_norm": linear_norm,
            })
        except Exception as exc:  # retain the failed stage in the record
            state.x.array[:] = previous
            state.x.scatter_forward()
            stage_log.append({"alpha": target, "accepted_by_solver": False,
                              "exception": f"{type(exc).__name__}: {exc}"})
            return False
        if not success:
            state.x.array[:] = previous
            state.x.scatter_forward()
        return success

    if not attempt(0.0):
        raise RuntimeError("Stokes initialisation failed; see stage log in the exception context")
    reached = 0.0
    while reached < 1.0 - 1.0e-14:
        target = min(1.0, reached + step)
        if attempt(target):
            reached = target
            step = min(0.25, step * 2.0)
        else:
            step *= 0.5
            if step < 0.015625:
                raise RuntimeError(f"Continuation stalled at alpha={reached}; stages={stage_log}")

    state.x.scatter_forward()
    return W, state, stage_log


def point_values(function, domain, xy_points):
    """Locate and evaluate a function at fixed physical points (serial draft)."""
    points = np.zeros((len(xy_points), 3), dtype=domain.geometry.x.dtype)
    points[:, :2] = xy_points
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)
    values = []
    for index in range(len(points)):
        cells = colliding.links(index)
        if len(cells) == 0:
            raise ValueError(f"Point not found in fluid mesh: {xy_points[index].tolist()}")
        cell = np.asarray([cells[0]], dtype=np.int32)
        values.append(function.eval(points[index:index + 1], cell)[0])
    return np.asarray(values)


def export_level(level, mesh_path: Path, output_dir: Path):
    """Solve one level and export the mesh, FE coefficients, sampled fields and QoIs."""
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("This draft intentionally supports serial runs only.")
    mesh_data = gmshio.read_from_msh(mesh_path, comm, rank=0, gdim=2)
    domain = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    if facet_tags is None:
        raise RuntimeError("Gmsh mesh has no boundary Physical Curve tags.")
    W, state, stages = solve_stationary(domain, facet_tags)
    velocity, _ = state.sub(0).collapse()
    pressure, _ = state.sub(1).collapse()
    velocity.name = "velocity"
    pressure.name = "pressure"

    # FEATFLOW defines sigma = nu*grad(u) - p I, with eta directed out of the
    # cylinder. The mesh's exterior normal points into the cylinder, so eta=-n.
    _, p_symbol = ufl.split(state)
    u_symbol, _ = ufl.split(state)
    normal_cylinder = -ufl.FacetNormal(domain)
    stress = viscosity_for_qoi(domain) * ufl.grad(u_symbol) - p_symbol * ufl.Identity(2)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    traction = ufl.dot(stress, normal_cylinder)
    local_drag = fem.assemble_scalar(fem.form(traction[0] * ds(TAGS["cylinder"])))
    local_lift = fem.assemble_scalar(fem.form(traction[1] * ds(TAGS["cylinder"])))
    drag_force = comm.allreduce(float(local_drag), op=MPI.SUM)
    lift_force = comm.allreduce(float(local_lift), op=MPI.SUM)
    scale = 2.0 / (RHO * UMEAN * UMEAN * DIAMETER)

    # The two prescribed pressure probes lie on the cylinder boundary. The
    # point locator is deliberately exposed as an unverified one-sided trace.
    probes = np.asarray([[0.15, 0.2], [0.25, 0.2]], dtype=np.float64)
    probe_pressure = point_values(pressure, domain, probes).reshape(-1)
    delta_p = float(probe_pressure[0] - probe_pressure[1])

    xg = np.linspace(0.0, L, 221)
    yg = np.linspace(0.0, H, 42)
    xx, yy = np.meshgrid(xg, yg, indexing="xy")
    grid_xy = np.column_stack((xx.ravel(), yy.ravel()))
    fluid_mask = ((grid_xy[:, 0] - CX) ** 2 + (grid_xy[:, 1] - CY) ** 2) > R ** 2
    grid_xy = grid_xy[fluid_mask]
    velocity_grid = point_values(velocity, domain, grid_xy)
    pressure_grid = point_values(pressure, domain, grid_xy).reshape(-1)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "fields_on_common_grid.npz",
        xy=grid_xy,
        velocity=velocity_grid,
        pressure=pressure_grid,
        pressure_probe_xy=probes,
        pressure_probe_values=probe_pressure,
    )
    np.savez_compressed(
        output_dir / "fem_coefficients.npz",
        mesh_geometry=domain.geometry.x,
        velocity_dof_coordinates=W.sub(0).collapse()[0].tabulate_dof_coordinates(),
        velocity_coefficients=velocity.x.array,
        pressure_dof_coordinates=W.sub(1).collapse()[0].tabulate_dof_coordinates(),
        pressure_coefficients=pressure.x.array,
        element_pair=np.asarray(["Lagrange-P2-vector", "Lagrange-P1-pressure"]),
    )
    qois = {
        "status": STATUS,
        "level": level["id"],
        "drag_force": drag_force,
        "lift_force": lift_force,
        "Cd": scale * drag_force,
        "Cl": scale * lift_force,
        "pressure_front": float(probe_pressure[0]),
        "pressure_rear": float(probe_pressure[1]),
        "delta_p_front_minus_rear": delta_p,
        "featflow_reference": FEATFLOW,
        "snes_stages": stages,
        "cell_count": int(domain.topology.index_map(domain.topology.dim).size_global),
        "geometry_nodes_local": int(domain.geometry.x.shape[0]),
        "dofs_global": int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs),
        "pressure_probe_policy": "direct point evaluation; boundary one-sided trace not yet independently verified",
        "reference_accepted": False,
    }
    (output_dir / "qois.json").write_text(json.dumps(qois, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with io.XDMFFile(comm, str(output_dir / "mesh.xdmf"), "w") as xdmf:
        xdmf.write_mesh(domain)
    return qois


def viscosity_for_qoi(domain):
    return fem.Constant(domain, PETSc.ScalarType(NU))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    parser.add_argument("--gmsh", default="gmsh", help="Gmsh executable on PATH")
    parser.add_argument("--max-levels", type=int, default=3,
                        help="run the first 1..6 planned mesh levels")
    parser.add_argument("--wall-cap-seconds", type=float, default=1450.0,
                        help="worker wall cap; completed level outputs persist as partial evidence")
    parser.add_argument("--minimum-level-reserve-seconds", type=float, default=180.0,
                        help="do not start a new mesh unless this much cap remains")
    args = parser.parse_args()
    if not 1 <= args.max_levels <= len(LEVELS):
        raise ValueError("--max-levels must be between 1 and the number of declared levels")
    if args.wall_cap_seconds <= 0 or args.minimum_level_reserve_seconds < 0:
        raise ValueError("wall cap must be positive and reserve non-negative")
    levels = LEVELS[:args.max_levels]
    started = time.perf_counter()
    deadline = started + args.wall_cap_seconds
    results = []
    summary_path = args.output / "summary.json"
    environment = {
        "python": sys.version.split()[0],
        "dolfinx": getattr(__import__("dolfinx"), "__version__", "unknown"),
        "gmsh": getattr(__import__("gmsh"), "__version__", "unknown"),
        "petsc": str(PETSc.Sys.getVersion()),
        "ufl": getattr(ufl, "__version__", "unknown"),
    }

    def persist(status: str, *, error: str | None = None) -> None:
        qoi_errors = None
        mesh_change = None
        if results:
            final = results[-1]
            qoi_errors = {
                "Cd_relative": abs(final["Cd"] - FEATFLOW["Cd"]) / FEATFLOW["Cd"],
                "Cl_absolute": abs(final["Cl"] - FEATFLOW["Cl"]),
                "delta_p_relative": abs(final["delta_p_front_minus_rear"] - FEATFLOW["delta_p"]) / FEATFLOW["delta_p"],
            }
        if len(results) >= 2:
            coarse, fine = results[-2:]
            mesh_change = {
                "Cd_relative": abs(fine["Cd"] - coarse["Cd"]) / max(abs(fine["Cd"]), 1e-30),
                "Cl_relative": abs(fine["Cl"] - coarse["Cl"]) / max(abs(fine["Cl"]), 1e-30),
                "Cl_absolute": abs(fine["Cl"] - coarse["Cl"]),
                "delta_p_relative": abs(fine["delta_p_front_minus_rear"] - coarse["delta_p_front_minus_rear"]) / max(abs(fine["delta_p_front_minus_rear"]), 1e-30),
            }
        evidence_ready = bool(
            status == "candidate_complete" and len(results) >= 3 and qoi_errors and mesh_change
            and qoi_errors["Cd_relative"] <= 0.005
            and qoi_errors["Cl_absolute"] <= 0.0005
            and qoi_errors["delta_p_relative"] <= 0.005
            and mesh_change["Cd_relative"] < 0.002
            and mesh_change["Cl_relative"] < 0.002
            and mesh_change["delta_p_relative"] < 0.002
        )
        payload = {
            "status": status,
            "reference_accepted": False,
            "scientific_accepted": False,
            "evidence_ready_for_human_gate": evidence_ready,
            "session_cap_seconds": float(args.wall_cap_seconds),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "planned_level_count": len(levels),
            "completed_level_count": len(results),
            "environment_versions": environment,
            "reference_qoi_errors_vs_featflow": qoi_errors,
            "two_finest_mesh_relative_changes": mesh_change,
            "levels": results,
        }
        if error:
            payload["error"] = error
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(summary_path)

    args.output.mkdir(parents=True, exist_ok=True)
    persist("running")
    try:
        for level in levels:
            if deadline - time.perf_counter() < args.minimum_level_reserve_seconds:
                persist("time_cap_partial")
                break
            mesh_path = make_mesh(level, args.output / "meshes", args.gmsh)
            result = export_level(level, mesh_path, args.output / level["id"])
            results.append(result)
            persist("running")
        else:
            persist("candidate_complete")
    except Exception as exc:
        persist("failed_partial", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
