"""Conforming P2 FEM for the effective 2-D coupled PVDF bimorph (P5).

This script implements exactly the 2-D constitutive contract written in
capitulos/marco/04-piezo.tex and instantiated in
capitulos/metodologia/01-problemas.tex.  It is a didactic numerical reference:
it is not a 3-D material model, an experimental validation, or a claim about
the thesis results.

The mesh is two-layer, interface-aligned and made of quadratic Lagrange
triangles.  The unknown vector is interleaved as (d_x, d_y, phi) per P2 node.
The weak form is assembled as

    [ K_dd      K_dphi ] [d]   = 0
    [ K_phid  - K_phiphi] [phi] = 0,

with sigma = C s + e^T grad(phi) and D = e s - kappa grad(phi).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Keep every matplotlib cache and generated artifact inside the declared run.
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
from matplotlib.ticker import ScalarFormatter


# ---------------------------------------------------------------------------
# Exact P5 contract
# ---------------------------------------------------------------------------
L = 0.1
H = 0.001
V_TOP = 100.0

C11 = 2.1836e9
C22 = 2.1836e9
C12 = 0.6332e9
C66 = 0.775e9
E21 = 0.02904
E22 = 0.05157
KX = 1.062e-10
KY = 1.041e-10

C_MAT = np.array(
    [[C11, C12, 0.0], [C12, C22, 0.0], [0.0, 0.0, C66]], dtype=np.float64
)
KAPPA = np.diag([KX, KY]).astype(np.float64)

# The following scales are the fixed P5 scales written in methodology.
E0 = V_TOP / H
C0 = C11
E0_COUPLING = E22
K0 = max(KX, KY)
S0 = E0_COUPLING * E0 / C0
D0_SCALE_DISP = L * S0
SIGMA0 = C0 * D0_SCALE_DISP / H + E0_COUPLING * E0
D0 = E0_COUPLING * D0_SCALE_DISP / H + K0 * E0
R_M = SIGMA0 / H
R_E = D0 / H
PHI0 = V_TOP

# Seven-point Dunavant rule.  The weights sum to one on a unit-area triangle;
# each is multiplied by |det(J)|/2 below for the reference triangle area.
QR = np.array(
    [1.0 / 3.0, 0.470142064105115, 0.059715871789770, 0.470142064105115,
     0.101286507323456, 0.797426985353087, 0.101286507323456],
    dtype=np.float64,
)
QS = np.array(
    [1.0 / 3.0, 0.470142064105115, 0.470142064105115, 0.059715871789770,
     0.101286507323456, 0.101286507323456, 0.797426985353087],
    dtype=np.float64,
)
QW = np.array(
    [0.225000000000000, 0.132394152788506, 0.132394152788506,
     0.132394152788506, 0.125939180544827, 0.125939180544827,
     0.125939180544827],
    dtype=np.float64,
)


def shape_grad_reference(r: float, s: float) -> np.ndarray:
    """Return [dN/dr,dN/ds] for the six P2 triangle basis functions."""

    lam = np.array([1.0 - r - s, r, s], dtype=np.float64)
    dlam = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    grad = np.empty((6, 2), dtype=np.float64)
    grad[0] = (4.0 * lam[0] - 1.0) * dlam[0]
    grad[1] = (4.0 * lam[1] - 1.0) * dlam[1]
    grad[2] = (4.0 * lam[2] - 1.0) * dlam[2]
    grad[3] = 4.0 * (lam[0] * dlam[1] + lam[1] * dlam[0])
    grad[4] = 4.0 * (lam[1] * dlam[2] + lam[2] * dlam[1])
    grad[5] = 4.0 * (lam[2] * dlam[0] + lam[0] * dlam[2])
    return grad


def patch_checks() -> dict:
    """Check P2 gradient, rigid-motion and linear-strain patches."""

    mesh = build_mesh(4, 1)
    element = mesh["elements"][0]
    element_xy = mesh["nodes"][element[:3]]
    local_xy = mesh["nodes"][element]
    jac = np.column_stack((element_xy[1] - element_xy[0], element_xy[2] - element_xy[0]))
    inv_j = np.linalg.inv(jac)
    max_partition = 0.0
    max_translation = 0.0
    max_rotation = 0.0
    max_linear_strain = 0.0
    for r, s in zip(QR, QS):
        grad = shape_grad_reference(float(r), float(s)) @ inv_j
        max_partition = max(max_partition, float(np.max(np.abs(np.sum(grad, axis=0)))))
        B = np.zeros((3, 12), dtype=np.float64)
        B[0, :6] = grad[:, 0]
        B[1, 6:] = grad[:, 1]
        B[2, :6] = grad[:, 1]
        B[2, 6:] = grad[:, 0]
        max_translation = max(
            max_translation,
            float(np.max(np.abs(B @ np.r_[np.ones(6), np.zeros(6)]))),
            float(np.max(np.abs(B @ np.r_[np.zeros(6), np.ones(6)]))),
        )
        max_rotation = max(
            max_rotation,
            float(np.max(np.abs(B @ np.r_[-local_xy[:, 1], local_xy[:, 0]]))),
        )
        linear = B @ np.r_[local_xy[:, 0], local_xy[:, 1]]
        max_linear_strain = max(
            max_linear_strain,
            float(np.max(np.abs(linear - np.array([1.0, 1.0, 0.0])))),
        )
    return {
        "representative_element_detJ_m2": float(np.linalg.det(jac)),
        "gradient_partition_inf": max_partition,
        "rigid_translation_strain_inf": max_translation,
        "rigid_rotation_strain_inf": max_rotation,
        "linear_patch_strain_error_inf": max_linear_strain,
        "expected_local_dof_order": "(d_x[0:6], d_y[0:6], phi[0:6])",
        "global_dof_order": "(d_x,d_y,phi) per node",
    }


def build_mesh(nx: int, elements_per_layer_y: int) -> dict:
    """Create an interface-aligned P2 mesh on (0,L)x(0,H).

    The rectangle is split into 2*elements_per_layer_y horizontal rows.  Each
    cell is divided into two consistently oriented triangles.  P2 edge nodes
    are deduplicated by the unordered pair of their endpoint vertex ids.
    """

    ny = 2 * elements_per_layer_y
    hx = L / nx
    hy = H / ny

    vertices = np.array(
        [(i * hx, j * hy) for j in range(ny + 1) for i in range(nx + 1)],
        dtype=np.float64,
    )

    def vertex_id(i: int, j: int) -> int:
        return j * (nx + 1) + i

    triangles: list[tuple[int, int, int]] = []
    layers: list[int] = []
    for j in range(ny):
        for i in range(nx):
            bl = vertex_id(i, j)
            br = vertex_id(i + 1, j)
            tl = vertex_id(i, j + 1)
            tr = vertex_id(i + 1, j + 1)
            # Both orientations are counter-clockwise.
            triangles.extend([(bl, br, tr), (bl, tr, tl)])
            layer = 0 if j < elements_per_layer_y else 1
            layers.extend([layer, layer])

    nodes = [tuple(xy) for xy in vertices]
    edge_nodes: dict[tuple[int, int], int] = {}
    p2_elements: list[list[int]] = []
    for v0, v1, v2 in triangles:
        local = [v0, v1, v2]
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            key = (min(a, b), max(a, b))
            if key not in edge_nodes:
                edge_nodes[key] = len(nodes)
                nodes.append(tuple((vertices[a] + vertices[b]) / 2.0))
            local.append(edge_nodes[key])
        p2_elements.append(local)

    return {
        "vertices": vertices,
        "nodes": np.asarray(nodes, dtype=np.float64),
        "triangles": np.asarray(triangles, dtype=np.int64),
        "elements": np.asarray(p2_elements, dtype=np.int64),
        "layers": np.asarray(layers, dtype=np.int8),
        "nx": nx,
        "ny_layer": elements_per_layer_y,
        "hx": hx,
        "hy": hy,
    }


def constitutive_e(layer: int) -> np.ndarray:
    """Return the layer-specific 2x3 e matrix.

    layer=0 is the lower layer (+ signs), layer=1 is the upper layer (- signs).
    """

    eta = 1.0 if layer == 0 else -1.0
    return eta * np.array(
        [[0.0, 0.0, 0.0], [E21, E22, 0.0]], dtype=np.float64
    )


def element_stiffness(element_xy: np.ndarray, layer: int) -> np.ndarray:
    """Assemble one 18x18 P2 coupled element matrix in physical units."""

    x0, x1, x2 = element_xy
    jac = np.column_stack((x1 - x0, x2 - x0))
    det_j = float(np.linalg.det(jac))
    if det_j <= 0.0:
        raise ValueError(f"Non-positive element orientation: detJ={det_j}")
    inv_j = np.linalg.inv(jac)
    e_mat = constitutive_e(layer)
    local = np.zeros((18, 18), dtype=np.float64)

    for r, s, w_ref in zip(QR, QS, QW):
        # Row gradients transform as grad_x = grad_ref @ J^{-1}.
        grad = shape_grad_reference(float(r), float(s)) @ inv_j
        weight = float(w_ref * det_j / 2.0)
        B = np.zeros((3, 12), dtype=np.float64)
        B[0, 0:6] = grad[:, 0]
        B[1, 6:12] = grad[:, 1]
        B[2, 0:6] = grad[:, 1]
        B[2, 6:12] = grad[:, 0]
        G = grad.T  # (2,6), so grad(phi)=G @ phi_local

        local[:12, :12] += weight * (B.T @ C_MAT @ B)
        local[:12, 12:] += weight * (B.T @ e_mat.T @ G)
        local[12:, :12] += weight * (G.T @ e_mat @ B)
        local[12:, 12:] += weight * (-G.T @ KAPPA @ G)

    return local


def assemble(mesh: dict) -> coo_matrix:
    """Assemble the global physical-unit matrix from P2 element matrices."""

    n_nodes = len(mesh["nodes"])
    n_dof = 3 * n_nodes
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for element, layer in zip(mesh["elements"], mesh["layers"]):
        local = element_stiffness(mesh["nodes"][element[:3]], int(layer))
        # The element matrix is block ordered as
        # (d_x[0:6], d_y[0:6], phi[0:6]), while the global vector is
        # interleaved as (d_x,d_y,phi) per node.  Keep this explicit map;
        # interleaving here would mix the mechanical blocks.
        dofs = [3 * int(node) for node in element]
        dofs.extend(3 * int(node) + 1 for node in element)
        dofs.extend(3 * int(node) + 2 for node in element)
        dofs_arr = np.asarray(dofs, dtype=np.int64)
        rr = np.repeat(dofs_arr, len(dofs_arr))
        cc = np.tile(dofs_arr, len(dofs_arr))
        rows.extend(rr.tolist())
        cols.extend(cc.tolist())
        vals.extend(local.ravel().tolist())

    return coo_matrix((np.asarray(vals), (rows, cols)), shape=(n_dof, n_dof)).tocsr()


def boundary_conditions(mesh: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return constrained global dofs, their dimensionless values and masks."""

    xy = mesh["nodes"]
    tol = 1e-12 * max(L, H)
    left = np.flatnonzero(np.isclose(xy[:, 0], 0.0, atol=tol, rtol=0.0))
    bottom = np.flatnonzero(np.isclose(xy[:, 1], 0.0, atol=tol, rtol=0.0))
    top = np.flatnonzero(np.isclose(xy[:, 1], H, atol=tol, rtol=0.0))
    right = np.flatnonzero(np.isclose(xy[:, 0], L, atol=tol, rtol=0.0))

    prescribed: dict[int, float] = {}
    for node in left:
        prescribed[3 * int(node)] = 0.0
        prescribed[3 * int(node) + 1] = 0.0
    for node in bottom:
        prescribed[3 * int(node) + 2] = 0.0
    for node in top:
        prescribed[3 * int(node) + 2] = 1.0  # phi / V_TOP

    dofs = np.array(sorted(prescribed), dtype=np.int64)
    values = np.array([prescribed[int(d)] for d in dofs], dtype=np.float64)
    mask = np.zeros(3 * len(xy), dtype=bool)
    mask[dofs] = True
    return dofs, values, {
        "left_nodes": left,
        "bottom_nodes": bottom,
        "top_nodes": top,
        "right_nodes": right,
        "dof_mask": mask,
    }


def scale_matrix(matrix: coo_matrix, n_nodes: int):
    """Apply the fixed P5 row/column scales to improve mixed-unit conditioning."""

    col_scale = np.tile(np.array([D0_SCALE_DISP, D0_SCALE_DISP, PHI0]), n_nodes)
    row_scale = np.tile(np.array([1.0 / R_M, 1.0 / R_M, 1.0 / R_E]), n_nodes)
    scaled = diags(row_scale) @ matrix @ diags(col_scale)
    return scaled.tocsr(), row_scale, col_scale


def constrained_solve(matrix: coo_matrix, mesh: dict) -> dict:
    """Solve the scaled saddle system with strong Dirichlet conditions."""

    n_nodes = len(mesh["nodes"])
    scaled, row_scale, col_scale = scale_matrix(matrix, n_nodes)
    bc_dofs, bc_values, bc = boundary_conditions(mesh)
    rhs = np.zeros(scaled.shape[0], dtype=np.float64)

    # Symmetric elimination is done on a copy of the scaled matrix.  Since the
    # source is zero, each prescribed column contributes directly to rhs.
    modified_rhs = rhs.copy()
    for dof, value in zip(bc_dofs, bc_values):
        modified_rhs -= scaled.getcol(int(dof)).toarray().ravel() * value

    constrained = scaled.tolil(copy=True)
    for dof, value in zip(bc_dofs, bc_values):
        dof = int(dof)
        constrained[:, dof] = 0.0
        constrained[dof, :] = 0.0
        constrained[dof, dof] = 1.0
        modified_rhs[dof] = value

    start = time.perf_counter()
    solution_nd = spsolve(constrained.tocsc(), modified_rhs)
    solve_seconds = time.perf_counter() - start
    if not np.all(np.isfinite(solution_nd)):
        raise FloatingPointError("Sparse solve returned non-finite values")

    solution = col_scale * solution_nd
    raw_residual = matrix @ solution
    scaled_residual = row_scale * raw_residual
    free = ~bc["dof_mask"]
    free_residual = scaled_residual[free]
    full_residual = scaled_residual

    # The residual rows of prescribed electric DOFs are the outward material
    # fluxes ∫_Γ D·n ds.  The free charge on the metal electrode has the
    # opposite sign, so keep both conventions explicit in the output.
    top_phi_dofs = 3 * bc["top_nodes"] + 2
    bottom_phi_dofs = 3 * bc["bottom_nodes"] + 2
    flux_top = float(np.sum(raw_residual[top_phi_dofs]))
    flux_bottom = float(np.sum(raw_residual[bottom_phi_dofs]))
    q_top = -flux_top
    q_bottom = -flux_bottom

    # Mechanical clamp reactions, reported per unit out-of-plane thickness.
    left_ux = 3 * bc["left_nodes"]
    left_uy = left_ux + 1
    clamp_fx = float(np.sum(raw_residual[left_ux]))
    clamp_fy = float(np.sum(raw_residual[left_uy]))

    xy = mesh["nodes"]
    vertex = mesh["vertices"]
    n_vertices = len(vertex)
    vertex_solution = solution[: 3 * n_vertices].reshape(n_vertices, 3)
    disp = vertex_solution[:, :2]
    max_disp = float(np.max(np.linalg.norm(disp, axis=1)))

    # Exact vertex at (L,H/2) exists for every aligned mesh.
    target = np.array([L, H / 2.0])
    tip_node = int(np.argmin(np.linalg.norm(xy - target, axis=1)))
    tip_error = float(np.linalg.norm(xy[tip_node] - target))
    if tip_error > 1e-10 * max(L, H):
        raise RuntimeError(f"Tip midpoint missing from mesh: error={tip_error}")

    tip = solution[3 * tip_node : 3 * tip_node + 3]
    deformation_factor = 1.0
    if max_disp > 0.0:
        deformation_factor = max(1.0, 0.30 * H / max_disp)

    return {
        "solution": solution,
        "solution_nd": solution_nd,
        "vertex_solution": vertex_solution,
        "raw_residual": raw_residual,
        "scaled_residual": scaled_residual,
        "free_residual_inf": float(np.max(np.abs(free_residual))),
        "free_residual_l2": float(np.linalg.norm(free_residual)),
        "full_residual_inf": float(np.max(np.abs(full_residual))),
        "material_flux_top": flux_top,
        "material_flux_bottom": flux_bottom,
        "q_top": q_top,
        "q_bottom": q_bottom,
        "q_balance": q_top + q_bottom,
        "clamp_fx": clamp_fx,
        "clamp_fy": clamp_fy,
        "tip_node": tip_node,
        "tip": tip,
        "max_disp": max_disp,
        "deformation_factor": deformation_factor,
        "solve_seconds": solve_seconds,
        "bc": bc,
        "bc_dofs": bc_dofs,
        "bc_values": bc_values,
    }


def run_level(nx: int, nlayer: int) -> dict:
    mesh = build_mesh(nx, nlayer)
    start = time.perf_counter()
    matrix = assemble(mesh)
    assemble_seconds = time.perf_counter() - start
    solved = constrained_solve(matrix, mesh)
    n_vertices = len(mesh["vertices"])
    n_nodes = len(mesh["nodes"])
    n_elements = len(mesh["elements"])
    result = {
        "mesh": mesh,
        "matrix": matrix,
        "solve": solved,
        "nx": nx,
        "ny_layer": nlayer,
        "n_vertices": n_vertices,
        "n_p2_nodes": n_nodes,
        "n_elements": n_elements,
        "h_x": mesh["hx"],
        "h_y": mesh["hy"],
        "assemble_seconds": assemble_seconds,
    }
    return result


def convergence_metrics(results: list[dict]) -> list[dict]:
    """Create JSON-friendly metrics and observed changes/rates."""

    metrics: list[dict] = []
    for result in results:
        s = result["solve"]
        tip = s["tip"]
        metrics.append(
            {
                "nx": result["nx"],
                "elements_per_layer_y": result["ny_layer"],
                "n_vertices": result["n_vertices"],
                "n_p2_nodes": result["n_p2_nodes"],
                "n_elements": result["n_elements"],
                "h_x_m": result["h_x"],
                "h_y_m": result["h_y"],
                "tip_dx_m": float(tip[0]),
                "tip_dy_m": float(tip[1]),
                "tip_phi_V": float(tip[2]),
                "electrode_free_charge_top_C_per_m": s["q_top"],
                "electrode_free_charge_bottom_C_per_m": s["q_bottom"],
                "electrode_free_charge_balance_C_per_m": s["q_balance"],
                "material_outward_flux_top_C_per_m": s["material_flux_top"],
                "material_outward_flux_bottom_C_per_m": s["material_flux_bottom"],
                "clamp_force_x_N_per_m": s["clamp_fx"],
                "clamp_force_y_N_per_m": s["clamp_fy"],
                "free_residual_inf_dimensionless": s["free_residual_inf"],
                "free_residual_l2_dimensionless": s["free_residual_l2"],
                "full_residual_inf_dimensionless": s["full_residual_inf"],
                "max_displacement_m": s["max_disp"],
                "deformation_amplification": s["deformation_factor"],
                "assembly_seconds": result["assemble_seconds"],
                "solve_seconds": s["solve_seconds"],
            }
        )

    if len(metrics) >= 2:
        for i in range(1, len(metrics)):
            prev = metrics[i - 1]
            cur = metrics[i]
            for name in ("tip_dy_m", "electrode_free_charge_top_C_per_m"):
                denominator = max(abs(cur[name]), 1e-30)
                cur[f"relative_change_{name}"] = abs(cur[name] - prev[name]) / denominator

    if len(metrics) >= 3:
        coarse = metrics[-3]
        medium = metrics[-2]
        fine = metrics[-1]
        for name in ("tip_dy_m", "electrode_free_charge_top_C_per_m"):
            d1 = abs(medium[name] - coarse[name])
            d2 = abs(fine[name] - medium[name])
            if d1 > 0.0 and d2 > 0.0:
                observed_p = math.log(d1 / d2, 2.0)
            else:
                observed_p = None
            if observed_p is not None and observed_p > 0.0:
                richardson = d2 / max(2.0**observed_p - 1.0, 1e-30)
            else:
                richardson = d2
            fine[f"observed_rate_{name}"] = observed_p
            fine[f"estimated_last_change_{name}"] = richardson
    return metrics


def plot_results(results: list[dict], metrics: list[dict]) -> tuple[Path, float]:
    """Save a clean four-panel figure for the finest P2 solution."""

    finest = results[-1]
    mesh = finest["mesh"]
    solved = finest["solve"]
    vertex = mesh["vertices"]
    triangles = mesh["triangles"]
    vsol = solved["vertex_solution"]
    tri = Triangulation(vertex[:, 0], vertex[:, 1], triangles)

    teal = "#137C80"
    orange = "#D66A35"
    ink = "#222C30"
    grey = "#738084"
    cmap = LinearSegmentedColormap.from_list("teal_white_orange", [teal, "#FAFAF6", orange])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": ink,
            "text.color": ink,
            "xtick.color": grey,
            "ytick.color": grey,
            "axes.edgecolor": "#C6CDCD",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.0), constrained_layout=True)
    ax = axes[0, 0]
    dy = vsol[:, 1]
    dy_scale = max(np.max(np.abs(dy)), 1e-30)
    im = ax.tripcolor(tri, dy * 1e6, cmap=cmap, shading="gouraud", vmin=-dy_scale * 1e6, vmax=dy_scale * 1e6)
    ax.set_title("a  Desplazamiento vertical calculado · P2 fino", loc="left")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, label="$d_y$ [$\\mu$m]")

    ax = axes[0, 1]
    phi = vsol[:, 2]
    im = ax.tripcolor(tri, phi, cmap=cmap, shading="gouraud", vmin=0.0, vmax=V_TOP)
    ax.set_title("b  Potencial eléctrico calculado · P2 fino", loc="left")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, label="$\u03c6$ [V]")

    ax = axes[1, 0]
    # Plot only the outer contour and the material interface.  Showing every
    # fine P2 edge would hide the curvature that this panel is meant to show.
    factor = solved["deformation_factor"]
    n_x = int(mesh["nx"])
    n_y = 2 * int(mesh["ny_layer"])

    def vertex_id(i: int, j: int) -> int:
        return j * (n_x + 1) + i

    perimeter = (
        [vertex_id(i, 0) for i in range(n_x + 1)]
        + [vertex_id(n_x, j) for j in range(1, n_y + 1)]
        + [vertex_id(i, n_y) for i in range(n_x - 1, -1, -1)]
        + [vertex_id(0, j) for j in range(n_y - 1, -1, -1)]
    )
    chains = [
        ("contorno exterior", perimeter),
        ("interfaz original", [vertex_id(i, int(mesh["ny_layer"])) for i in range(n_x + 1)]),
    ]
    for label, ids in chains:
        ids = np.asarray(ids, dtype=np.int64)
        ax.plot(vertex[ids, 0], vertex[ids, 1], color="#C8CECE", lw=1.0,
                alpha=0.9, label=label if label else None)
    deformed = vertex + factor * vsol[:, :2]
    for idx, (label, ids) in enumerate(chains):
        ids = np.asarray(ids, dtype=np.int64)
        is_interface = label == "interfaz original"
        ax.plot(
            deformed[ids, 0],
            deformed[ids, 1],
            color=orange if is_interface else teal,
            lw=1.5 if is_interface else 1.4,
            ls="--" if is_interface else "-",
            alpha=0.95,
            label="deformada · interfaz" if is_interface else "deformada amplificada",
        )
    ax.set_title(f"c  Deformada P2 × {factor:.3g} · factor calculado", loc="left")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1, 1]
    hx = np.array([m["h_x_m"] for m in metrics])
    tip_um = np.array([m["tip_dy_m"] * 1e6 for m in metrics])
    charge_uC = np.array([m["electrode_free_charge_top_C_per_m"] * 1e6 for m in metrics])
    ax.plot(hx, tip_um, "o-", color=teal, label="$d_y$ punta [$\\mu$m]")
    ax.set_xlabel("h_x [m]")
    ax.set_ylabel("$d_y$ punta [$\\mu$m]", color=teal)
    ax.tick_params(axis="y", labelcolor=teal)
    ax.invert_xaxis()
    ax2 = ax.twinx()
    ax2.plot(hx, charge_uC, "s--", color=orange, label="$Q_T$ electrodo [$\\mu$C/m]")
    ax2.set_ylabel("carga libre electrodo superior [$\\mu$C/m]", color=orange)
    ax2.tick_params(axis="y", labelcolor=orange)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax2.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_title("d  Convergencia de flecha y carga", loc="left")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8, loc="best")

    for a in axes.flat:
        a.grid(alpha=0.15, linewidth=0.6)

    paths = []
    for ext in ("pdf", "png", "svg"):
        path = ROOT / f"p5-solution.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths[0], factor


def write_integration_fragments(metrics: list[dict], factor: float) -> None:
    """Write short LaTeX fragments for the atlas integration.

    These are intentionally self-contained fragments: they do not depend on
    labels, equation numbers, or figures defined elsewhere in the thesis.
    """

    fine = metrics[-1]

    def tex_sci(value: float) -> str:
        mantissa, exponent = f"{value:.6e}".split("e")
        return f"{mantissa}\\times10^{{{int(exponent)}}}"

    method = (
        r"""La referencia numérica P5 se resolvió en el rectángulo efectivo
de dimensiones \(L=0.1\,\mathrm{m}\) y \(H=10^{-3}\,\mathrm{m}\), dividido en
dos capas con interfaz alineada en \(y=H/2\). Se usaron triángulos
conformes cuadráticos \(P_2\), una regla de cuadratura de Dunavant de siete
puntos y el sistema acoplado físico
\[
\begin{bmatrix}K_{dd}&K_{d\phi}\\K_{\phi d}&-K_{\phi\phi}\end{bmatrix}
\begin{bmatrix}d\\\phi\end{bmatrix}=0,
\]
con el orden global \((d_x,d_y,\phi)\) por nodo. El borde izquierdo se fijó
en \(d=(0,0)\); los demás bordes mecánicos son naturales, mientras que
\(\phi=0\) en la cara inferior, \(\phi=100\,\mathrm{V}\) en la superior y los
extremos laterales tienen \(D\!\cdot n=0\). La interfaz comparte los grados
de libertad conformes y por ello impone continuidad de \(d\) y \(\phi\), con
las condiciones de transmisión naturales para tracción y flujo normal.

La malla fina usada en la figura tiene 3072 elementos y 6369 nodos \(P_2\).
La flecha en \((L,H/2)\) es
"""
        + f"\\(d_y={tex_sci(fine['tip_dy_m'])}\\,\\mathrm{{m}}\\)"
        + r"""; el cambio relativo entre las dos mallas más finas es
"""
        + f"{fine.get('relative_change_tip_dy_m', float('nan')) * 100.0:.3f}\\%"
        + r""". La carga libre del electrodo superior, por unidad de espesor
fuera del plano, es
"""
        + f"\\(Q_T={tex_sci(fine['electrode_free_charge_top_C_per_m'])}\\,\\mathrm{{C/m}}\\)"
        + r""" y cambia
"""
        + f"{fine.get('relative_change_electrode_free_charge_top_C_per_m', float('nan')) * 100.0:.3f}\\%"
        + r""" en esa misma comparación. El residual infinito de los grados
de libertad libres es
"""
        + f"\\({tex_sci(fine['free_residual_inf_dimensionless'])}\\)"
        + r""" en las unidades escaladas de P5. Para no invertir la convención
de signos, \(Q_T=-\int_{\Gamma_T}D\!\cdot n\,ds\): el residual eléctrico
de potencial prescrito es el flujo saliente del material y tiene el signo
opuesto al de la carga libre sobre el electrodo. La deformada del panel
correspondiente se amplifica por el factor calculado
"""
        + f"{factor:.6g}"
        + r"""; ese factor es solo visual. Esta salida es una referencia
numérica didáctica de la formulación efectiva 2-D, no una validación
experimental ni una aprobación de resultados de investigación.
"""
    )
    caption = (
        r"""Solución FEM acoplada calculada para P5 con elementos triangulares
cuadráticos \(P_2\) y la malla fina alineada con la interfaz. (a) Campo
vertical \(d_y\) en micrómetros; (b) potencial \(\phi\) en voltios; (c)
contorno exterior e interfaz originales y deformados, cuya escala visual es el factor calculado
"""
        + f"{factor:.6g}"
        + r"""; (d) cambio de la flecha de punta y de la carga libre del
electrodo superior al refinar. La deformada se dibuja con coordenadas
geométricas \(x,y\) en metros y con razón de aspecto ajustada para facilitar
la lectura. Como \(L/H=100\), los ejes no usan la misma escala gráfica; además,
el desplazamiento se multiplica por el factor indicado solo para ser visible.
La carga mostrada usa
\(Q_T=-\int_{\Gamma_T}D\!\cdot n\,ds\), mientras el flujo saliente del
material tiene el signo contrario.
"""
    )
    (ROOT / "method.tex").write_text(method, encoding="utf-8")
    (ROOT / "caption.tex").write_text(caption, encoding="utf-8")


def save_outputs(
    results: list[dict],
    metrics: list[dict],
    figure_path: Path,
    factor: float,
    elapsed: float,
    patches: dict,
) -> None:
    finest = results[-1]
    mesh = finest["mesh"]
    solved = finest["solve"]
    vsol = solved["vertex_solution"]

    np.savez_compressed(
        ROOT / "pvdf_fem_results.npz",
        vertices=mesh["vertices"],
        nodes=mesh["nodes"],
        triangles=mesh["triangles"],
        elements=mesh["elements"],
        layers=mesh["layers"],
        solution=solved["solution"],
        vertex_solution=vsol,
        metrics_json=np.array(json.dumps(metrics, ensure_ascii=False)),
        scales=np.array([D0_SCALE_DISP, PHI0, R_M, R_E], dtype=np.float64),
        deformation_amplification=np.array([factor], dtype=np.float64),
    )
    write_integration_fragments(metrics, factor)

    convergence = {
        "tip_dy_m": metrics[-1]["tip_dy_m"],
        "electrode_free_charge_top_C_per_m": metrics[-1]["electrode_free_charge_top_C_per_m"],
        "estimated_last_change_tip_dy_m": metrics[-1].get("estimated_last_change_tip_dy_m"),
        "estimated_last_change_electrode_free_charge_top_C_per_m": metrics[-1].get(
            "estimated_last_change_electrode_free_charge_top_C_per_m"
        ),
        "observed_rate_tip_dy_m": metrics[-1].get("observed_rate_tip_dy_m"),
        "observed_rate_electrode_free_charge_top_C_per_m": metrics[-1].get(
            "observed_rate_electrode_free_charge_top_C_per_m"
        ),
    }

    evidence = {
        "status": "didactic_verified_fem_reference_not_research_claim",
        "runtime": {
            "python": sys.executable,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "matplotlib": matplotlib.__version__,
            "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        },
        "source_contract": {
            "methodology": "capitulos/metodologia/01-problemas.tex:P5",
            "constitutive": "capitulos/marco/04-piezo.tex:eq:constitutivo-2d",
            "geometry": {"L_m": L, "H_m": H, "interface_y_m": H / 2.0},
            "boundary_conditions": {
                "mechanical_left": "d=(0,0)",
                "mechanical_other": "sigma n=0",
                "potential_top_V": V_TOP,
                "potential_bottom_V": 0.0,
                "electrical_left_right": "D n=0",
                "interface": "continuous d, phi, traction, normal D",
            },
            "coefficients": {
                "C11_Pa": C11,
                "C22_Pa": C22,
                "C12_Pa": C12,
                "C66_Pa": C66,
                "e21_lower_C_per_m2": E21,
                "e22_lower_C_per_m2": E22,
                "e21_upper_C_per_m2": -E21,
                "e22_upper_C_per_m2": -E22,
                "kappa_x_F_per_m": KX,
                "kappa_y_F_per_m": KY,
            },
        },
        "method": {
            "finite_element": "conforming quadratic Lagrange P2 triangles",
            "mesh": "structured rectangular cells split into two CCW triangles; interface aligned",
            "quadrature": "seven-point Dunavant on each triangle",
            "assembly": "physical coupled weak form with [Kdd Kdphi; Kphid -Kphiphi]",
            "linear_solver": "scipy.sparse.linalg.spsolve after fixed P5 row/column scaling",
            "unknown_order": "(d_x,d_y,phi) per global P2 node",
            "charge_unit": "C/m out-of-plane thickness (effective 2-D model)",
            "charge_convention": "reported electrode free charge is Q_electrode=-∫Γ D·n ds; the raw prescribed-potential residual is the outward material flux ∫Γ D·n ds",
        },
        "fixed_scales": {
            "E0_V_per_m": E0,
            "strain_scale": S0,
            "displacement_scale_m": D0_SCALE_DISP,
            "sigma_scale_Pa": SIGMA0,
            "D_scale_C_per_m2": D0,
            "mechanical_residual_scale": R_M,
            "electrical_residual_scale": R_E,
        },
        "meshes": metrics,
        "convergence": convergence,
        "patch_checks": patches,
        "tolerances": {
            "assembly_quadrature": "exact for polynomial degree of P2 gradient products; 7-point rule used",
            "linear_residual_target_dimensionless": 1e-9,
            "tip_coordinate_tolerance_m": 1e-13,
            "refinement_interpretation": "last-change and observed-rate estimates; no asymptotic theorem claimed",
        },
        "figures": {
            "pdf": figure_path.name,
            "png": "p5-solution.png",
            "svg": "p5-solution.svg",
            "fields": ["d_y", "phi"],
            "deformed_shape": "vertex mesh amplified by factor stored in deformation_amplification",
            "deformation_amplification": factor,
        },
        "limitations": [
            "Effective 2-D constitutive model as written; no plane-stress or plane-strain closure is inferred.",
            "No dynamics, body force, free volume charge, nonlinear material law, 3-D effects, or experimental validation.",
            "The electrode charge is per unit out-of-plane thickness.",
            "Self-convergence is a numerical diagnostic, not an independently certified FEM reference.",
            "No result is used to approve a research claim or to compare a PIKAN method.",
        ],
        "artifacts": [
            "solve_pvdf_fem.py",
            "pvdf_fem_results.npz",
            "p5-solution.pdf",
            "p5-solution.png",
            "p5-solution.svg",
            "method.tex",
            "caption.tex",
            "evidence.json",
        ],
        "elapsed_seconds": elapsed,
    }
    (ROOT / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    patches = patch_checks()
    print(
        "patch checks: "
        f"grad_partition={patches['gradient_partition_inf']:.3e} "
        f"rigid_translation={patches['rigid_translation_strain_inf']:.3e} "
        f"rigid_rotation={patches['rigid_rotation_strain_inf']:.3e} "
        f"linear_strain={patches['linear_patch_strain_error_inf']:.3e}",
        flush=True,
    )
    # Geometric doubling keeps the physical aspect ratio and gives three
    # independent meshes while staying tractable in the bundled runtime.
    mesh_specs = [(24, 2), (48, 4), (96, 8)]
    results: list[dict] = []
    for nx, nlayer in mesh_specs:
        result = run_level(nx, nlayer)
        results.append(result)
        solve = result["solve"]
        print(
            f"mesh nx={nx} nlayer={nlayer}: elements={result['n_elements']} "
            f"p2_nodes={result['n_p2_nodes']} tip_dy={solve['tip'][1]:.8e} "
            f"Q_top={solve['q_top']:.8e} residual_free_inf={solve['free_residual_inf']:.3e}",
            flush=True,
        )

    metrics = convergence_metrics(results)
    figure_path, factor = plot_results(results, metrics)
    elapsed = time.perf_counter() - start
    save_outputs(results, metrics, figure_path, factor, elapsed, patches)
    print(f"Wrote P5 FEM artifacts to {ROOT}", flush=True)
    print(f"figure={figure_path.name} deformation_factor={factor:.8e}", flush=True)


if __name__ == "__main__":
    main()
