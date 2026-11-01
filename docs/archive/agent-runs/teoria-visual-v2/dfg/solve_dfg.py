"""Reproducible Taylor--Hood P2/P1 solve for the DFG 2D-1 Re=20 benchmark.

The script deliberately keeps the benchmark contract explicit.  The circular
obstacle is represented by a polygon with an explicit number of boundary
segments; each refinement increases the background resolution and the polygon
resolution.  The nonlinear solve uses Picard iterations followed by Newton
corrections.  No streamlines or field values in the figures are prescribed:
the solution plots are evaluated from the converged finite-element fields.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.spatial import Delaunay
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
import skfem

from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    FacetBasis,
    LinearForm,
    MeshTri,
    asm,
    condense,
)
from skfem.helpers import ddot, div, dot, grad

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.path import Path as PolygonPath
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
LENGTH = 2.2
HEIGHT = 0.41
CENTER = np.array([0.2, 0.2], dtype=float)
RADIUS = 0.05
NU = 0.001
U_MAX = 0.3
U_MEAN = 0.2
DIAMETER = 0.1

REFERENCE = {
    "Cd": 5.57953523384,
    "Cl": 0.010618948146,
    "p_diff": 0.11752016697,
}
SOURCE_URL = (
    "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/"
    "cfdbenchmarking/flow/dfg_benchmark1_re20.html"
)

SPEED_CMAP = LinearSegmentedColormap.from_list(
    "p4_speed", ["#FAFAF6", "#137C80"]
)
PRESSURE_CMAP = LinearSegmentedColormap.from_list(
    "p4_pressure", ["#137C80", "#FAFAF6", "#D66A35"]
)


def inlet_u(y: np.ndarray | float) -> np.ndarray | float:
    """DFG inflow: maximum 0.3 and transverse mean 0.2."""

    return 4.0 * U_MAX * np.asarray(y) * (HEIGHT - np.asarray(y)) / HEIGHT**2


def polygon_vertices(ncircle: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, ncircle, endpoint=False)
    return CENTER[:, None] + RADIUS * np.vstack((np.cos(theta), np.sin(theta)))


def _segment_samples(a: np.ndarray, b: np.ndarray, n: int = 25) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - t) * a[None, :] + t * b[None, :]


def build_mesh(nx: int, ny: int, ncircle: int) -> tuple[MeshTri, np.ndarray]:
    """Build a polygonal-hole Delaunay mesh with explicit boundary points."""

    poly = polygon_vertices(ncircle)
    # Path(..., closed=True) treats its final supplied vertex as the
    # CLOSEPOLY marker.  Supply the first vertex explicitly so the final
    # polygon edge is retained; otherwise the polygonal cylinder is opened
    # by one chord and the wake/lift becomes asymmetric.
    poly_path = PolygonPath(np.vstack((poly.T, poly.T[0])), closed=True)
    points: list[tuple[float, float]] = []
    keys: set[tuple[float, float]] = set()

    def add(x: float, y: float) -> None:
        if x < -1e-12 or x > LENGTH + 1e-12:
            return
        if y < -1e-12 or y > HEIGHT + 1e-12:
            return
        key = (round(float(x), 11), round(float(y), 11))
        if key not in keys:
            keys.add(key)
            points.append(key)

    # Rectangular background grid.  The polygon boundary is inserted
    # separately, so the hole has a known, measurable polygonal approximation.
    for x in np.linspace(0.0, LENGTH, nx + 1):
        for y in np.linspace(0.0, HEIGHT, ny + 1):
            if not poly_path.contains_point((x, y), radius=-1e-12):
                add(float(x), float(y))

    # Nested rings resolve the cylinder and make the first layer less skewed.
    for radial_offset, ring_count in (
        (0.012, ncircle),
        (0.028, ncircle),
        (0.055, max(16, ncircle // 2)),
    ):
        theta = np.linspace(0.0, 2.0 * np.pi, ring_count, endpoint=False)
        for angle in theta:
            x, y = CENTER + (RADIUS + radial_offset) * np.array(
                [math.cos(angle), math.sin(angle)]
            )
            add(float(x), float(y))

    for x, y in poly.T:
        add(float(x), float(y))

    p = np.asarray(points, dtype=float)
    tri = Delaunay(p).simplices

    def triangle_is_fluid(vertices: np.ndarray) -> bool:
        q = p[vertices]
        if np.any(poly_path.contains_points(q, radius=-1e-10)):
            return False
        if poly_path.contains_point(np.mean(q, axis=0), radius=-1e-10):
            return False
        # A Delaunay edge can cross a convex hole without placing a vertex
        # inside it.  Sampling each edge catches those slivers while preserving
        # the polygon's own boundary chords.
        for i, j in ((0, 1), (1, 2), (2, 0)):
            samples = _segment_samples(q[i], q[j], 25)
            if np.any(poly_path.contains_points(samples, radius=-1e-10)):
                return False
        return True

    keep = np.array([triangle_is_fluid(t) for t in tri], dtype=bool)
    tri = tri[keep]

    # Ensure a consistent positive orientation for the affine mapping.
    area2 = (
        (p[tri[:, 1], 0] - p[tri[:, 0], 0])
        * (p[tri[:, 2], 1] - p[tri[:, 0], 1])
        - (p[tri[:, 1], 1] - p[tri[:, 0], 1])
        * (p[tri[:, 2], 0] - p[tri[:, 0], 0])
    )
    flip = area2 < 0.0
    tri[flip, 1], tri[flip, 2] = tri[flip, 2], tri[flip, 1].copy()
    tri = tri.T.astype(np.int32)

    mesh = MeshTri(p.T, tri).remove_unused_nodes()
    if mesh.nelements == 0:
        raise RuntimeError("La triangulación produjo cero elementos.")
    areas = np.abs(
        (
            (mesh.p[0, mesh.t[1]] - mesh.p[0, mesh.t[0]])
            * (mesh.p[1, mesh.t[2]] - mesh.p[1, mesh.t[0]])
            - (mesh.p[1, mesh.t[1]] - mesh.p[1, mesh.t[0]])
            * (mesh.p[0, mesh.t[2]] - mesh.p[0, mesh.t[0]])
        )
        / 2.0
    )
    if float(np.min(areas)) <= 1e-13:
        raise RuntimeError("La malla contiene un triángulo degenerado.")
    return mesh, poly


def classify_facets(mesh: MeshTri) -> dict[str, np.ndarray]:
    """Classify the physical boundary facets from their endpoint coordinates."""

    facets = mesh.boundary_facets()
    verts = mesh.facets[:, facets]
    xy = mesh.p[:, verts]
    tol = 3e-9
    x = xy[0]
    y = xy[1]
    inlet = facets[np.max(np.abs(x), axis=0) < tol]
    outlet = facets[np.max(np.abs(x - LENGTH), axis=0) < tol]
    bottom = facets[np.max(np.abs(y), axis=0) < tol]
    top = facets[np.max(np.abs(y - HEIGHT), axis=0) < tol]
    radial = np.linalg.norm(
        xy.transpose(1, 2, 0) - CENTER[None, None, :], axis=2
    )
    cylinder = facets[np.max(np.abs(radial - RADIUS), axis=0) < 1e-8]
    groups = {
        "inlet": np.unique(inlet),
        "outlet": np.unique(outlet),
        "bottom": np.unique(bottom),
        "top": np.unique(top),
        "cylinder": np.unique(cylinder),
    }
    covered = np.unique(np.concatenate(list(groups.values())))
    if len(covered) != len(facets):
        # Keep an explicit diagnostic rather than silently treating unknown
        # facets as natural outflow.
        unknown = np.setdiff1d(facets, covered)
        raise RuntimeError(
            f"Facetas de frontera sin clasificar: {len(unknown)}."
        )
    if any(len(groups[k]) == 0 for k in ("inlet", "outlet", "cylinder")):
        raise RuntimeError(f"Frontera incompleta: { {k: len(v) for k, v in groups.items()} }")
    return groups


@BilinearForm
def vector_laplace(u, v, w):
    return ddot(grad(u), grad(v))


@BilinearForm
def picard_convection(u, v, w):
    directional = np.einsum("j...,ij...->i...", w.beta, grad(u))
    return dot(directional, v)


@BilinearForm
def newton_convection(u, v, w):
    # D[(beta . grad) beta] [delta] =
    # (beta . grad) delta + (delta . grad) beta.
    first = np.einsum("j...,ij...->i...", w.beta, grad(u))
    second = np.einsum("j...,ij...->i...", u, w.beta.grad)
    return dot(first + second, v)


@BilinearForm
def pressure_gradient(p, v, w):
    return -p * div(v)


@BilinearForm
def incompressibility(u, q, w):
    return -div(u) * q


@LinearForm
def nonlinear_load(v, w):
    convective = np.einsum("j...,ij...->i...", w.beta, w.beta.grad)
    return dot(convective, v)


def make_bases(mesh: MeshTri) -> tuple[Basis, Basis, ElementVector]:
    elem_u = ElementVector(ElementTriP2(), dim=2)
    elem_p = ElementTriP1()
    # Equal quadrature order is required for mixed forms.
    basis_u = Basis(mesh, elem_u, intorder=6)
    basis_p = Basis(mesh, elem_p, intorder=6)
    return basis_u, basis_p, elem_u


def dirichlet_values(
    basis_u: Basis, groups: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    d_in = basis_u.get_dofs(facets=groups["inlet"])
    d_all = basis_u.get_dofs(
        facets=np.unique(
            np.concatenate(
                [groups["inlet"], groups["bottom"], groups["top"], groups["cylinder"]]
            )
        )
    )
    u_d = np.zeros(basis_u.N, dtype=float)
    d_x = d_in.all("u^1")
    d_y = d_in.all("u^2")
    u_d[d_x] = inlet_u(basis_u.doflocs[1, d_x])
    u_d[d_y] = 0.0
    return np.asarray(d_all.all(), dtype=np.int32), u_d


def assemble_blocks(
    basis_u: Basis,
    basis_p: Basis,
    beta: Any,
    newton: bool = False,
) -> tuple[Any, Any, Any, Any]:
    viscous = NU * asm(vector_laplace, basis_u)
    if newton:
        convection = asm(newton_convection, basis_u, basis_u, beta=beta)
    else:
        convection = asm(picard_convection, basis_u, basis_u, beta=beta)
    up = asm(pressure_gradient, basis_p, basis_u)
    pu = asm(incompressibility, basis_u, basis_p)
    A = bmat([[viscous + convection, up], [pu, None]], format="csr")
    return A, viscous, up, pu


def residual_vector(
    basis_u: Basis,
    basis_p: Basis,
    U: np.ndarray,
    P: np.ndarray,
    viscous: Any,
    up: Any,
    pu: Any,
) -> np.ndarray:
    beta = basis_u.interpolate(U)
    nonlinear = asm(nonlinear_load, basis_u, beta=beta)
    ru = viscous @ U + nonlinear + up @ P
    rp = pu @ U
    return np.concatenate((np.asarray(ru).ravel(), np.asarray(rp).ravel()))


def solve_linear(
    A: Any,
    rhs: np.ndarray,
    D_u: np.ndarray,
    x_boundary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    D = np.asarray(D_u, dtype=np.int32)
    x = np.asarray(x_boundary, dtype=float).copy()
    if x.shape[0] != n:
        raise ValueError("La condición de frontera tiene dimensión incorrecta.")
    condensed, bcond, x0, I = condense(A, rhs, D=D, x=x)
    with np.errstate(all="ignore"):
        solution = spsolve(condensed, bcond)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("El sistema lineal produjo NaN/Inf.")
    full = x0.copy()
    full[I] = solution
    return full, np.asarray(I, dtype=np.int32)


def boundary_vector(basis_u: Basis, u_d: np.ndarray, basis_p: Basis) -> np.ndarray:
    return np.concatenate((u_d, np.zeros(basis_p.N, dtype=float)))


def solve_level(
    mesh: MeshTri,
    groups: dict[str, np.ndarray],
    picard_max: int,
    newton_max: int,
    nonlinear_tol: float,
) -> dict[str, Any]:
    basis_u, basis_p, elem_u = make_bases(mesh)
    d_u, u_d = dirichlet_values(basis_u, groups)
    n_u, n_p = basis_u.N, basis_p.N
    boundary = boundary_vector(basis_u, u_d, basis_p)
    U = u_d.copy()
    P = np.zeros(n_p, dtype=float)
    picard_records: list[dict[str, float]] = []
    t0 = time.perf_counter()

    # Picard fixed-point continuation.
    picard_converged = False
    free = None
    for it in range(1, picard_max + 1):
        beta = basis_u.interpolate(U)
        A, viscous, up, pu = assemble_blocks(
            basis_u, basis_p, beta, newton=False
        )
        z, free = solve_linear(A, np.zeros(n_u + n_p), d_u, boundary)
        U_new, P_new = z[:n_u], z[n_u:]
        change = np.linalg.norm(U_new - U) / max(1.0, np.linalg.norm(U_new))
        R = residual_vector(basis_u, basis_p, U_new, P_new, viscous, up, pu)
        rfree = float(np.linalg.norm(R[free]) / max(1.0, np.linalg.norm(z)))
        picard_records.append({"iteration": it, "change": float(change), "residual": rfree})
        U, P = U_new, P_new
        if change < nonlinear_tol and rfree < 10.0 * nonlinear_tol:
            picard_converged = True
            break

    # Newton correction.  Boundary increments are zero, so the physical
    # Dirichlet values remain exactly those of the benchmark.
    newton_records: list[dict[str, float]] = []
    newton_converged = False
    for it in range(1, newton_max + 1):
        beta = basis_u.interpolate(U)
        A, viscous, up, pu = assemble_blocks(
            basis_u, basis_p, beta, newton=True
        )
        R = residual_vector(basis_u, basis_p, U, P, viscous, up, pu)
        rfree_before = float(np.linalg.norm(R[free]) / max(1.0, np.linalg.norm(U)))
        correction, free_newton = solve_linear(
            A,
            -R,
            d_u,
            np.zeros(n_u + n_p, dtype=float),
        )
        dU, dP = correction[:n_u], correction[n_u:]
        U += dU
        P += dP
        U[d_u] = u_d[d_u]
        change = np.linalg.norm(dU) / max(1.0, np.linalg.norm(U))
        R_after = residual_vector(basis_u, basis_p, U, P, viscous, up, pu)
        rfree_after = float(
            np.linalg.norm(R_after[free_newton]) / max(1.0, np.linalg.norm(U))
        )
        newton_records.append(
            {
                "iteration": it,
                "change": float(change),
                "residual_before": rfree_before,
                "residual": rfree_after,
            }
        )
        free = free_newton
        if change < nonlinear_tol and rfree_after < 10.0 * nonlinear_tol:
            newton_converged = True
            break

    # Refresh final blocks and residual with the final state.
    beta = basis_u.interpolate(U)
    A, viscous, up, pu = assemble_blocks(
        basis_u, basis_p, beta, newton=True
    )
    R = residual_vector(basis_u, basis_p, U, P, viscous, up, pu)
    if free is None:
        free = np.arange(n_u + n_p)
    return {
        "mesh": mesh,
        "groups": groups,
        "basis_u": basis_u,
        "basis_p": basis_p,
        "elem_u": elem_u,
        "U": U,
        "P": P,
        "u_d": u_d,
        "d_u": d_u,
        "free": free,
        "viscous": viscous,
        "up": up,
        "pu": pu,
        "residual": R,
        "picard_records": picard_records,
        "newton_records": newton_records,
        "picard_converged": picard_converged,
        "newton_converged": newton_converged,
        "elapsed_s": time.perf_counter() - t0,
    }


def integrate_flux(
    mesh: MeshTri, elem_u: ElementVector, facets: np.ndarray, U: np.ndarray
) -> float:
    fb = FacetBasis(mesh, elem_u, facets=facets, intorder=6)
    field = fb.interpolate(U)
    return float(np.sum(np.einsum("i...,i...->...", field, fb.normals) * fb.dx))


def boundary_diagnostics(
    sol: dict[str, Any], poly: np.ndarray
) -> dict[str, float]:
    mesh = sol["mesh"]
    groups = sol["groups"]
    basis_u = sol["basis_u"]
    basis_p = sol["basis_p"]
    elem_u = sol["elem_u"]
    U = sol["U"]
    P = sol["P"]
    field = basis_u.interpolate(U)
    div_field = div(field)
    volume = float(np.sum(basis_u.dx))
    divergence_l2 = math.sqrt(float(np.sum(div_field**2 * basis_u.dx) / volume))

    R = sol["residual"]
    free = sol["free"]
    weak_residual = float(np.linalg.norm(R[free]) / max(1.0, np.linalg.norm(U)))
    continuity_l2 = float(
        np.linalg.norm(sol["pu"] @ U) / max(1.0, np.linalg.norm(U))
    )

    inlet_flux = integrate_flux(mesh, elem_u, groups["inlet"], U)
    outlet_flux = integrate_flux(mesh, elem_u, groups["outlet"], U)
    cylinder_flux = integrate_flux(mesh, elem_u, groups["cylinder"], U)

    # Dirichlet traces and outlet traction use the fluid-domain normal.
    fb_in = FacetBasis(mesh, elem_u, facets=groups["inlet"], intorder=6)
    in_field = fb_in.interpolate(U)
    in_xy = fb_in.global_coordinates()
    inlet_error = np.sqrt(
        np.sum(
            ((in_field[0] - inlet_u(in_xy[1])) ** 2 + in_field[1] ** 2)
            * fb_in.dx
        )
        / np.sum(fb_in.dx)
    )
    fb_out_u = FacetBasis(mesh, elem_u, facets=groups["outlet"], intorder=6)
    fb_out_p = FacetBasis(mesh, ElementTriP1(), facets=groups["outlet"], intorder=6)
    out_u = fb_out_u.interpolate(U)
    out_p = fb_out_p.interpolate(P)
    out_trac = (
        NU
        * np.einsum("ij...,j...->i...", out_u.grad, fb_out_u.normals)
        - out_p * fb_out_u.normals
    )
    outlet_natural_l2 = math.sqrt(
        float(np.sum(np.sum(out_trac**2, axis=0) * fb_out_u.dx) / np.sum(fb_out_u.dx))
    )

    # The fluid boundary normal points into the obstacle.  The benchmark
    # defines the stress integral with the outward normal of the cylinder.
    fb_c_u = FacetBasis(mesh, elem_u, facets=groups["cylinder"], intorder=6)
    fb_c_p = FacetBasis(mesh, ElementTriP1(), facets=groups["cylinder"], intorder=6)
    c_u = fb_c_u.interpolate(U)
    c_p = fb_c_p.interpolate(P)
    n_cyl = -np.asarray(fb_c_u.normals)
    stress = NU * c_u.grad
    stress = np.array(stress, copy=True)
    stress[0, 0] -= c_p
    stress[1, 1] -= c_p
    traction = np.einsum("ij...,j...->i...", stress, n_cyl)
    force = np.sum(traction * fb_c_u.dx, axis=(1, 2))
    fd, fl = float(force[0]), float(force[1])
    cd = 2.0 * fd / (U_MEAN**2 * DIAMETER)
    cl = 2.0 * fl / (U_MEAN**2 * DIAMETER)

    p_interp = basis_p.interpolator(P)
    eps = 1e-7
    probe = np.array(
        [
            [CENTER[0] - RADIUS - eps, CENTER[0] + RADIUS + eps],
            [CENTER[1], CENTER[1]],
        ]
    )
    pvals = np.asarray(p_interp(probe)).ravel()
    p_diff = float(pvals[0] - pvals[1])

    # Boundary no-slip is checked through facet quadrature rather than only
    # vertex values; the quadratic edge modes are therefore included.
    no_slip = []
    for name in ("bottom", "top", "cylinder"):
        fb = FacetBasis(mesh, elem_u, facets=groups[name], intorder=6)
        ff = fb.interpolate(U)
        no_slip.append(float(np.max(np.sqrt(ff[0] ** 2 + ff[1] ** 2))))

    return {
        "divergence_l2": divergence_l2,
        "continuity_weak": continuity_l2,
        "weak_residual_free": weak_residual,
        "inlet_trace_l2": float(inlet_error),
        "outlet_natural_l2": outlet_natural_l2,
        "inlet_flux": inlet_flux,
        "outlet_flux": outlet_flux,
        "cylinder_flux": cylinder_flux,
        "flux_closure_abs": abs(inlet_flux + outlet_flux + cylinder_flux),
        "flux_closure_rel": abs(inlet_flux + outlet_flux + cylinder_flux)
        / max(abs(inlet_flux), 1e-12),
        "Fd": fd,
        "Fl": fl,
        "Cd": cd,
        "Cl": cl,
        "p_diff": p_diff,
        "noslip_max": max(no_slip),
        "mesh_area_min": float(
            np.min(
                np.abs(
                    (
                        (mesh.p[0, mesh.t[1]] - mesh.p[0, mesh.t[0]])
                        * (mesh.p[1, mesh.t[2]] - mesh.p[1, mesh.t[0]])
                        - (mesh.p[1, mesh.t[1]] - mesh.p[1, mesh.t[0]])
                        * (mesh.p[0, mesh.t[2]] - mesh.p[0, mesh.t[0]])
                    )
                    / 2.0
                )
            )
        ),
    }


def interpolate_grid(
    basis: Basis,
    coeff: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    poly: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(xgrid, ygrid)
    polygon = PolygonPath(np.vstack((poly.T, poly.T[0])), closed=True)
    solid = polygon.contains_points(
        np.column_stack((X.ravel(), Y.ravel())), radius=1e-11
    ).reshape(X.shape)
    fluid = ~solid
    # ``Basis.interpolator`` returns a one-dimensional array for a scalar
    # element and a (ncomponent, npoint) array for ElementVector.  Reserving
    # two components unconditionally corrupts scalar pressure fields in the
    # fallback path (the row/column indices are then mistaken for a
    # component), which can produce artificial horizontal bands.
    ncomponent = int(getattr(basis.elem, "nodal_dofs", 1))
    values = np.full((ncomponent,) + X.shape, np.nan)
    interp = basis.interpolator(coeff)
    flat = np.flatnonzero(fluid)
    values_flat = values.reshape(ncomponent, -1)
    # scikit-fem's element finder forms an intermediate array whose size is
    # proportional to (number of elements) x (number of probe points).  A
    # dense fine-grid call can otherwise exceed RAM, so interpolate in
    # bounded chunks while retaining the exact resolved coefficients.
    chunk_size = 4000
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        indices = flat[start:stop]
        pts = np.vstack((X.ravel()[indices], Y.ravel()[indices]))
        try:
            raw = np.asarray(interp(pts))
            if ncomponent == 1:
                if raw.ndim == 2:
                    raw = raw[0]
                values_flat[0, indices] = raw.ravel()
            elif raw.ndim == 2:
                values_flat[:, indices] = raw
            else:
                values_flat[:, indices] = raw.reshape(ncomponent, -1)
        except (ValueError, IndexError, MemoryError):
            # A point on a polygonal edge may be assigned ambiguously by the
            # element finder.  Recover only that chunk point by point.
            for flat_index in indices:
                try:
                    point = np.vstack((X.ravel()[flat_index], Y.ravel()[flat_index]))
                    raw = np.asarray(interp(point))
                    if ncomponent == 1:
                        values_flat[0, flat_index] = raw.ravel()[0]
                    elif raw.ndim == 2:
                        values_flat[:, flat_index] = raw[:, 0]
                    else:
                        values_flat[:, flat_index] = raw.ravel()[:ncomponent]
                except (ValueError, IndexError, MemoryError):
                    continue
    return values, solid


def field_grid(
    sol: dict[str, Any], poly: np.ndarray, nx: int = 360, ny: int = 120
) -> dict[str, np.ndarray]:
    xgrid = np.linspace(0.0, LENGTH, nx)
    ygrid = np.linspace(0.0, HEIGHT, ny)
    ugrid, solid = interpolate_grid(sol["basis_u"], sol["U"], xgrid, ygrid, poly)
    pgrid, _ = interpolate_grid(sol["basis_p"], sol["P"], xgrid, ygrid, poly)
    speed = np.sqrt(ugrid[0] ** 2 + ugrid[1] ** 2)
    return {
        "x": xgrid,
        "y": ygrid,
        "X": np.meshgrid(xgrid, ygrid)[0],
        "Y": np.meshgrid(xgrid, ygrid)[1],
        "u": ugrid[0],
        "v": ugrid[1],
        "speed": speed,
        "pressure": pgrid[0],
        "solid": solid,
    }


def scalar_affine_interpolation_test(mesh: MeshTri, poly: np.ndarray) -> dict[str, Any]:
    """Check scalar interpolation independently of the Navier--Stokes solve.

    For a P1 field with nodal values p=x+2y, interpolation must reproduce
    that affine function up to floating-point roundoff.  This catches the
    scalar/vector shape mix-up that would otherwise make a pressure plot
    depend on array indices rather than on the resolved pressure field.
    """
    basis = Basis(mesh, ElementTriP1(), intorder=4)
    coeff = basis.doflocs[0] + 2.0 * basis.doflocs[1]
    xgrid = np.linspace(0.0, LENGTH, 19)
    ygrid = np.linspace(0.0, HEIGHT, 9)
    values, solid = interpolate_grid(basis, coeff, xgrid, ygrid, poly)
    X, Y = np.meshgrid(xgrid, ygrid)
    valid = (~solid) & np.isfinite(values[0])
    error = float(np.max(np.abs(values[0][valid] - (X + 2.0 * Y)[valid])))
    return {
        "function": "p(x,y)=x+2y",
        "max_abs_error": error,
        "points_checked": int(np.count_nonzero(valid)),
        "pass": bool(error < 1e-10),
    }


def plot_geometry(ax: Any, poly: np.ndarray, show_labels: bool = True) -> None:
    ax.set_xlim(0.0, LENGTH)
    ax.set_ylim(0.0, HEIGHT)
    ax.set_aspect("equal")
    ax.add_patch(
        PolygonPatch(
            poly.T,
            closed=True,
            facecolor="white",
            edgecolor="#D66A35",
            linewidth=1.8,
            zorder=5,
        )
    )
    ax.plot([0.0, 0.0], [0.0, HEIGHT], color="#137C80", linewidth=2.5)
    ax.plot([LENGTH, LENGTH], [0.0, HEIGHT], color="#137C80", linewidth=2.0)
    ax.plot([0.0, LENGTH], [0.0, 0.0], color="#137C80", linewidth=1.7)
    ax.plot([0.0, LENGTH], [HEIGHT, HEIGHT], color="#137C80", linewidth=1.7)
    if show_labels:
        ax.text(0.01, HEIGHT * 0.88, "entrada", color="#137C80", fontsize=9)
        ax.text(LENGTH - 0.22, HEIGHT * 0.88, "salida", color="#137C80", fontsize=9)
        ax.text(CENTER[0] - 0.035, CENTER[1] + 0.065, "cilindro", color="#D66A35", fontsize=9)
        ax.annotate(
            "r = 0.05 m",
            xy=(CENTER[0] + RADIUS, CENTER[1]),
            xytext=(CENTER[0] + 0.10, CENTER[1] + 0.05),
            arrowprops={"arrowstyle": "->", "color": "#D66A35"},
            color="#D66A35",
            fontsize=8,
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def save_solution_figure(grid: dict[str, np.ndarray], poly: np.ndarray) -> None:
    xgrid, ygrid = grid["x"], grid["y"]
    X, Y = grid["X"], grid["Y"]
    solid = grid["solid"]
    speed = np.ma.masked_where(solid | ~np.isfinite(grid["speed"]), grid["speed"])
    pressure = np.ma.masked_where(solid | ~np.isfinite(grid["pressure"]), grid["pressure"])
    u = np.ma.masked_where(solid | ~np.isfinite(grid["u"]), grid["u"])
    v = np.ma.masked_where(solid | ~np.isfinite(grid["v"]), grid["v"])
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.2, 10.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
    )
    ax0, ax1, ax2 = axes
    # Teal-dominant speed map and the orange cylinder edge keep the visual
    # contract consistent with the companion problem figures.
    cf0 = ax0.contourf(X, Y, speed, levels=28, cmap=SPEED_CMAP)
    ax0.contour(X, Y, speed, levels=12, colors="white", linewidths=0.25, alpha=0.45)
    cb0 = fig.colorbar(
        cf0,
        ax=ax0,
        orientation="horizontal",
        fraction=0.075,
        pad=0.17,
        label="|u| [m/s]",
    )
    cb0.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax0.set_title("Navier--Stokes DFG 2D-1: velocidad resuelta")
    ax1_data = pressure.compressed()
    if ax1_data.size:
        pmin, pmax = float(np.nanpercentile(ax1_data, 2)), float(np.nanpercentile(ax1_data, 98))
        if pmin == pmax:
            pmin, pmax = pmin - 1.0, pmax + 1.0
        norm = TwoSlopeNorm(vmin=pmin, vcenter=0.5 * (pmin + pmax), vmax=pmax)
    else:
        norm = Normalize()
    cf1 = ax1.contourf(X, Y, pressure, levels=28, cmap=PRESSURE_CMAP, norm=norm)
    cb1 = fig.colorbar(
        cf1,
        ax=ax1,
        orientation="horizontal",
        fraction=0.075,
        pad=0.17,
        label="p [m$^2$/s$^2$]",
    )
    cb1.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.set_title("Navier--Stokes DFG 2D-1: presión resuelta")
    ax2.set_title("Navier--Stokes DFG 2D-1: líneas de corriente resueltas")
    try:
        ax2.streamplot(
            xgrid,
            ygrid,
            u,
            v,
            color="#137C80",
            density=1.35,
            linewidth=0.55,
            arrowsize=0.75,
        )
    except (ValueError, RuntimeError):
        # A grid point exactly on a polygon chord can make streamplot reject
        # the array; the speed and pressure panels remain valid.
        ax2.text(
            0.5,
            0.5,
            "streamplot no disponible\npara esta malla",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )
    for ax in axes:
        ax.add_patch(
            PolygonPatch(
                poly.T,
                closed=True,
                facecolor="white",
                edgecolor="#D66A35",
                linewidth=1.4,
                zorder=10,
            )
        )
        ax.set_xlim(0.0, LENGTH)
        ax.set_ylim(0.0, HEIGHT)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.suptitle(
        "DFG 2D-1, Re=20: campos de la solución Taylor--Hood P2/P1",
        fontsize=13,
    )
    for ext in ("pdf", "png", "svg"):
        fig.savefig(HERE / f"p4-solution.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)


def save_illustration(poly: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), constrained_layout=True)
    ax = axes[0]
    plot_geometry(ax, poly, show_labels=True)
    ax.set_title("Contrato geométrico DFG 2D-1")
    ax2 = axes[1]
    y = np.linspace(0.0, HEIGHT, 300)
    ax2.plot(inlet_u(y), y, color="#137C80", linewidth=2.4)
    ax2.axvline(0.0, color="#D66A35", linewidth=0.9)
    ax2.set_xlabel("u-in [m/s]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Entrada: máximo 0.3, media 0.2")
    ax2.grid(alpha=0.22)
    ax2.text(
        0.97,
        0.04,
        r"$-\nu\Delta\mathbf{u}+(\mathbf{u}\cdot\nabla)\mathbf{u}+\nabla p=0$"
        "\n"
        r"$\nabla\cdot\mathbf{u}=0$",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#D66A35",
        bbox={"facecolor": "white", "edgecolor": "#D66A35", "alpha": 0.92},
    )
    fig.suptitle("DFG 2D-1: ilustración del contrato físico", fontsize=13)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(HERE / f"p4-illustration.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)


def run_level(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    mesh, poly = build_mesh(config["nx"], config["ny"], config["ncircle"])
    groups = classify_facets(mesh)
    sol = solve_level(
        mesh,
        groups,
        picard_max=args.picard_max,
        newton_max=args.newton_max,
        nonlinear_tol=args.nonlinear_tol,
    )
    diagnostics = boundary_diagnostics(sol, poly)
    h = max(LENGTH / config["nx"], HEIGHT / config["ny"])
    return {
        "name": config["name"],
        "nx": config["nx"],
        "ny": config["ny"],
        "ncircle": config["ncircle"],
        "h_proxy": h,
        "vertices": int(mesh.nvertices),
        "elements": int(mesh.nelements),
        "boundary_facets": {k: int(len(v)) for k, v in groups.items()},
        "dofs_velocity": int(sol["basis_u"].N),
        "dofs_pressure": int(sol["basis_p"].N),
        "picard": {
            "converged": bool(sol["picard_converged"]),
            "iterations": len(sol["picard_records"]),
            "records": sol["picard_records"],
        },
        "newton": {
            "converged": bool(sol["newton_converged"]),
            "iterations": len(sol["newton_records"]),
            "records": sol["newton_records"],
        },
        "diagnostics": diagnostics,
        "elapsed_s": float(time.perf_counter() - started),
        "solution": sol,
        "poly": poly,
    }


def convergence_report(levels: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for a, b in zip(levels[:-1], levels[1:]):
        ga, gb = a["grid"], b["grid"]
        valid = (
            np.isfinite(ga["u"])
            & np.isfinite(gb["u"])
            & np.isfinite(ga["v"])
            & np.isfinite(gb["v"])
            & np.isfinite(ga["pressure"])
            & np.isfinite(gb["pressure"])
        )
        if not np.any(valid):
            comparisons.append({"from": a["name"], "to": b["name"], "valid": False})
            continue
        du = np.sqrt((ga["u"] - gb["u"]) ** 2 + (ga["v"] - gb["v"]) ** 2)
        dp = ga["pressure"] - gb["pressure"]
        u_norm = np.sqrt(ga["u"] ** 2 + ga["v"] ** 2)
        p_centered_a = ga["pressure"] - np.nanmean(ga["pressure"][valid])
        p_centered_b = gb["pressure"] - np.nanmean(gb["pressure"][valid])
        u_rel = float(
            np.sqrt(np.nanmean((du[valid]) ** 2))
            / max(np.sqrt(np.nanmean(u_norm[valid] ** 2)), 1e-12)
        )
        p_rel = float(
            np.sqrt(np.nanmean((p_centered_a[valid] - p_centered_b[valid]) ** 2))
            / max(np.sqrt(np.nanmean(p_centered_a[valid] ** 2)), 1e-12)
        )
        da = a["diagnostics"]
        db = b["diagnostics"]
        def relative_metric(key: str) -> float:
            return abs(db[key] - da[key]) / max(abs(db[key]), abs(da[key]), 1e-12)
        comparisons.append(
            {
                "from": a["name"],
                "to": b["name"],
                "valid": True,
                "velocity_grid_rel_l2": u_rel,
                "pressure_grid_rel_l2_zero_mean": p_rel,
                "Cd_rel_change": relative_metric("Cd"),
                "Cl_rel_change": relative_metric("Cl"),
                "p_diff_rel_change": relative_metric("p_diff"),
            }
        )
    last = comparisons[-1] if comparisons else {}
    converged = bool(
        last.get("valid", False)
        and last.get("velocity_grid_rel_l2", float("inf")) < 0.03
        and last.get("pressure_grid_rel_l2_zero_mean", float("inf")) < 0.05
        and last.get("Cd_rel_change", float("inf")) < 0.03
        and last.get("p_diff_rel_change", float("inf")) < 0.03
    )
    return {
        "pairwise": comparisons,
        "criterion": {
            "velocity_grid_rel_l2": "< 0.03",
            "pressure_grid_rel_l2_zero_mean": "< 0.05",
            "Cd_rel_change": "< 0.03",
            "p_diff_rel_change": "< 0.03",
        },
        "converged_under_declared_criterion": converged,
        "status": "converged" if converged else "not_certified",
    }


def save_solution_npz(final: dict[str, Any], grid: dict[str, np.ndarray]) -> None:
    sol = final["solution"]
    mesh = sol["mesh"]
    np.savez_compressed(
        HERE / "p4-solution.npz",
        mesh_points=mesh.p,
        mesh_triangles=mesh.t,
        velocity_dofs=sol["U"],
        pressure_dofs=sol["P"],
        grid_x=grid["x"],
        grid_y=grid["y"],
        grid_u=grid["u"],
        grid_v=grid["v"],
        grid_speed=grid["speed"],
        grid_pressure=grid["pressure"],
        solid_mask=grid["solid"],
        polygon=final["poly"],
    )


def evidence_payload(
    levels: list[dict[str, Any]],
    convergence: dict[str, Any],
    interpolation_test: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    serial_levels = []
    for level in levels:
        entry = {k: v for k, v in level.items() if k not in ("solution", "poly", "grid")}
        serial_levels.append(entry)
    final = levels[-1]
    return {
        "problem": {
            "id": "P4",
            "name": "DFG 2D-1 stationary cylinder benchmark",
            "domain": "[0,2.2] x [0,0.41] minus polygonal approximation of "
            "circle center (0.2,0.2), radius 0.05",
            "nu": NU,
            "rho": 1.0,
            "inlet_umax": U_MAX,
            "inlet_mean": U_MEAN,
            "reynolds": U_MEAN * DIAMETER / NU,
            "outflow": "nu partial_n u - p n = 0; no pressure Dirichlet condition",
            "source_url": SOURCE_URL,
        },
        "method": {
            "finite_element": "Taylor-Hood P2/P1 on triangles",
            "nonlinear": "Picard fixed-point continuation followed by Newton corrections",
            "mesh": "Delaunay triangulation with explicit polygonal cylinder boundary; "
            "circle segment count increases with level",
            "plot_mask": "same polygon used by the final mesh; solid is never interpolated",
            "picard_max": args.picard_max,
            "newton_max": args.newton_max,
            "nonlinear_tol": args.nonlinear_tol,
            "python": sys.executable,
            "packages": {
                "scikit-fem": skfem.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "scalar_interpolation_test": interpolation_test,
        },
        "levels": serial_levels,
        "convergence": convergence,
        "reference": REFERENCE,
        "final_metrics": final["diagnostics"],
        "limitations": [
            "The obstacle is polygonal at each level; circle-segment refinement is explicit.",
            "No certificate is issued unless the declared field and QoI convergence criteria pass.",
            "The figures show the resolved fields only; the separate illustration contains no solved field.",
            "Drag/lift use sigma = nu grad(u) - p I and the outward-from-cylinder normal.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--picard-max", type=int, default=35)
    parser.add_argument("--newton-max", type=int, default=12)
    parser.add_argument("--nonlinear-tol", type=float, default=2e-9)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="coarse/medium levels only, useful for a first viability check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = [
        {"name": "coarse", "nx": 34, "ny": 8, "ncircle": 32},
        {"name": "medium", "nx": 52, "ny": 12, "ncircle": 48},
        {"name": "fine", "nx": 76, "ny": 18, "ncircle": 64},
    ]
    if args.quick:
        configs = configs[:2]
    levels: list[dict[str, Any]] = []
    for config in configs:
        print(
            f"[DFG] {config['name']}: nx={config['nx']} ny={config['ny']} "
            f"ncircle={config['ncircle']}",
            flush=True,
        )
        level = run_level(config, args)
        level["grid"] = field_grid(level["solution"], level["poly"], nx=260, ny=86)
        levels.append(level)
        diag = level["diagnostics"]
        print(
            f"[DFG] {config['name']} elements={level['elements']} "
            f"Nu={level['dofs_velocity']} Np={level['dofs_pressure']} "
            f"Picard={level['picard']['iterations']} "
            f"Newton={level['newton']['iterations']} "
            f"div={diag['divergence_l2']:.3e} "
            f"fluxrel={diag['flux_closure_rel']:.3e} "
            f"Cd={diag['Cd']:.8f} Cl={diag['Cl']:.8f} "
            f"p_diff={diag['p_diff']:.8f}",
            flush=True,
        )

    convergence = convergence_report(levels)
    final = levels[-1]
    # Use a denser grid only for the final visual, while the convergence
    # comparisons stay on the common inexpensive diagnostic grid.
    final_grid = field_grid(final["solution"], final["poly"], nx=420, ny=140)
    save_solution_npz(final, final_grid)
    save_solution_figure(final_grid, final["poly"])
    save_illustration(final["poly"])
    interpolation_test = scalar_affine_interpolation_test(
        final["solution"]["mesh"], final["poly"]
    )
    evidence = evidence_payload(levels, convergence, interpolation_test, args)
    (HERE / "evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[DFG] status={convergence['status']} "
        f"artifacts={HERE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
