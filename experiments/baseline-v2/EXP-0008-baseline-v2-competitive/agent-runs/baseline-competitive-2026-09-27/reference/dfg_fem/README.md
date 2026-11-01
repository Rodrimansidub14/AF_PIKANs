# DFG FEM reference draft — DRAFT_UNVERIFIED

This directory contains an original, protocol-aligned scaffold for a stationary DFG 2D-1 Re=20 reference: `make_mesh.geo` defines the curved channel/cylinder geometry and physical tags; `solve_dfg.py` generates the planned mesh levels, sets up Taylor–Hood P2/P1, solves Stokes followed by a continuation to steady Navier–Stokes with a natural outlet, and writes per-level fields and QoIs. `manifest.json` freezes the intended physical contract and stopping rules.

**This is not a numerical reference.** The script, Gmsh geometry, nonlinear solve, pressure probes, field export, and QoIs have not been run or validated. Do not use any output from this draft as a locked evaluation target. No solve, mesh generation, preflight, test, or local validator was run for this task.

## Contract encoded

- Fluid region `[0,2.2] × [0,0.41]` minus the circle centered at `(0.2,0.2)`, radius `0.05`; the cylinder is deliberately below the channel midline `y=0.205`.
- `rho=1`, `nu=0.001`, inlet `u_x=4*0.3*y*(0.41-y)/(0.41^2)`, `u_y=0`; no-slip at both channel walls and cylinder; no essential condition or explicit weak-form boundary term at outlet.
- Quadratic curved Gmsh geometry; physical groups `inlet=1`, `walls=2`, `cylinder=3`, `outlet=4`, `fluid=5`. The target sizes for L0–L3 and contingency L4–L5 match `reference-protocol-ac-dfg.md`.
- The weak form and stress convention follow the local protocol. Drag/lift use `sigma=nu*grad(u)-pI` and the outward normal of the solid cylinder; pressure difference is `p(0.15,0.2)-p(0.25,0.2)`. Published FEATFLOW targets are recorded in the manifest; no comparison has been computed.

## Intended environment and invocation

`environment.yml` is a **draft conda environment specification, not a resolved lockfile**. It targets Linux/WSL x86-64 with DOLFINx 0.11, Gmsh 4.15, MPI/PETSc and a PETSc build with MUMPS. Exact build strings, ABI compatibility, and availability of the MUMPS solver must be resolved and recorded before use. The DOLFINx API shape was checked against its 0.11 documentation, but no runtime is installed or exercised here.

Intended invocation after environment review, not run in this task:

```bash
python solve_dfg.py --output runs
python solve_dfg.py --output runs-with-contingency --include-contingency
```

The script invokes `gmsh` from `PATH` with `-setnumber lc_bulk ... -setnumber lc_cyl ...`; `make_mesh.geo` defaults to L0. Output is per-level Gmsh mesh, XDMF mesh for visualization, FE coefficient arrays, velocity/pressure at a fixed common grid, probe pressures, and `qois.json`. No SHA sum file is created until final output bytes exist.

## Gaps before this can be called a reference

1. Resolve and freeze a compatible Python/DOLFINx/Basix/UFL/FFCx/PETSc/petsc4py/MPI/Gmsh environment; verify the `.geo` CLI parameter override, Boolean result, physical group counts, and curved P2 geometry. The environment file is intentionally not a lockfile.
2. Run the solver first on L0 in a disposable environment and review the mixed-space BC construction, nonlinear continuation, PETSc line search, SNES/KSP status, and memory/time. The scaffold records solver reasons and halves continuation steps, but it does **not** yet establish the protocol's two-consecutive-iteration relative update criterion or verify the final relative linear residual `<=1e-10`.
3. Resolve the one-sided pressure trace at the two FEATFLOW probe points: both points lie exactly on the cylinder boundary. The current point locator is an unverified direct evaluation, not an approved projection/trace rule.
4. Independently audit traction sign, the cylinder normal, quadrature convergence, and the chosen FEATFLOW stress convention. The local protocol requests traction QoIs as primary and reaction forces only as a diagnostic; reaction-force cross-check is not implemented here.
5. Validate that field sampling excludes the solid, maps the same physical points on every level, and supports the required pressure mean-removal comparison. The script writes a fixed grid but does not yet compute the protocol's mesh-to-mesh convergence table or apply acceptance gates.
6. Complete L0–L3 (L4/L5 only if needed), report actual elements/nodes/DOFs/quality/runtime/solver statistics, test two consecutive refinements against every protocol threshold and FEATFLOW target, then have an independent review of QoIs and artifacts. Only after that should final XDMF/HDF5 outputs and SHA-256 sums be frozen.

## Source basis

No external implementation block was copied. The geometry, PDE data, natural outlet, QoIs, and gates follow the project's [`reference-protocol-ac-dfg.md`](../../reference-protocol-ac-dfg.md), which cites [FEATFLOW/TU Dortmund's DFG Re=20 benchmark](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html). The Taylor–Hood pair and outflow interpretation are cross-checked against the [FEniCS Hands-on cylinder tutorial](https://fenics-handson.readthedocs.io/en/latest/navierstokes/doc.html) and [jNO's DFG 2D-1 tutorial](https://fhg-iisb.github.io/jNO/tutorials/08-fem-and-varpinns/navier-stokes-cylinder-dfg/); the latter's reaction-force path is documented as a future diagnostic, not substituted for FEATFLOW's primary traction QoI. The draft APIs are based on the [DOLFINx 0.11 nonlinear PETSc interface](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.fem.petsc.html), [DOLFINx 0.11 Gmsh reader](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.io.gmsh.html), and [Gmsh documentation](https://gmsh.info/doc/texinfo/).
