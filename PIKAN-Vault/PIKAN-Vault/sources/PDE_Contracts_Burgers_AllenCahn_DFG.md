---
title: "Fuentes de referencia de Burgers, Allen–Cahn y DFG 2D-1"
year: 2026
venue: "Registro de contratos PDE para AF_PIKANs"
layer: FOUND
status: reading
source_role: literature_scout
tags: #found #pde #burgers #allen-cahn #dfg #benchmark
---

# Fuentes de referencia de Burgers, Allen–Cahn y DFG 2D-1

**Registro de fuentes primarias usadas en el brief de avance · 2026**

> Reúne artículos, benchmark oficial y documentos que definen/contextualizan las PDE. Los valores solo se comparan cuando coinciden geometría, dimensión, condiciones y normalización.

## Burgers

- Raissi, Perdikaris y Karniadakis, PINNs, Journal of Computational Physics (2019): https://doi.org/10.1016/j.jcp.2018.10.045
- PINNacle, Hao et al. (2024): https://arxiv.org/abs/2306.08827
- Wu et al. (2023), estudio de muestreo adaptativo, Computer Methods in Applied Mechanics and Engineering 403, 115671: https://doi.org/10.1016/j.cma.2022.115671 (versión arXiv inicial: https://arxiv.org/abs/2207.10289).
- El contrato de Burgers transitorio usado por EXP-0005 es una instancia registrada con referencia Cole–Hopf y se conserva separada del Burgers estacionario de EXP-0002.

## Allen–Cahn

- Allen y Cahn (1979) presenta el contexto físico de movimiento de fronteras de antífase y coarsening en aleaciones binarias: https://doi.org/10.1016/0001-6160(79)90196-2.
- El contrato escalar 1D del piloto sigue el ejemplo de difusión-reacción de DeepXDE: `D=0.001`, reacción `5(u-u³)`, `x∈[-1,1]`, `t∈[0,1]`, `u(x,0)=x²cos(πx)` y Dirichlet `u(±1,t)=-1`: https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/allen.cahn.html
- Lee et al. (2025) publican otros problemas benchmark de Allen–Cahn 1D–3D para PINNs: https://doi.org/10.3934/math.2025335. Se registran como contexto y candidato comparativo; no se tratan como el mismo contrato sin reconciliar parámetros, datos, condiciones y referencia.
- Registro detallado de literatura, señales de energía y contraste de arquitecturas: [[Allen_Cahn_EXP-0003_References_2026-09-23]].
- La referencia candidata local usa diferencias finitas espaciales centradas y BDF implícito. No se marca convergida mientras el criterio de efecto no tenga definición operativa y no se audite el estudio completo.

## DFG 2D-1 a Reynolds 20

- FEATFLOW define el benchmark estacionario 2D-1, la geometría del canal/cilindro, las condiciones de frontera y los QoIs Cd, Cl y Δp. Publica como targets Cd=5.57953523384, Cl=0.010618948146 y Δp=0.11752016697: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
- Schäfer, Turek, Durst, Krause y Rannacher (1996), “Benchmark Computations of Laminar Flow Around a Cylinder,” capítulo de *Flow Simulation with High-Performance Computers II*, pp. 547–566, DOI https://doi.org/10.1007/978-3-322-89849-4_39. Springer registra NNFM vol. 48; FEATFLOW y algunos catálogos indican vol. 52, así que DOI/páginas se conservan como identificadores estables.
- El informe 411 de Turek, Mierka, Hysing y Kuzmin (2010), “A high order 3D FEM-Level Set approach for multiphase flows with application to monodisperse droplet generation,” incluye una tabla de cilindro Re=20 en §5.1/Table I. Su contexto es un manuscrito FEM 3D y reporta Cd≈6.1853, Cl≈0.0094012 y Δp≈0.171019, frente a los targets de FEATFLOW 2D-1. No se interpreta como discrepancia directa entre referencias equivalentes: hay que reconciliar dimensión, geometría y normalización antes de usar esas cifras para evaluar el contrato 2D-1.
- PDF del informe 411: https://www.mathematik.tu-dortmund.de/papers/TurekMierkaHysingKuzmin2010.pdf
- La entrada anterior usaba el título descriptivo “DFG Benchmark Reference Values…” para ese PDF. Se corrigieron el título y el tipo bibliográfico; el informe se cita como documento técnico, no como artículo final de revista con volumen/páginas que el manuscrito no especifica.

## Registro de citas

    @article{raissi2019physicsinformed,
      title = {Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
      author = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
      journal = {Journal of Computational Physics},
      volume = {378},
      pages = {686--707},
      year = {2019},
      doi = {10.1016/j.jcp.2018.10.045}
    }

    @misc{featflow_dfg2d1_re20,
      title = {DFG Benchmark 2D-1: Flow Around a Cylinder at Reynolds Number 20},
      organization = {FEATFLOW},
      url = {https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html},
      note = {Official contract, boundary conditions, quantities of interest, and published reference values}
    }

    @incollection{schaefer1996benchmarkcylinder,
      title = {Benchmark Computations of Laminar Flow Around a Cylinder},
      author = {Schaefer, Michael and Turek, Stefan and Durst, F. and Krause, E. and Rannacher, R.},
      booktitle = {Flow Simulation with High-Performance Computers II},
      editor = {Hirschel, Ernst H.},
      pages = {547--566},
      year = {1996},
      publisher = {Vieweg+Teubner Verlag},
      doi = {10.1007/978-3-322-89849-4_39},
      note = {Springer metadata reports NNFM volume 48; FEATFLOW records volume 52}
    }

    @techreport{turek2010highorder3dlevelset,
      title = {A high order 3D FEM-Level Set approach for multiphase flows with application to monodisperse droplet generation},
      author = {Turek, Stefan and Mierka, Otto and Hysing, Shu-Ren and Kuzmin, Dmitri},
      institution = {TU Dortmund, Institute of Applied Mathematics},
      type = {Ergebnisberichte des Instituts fuer Angewandte Mathematik},
      number = {411},
      year = {2010},
      url = {https://www.mathematik.tu-dortmund.de/papers/TurekMierkaHysingKuzmin2010.pdf},
      note = {Section 5.1/Table I reports a Re=20 cylinder case in a 3D FEM-Level Set manuscript; not interchangeable with DFG 2D-1 before reconciling case definition and normalization}
    }

## Heat 1D — fuente de definición

- MIT OpenCourseWare, 18.085, *Problem Set 8 Solutions*: contexto docente para conducción 1D y solución por modos seno: https://math.mit.edu/classes/18.085/summer2014/pset/pset8_sol.pdf (consultado 2026-09-23).

    @misc{mit_ocw_18085_pset8,
      title = {18.085 Problem Set 8 Solutions},
      organization = {MIT OpenCourseWare},
      url = {https://math.mit.edu/classes/18.085/summer2014/pset/pset8_sol.pdf},
      note = {Consultado 2026-09-23; fuente docente para la solución Heat 1D}
    }
