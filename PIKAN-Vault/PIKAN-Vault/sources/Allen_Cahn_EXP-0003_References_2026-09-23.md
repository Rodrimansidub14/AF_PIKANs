---
title: "Allen–Cahn: fuentes del piloto exploratorio EXP-0003"
year: 2026
venue: "Registro bibliográfico de AF_PIKANs"
layer: FOUND
status: consulted
source_role: research_brief
tags: #found #allen-cahn #pinn #pikan #sampling #energy
---

# Allen–Cahn: fuentes del piloto exploratorio EXP-0003

Registro de literatura y documentación efectivamente consultadas para definir,
interpretar y documentar el piloto. Las entradas BibTeX están centralizadas en
`AF_PIKANs_Research_Brief_References_2026-09-23.bib`.

## Contrato del problema

- **Allen y Cahn (1979)** — fundamento físico de la dinámica de fronteras de
  antífase y el coarsening en aleaciones binarias. DOI:
  https://doi.org/10.1016/0001-6160(79)90196-2. BibTeX:
  `allen1979antiphaseboundary`.
- **DeepXDE 1.12.2, ejemplo Allen–Cahn** — fuente operativa del caso escalar
  1D que se reproduce: dominio `[-1,1]×[0,1]`, `D=0.001`, reacción
  `5(u-u³)`, condición inicial `x²cos(πx)` y transformada dura para IC y
  Dirichlet. La documentación usa 8.000 puntos interiores, 400 de frontera,
  800 iniciales, y Adam `1e-3` seguido por L-BFGS. EXP-0003 conserva el PDE y
  transformada, pero cambia explícitamente el muestreo y el presupuesto;
  por ello no es reproducción numérica del entrenamiento de DeepXDE. BibTeX:
  `deepxde_allencahn_docs`.
- **Lee et al. (2025)** — benchmark reciente de Allen–Cahn para PINNs, con
  problemas 1D–3D y código fuente. Su alcance apoya el uso de Allen–Cahn como
  prueba de dinámicas de interfaz, pero no se asume que sus parámetros,
  dominios o targets coincidan con EXP-0003. DOI:
  https://doi.org/10.3934/math.2025335. BibTeX:
  `lee2025allencahnpinnbenchmark`.

## Arquitectura y señales

- **Shukla et al. (2024), comparación FAIR MLP/KAN** — incluye un caso
  Allen–Cahn y compara representaciones PINN/PIKAN; informa límites de
  estabilidad y reporta hardware/configuración propios. Se usa para motivar
  comparaciones controladas de representación y semillas, no para atribuir
  una expectativa de superioridad a la RBF-PIKAN de este piloto. DOI:
  https://doi.org/10.1016/j.cma.2024.117290. BibTeX:
  `shukla2024comprehensivefaircomparisonmlp`.
- **Kütük y Yücel (2024; v2, 2025), energy-dissipation-preserving PINN** —
  estudia incorporar disipación de energía en la pérdida de PINNs de
  Allen–Cahn. EXP-0003 solo mide la trayectoria de energía como señal física;
  no la añade a la pérdida ni impone monotonía como gate. BibTeX:
  `kutuk2024energydisspinnallen`.
- **Li, Yu y Wang (2025), EDRAS** — preprint de muestreo adaptativo guiado por
  densidad local de disipación energética para Allen–Cahn, con resultados en
  dominios 2D irregulares y dinámica superficie-volumen. Es una referencia
  directa para una futura hipótesis de focalización después de cerrar referencia
  y métricas; EXP-0003 solo mide energía y no implementa ni compara EDRAS. Los
  dominios y condiciones de borde del preprint no coinciden con el contrato 1D
  actual. arXiv:2507.09757. BibTeX: `li2025edrasallencahn`.
- **Wu et al. (2023), RAD/RAR-D** — compara muestreo no adaptativo,
  cuasi-Monte Carlo y estrategias adaptativas residuales. Justifica evaluar
  más adelante una intervención de muestreo contra Sobol con pool de
  evaluación independiente y coste medido; no predice que RAD mejorará este
  problema. BibTeX: `wu2023adaptivesamplingpinns`; nota existente:
  [[Wu_2022_Adaptive_Sampling_PINNs]].
- **SPIKANs (Jacob et al., 2025)** — referencia arquitectural SciML para
  evaluar alternativas separables cuando el coste crezca; no es parte del
  modelo ni de la hipótesis de EXP-0003. DOI:
  https://doi.org/10.1088/2632-2153/ae05af. BibTeX: `jacob2025spikans`.

## Cómo se aplican al piloto

EXP-0003 fija centros y anchos de la base RBF y usa puntos Sobol scrambled
fijos; mide residuo sobre 16.384 puntos independientes, cumplimiento numérico
de las trazas duras, energía, cobertura/saturación de bases, autodiferenciación
y coste. El error de campo sigue bloqueado hasta converger y auditar la
referencia BDF. Estas fuentes contextualizan el diseño; no sustituyen la
validación, no elevan el piloto a confirmatorio y no justifican un claim SOTA.

## Registro

- BibTeX central: `sources/AF_PIKANs_Research_Brief_References_2026-09-23.bib`
- Contratos PDE: [[PDE_Contracts_Burgers_AllenCahn_DFG]]
- Muestreo adaptativo: [[Wu_2022_Adaptive_Sampling_PINNs]]
- Comparación PIKAN/MLP y benchmarks: [[Takamoto_2022_PDEBench]], [[Hao_2024_PINNacle]]
