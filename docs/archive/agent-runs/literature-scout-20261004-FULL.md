# Informe completo del scout ampliado — 2026-10-04 (transcripción compacta de las tablas del subagente)

Solo lectura. Provisional: nada de aquí prueba ausencia. Estados: ABIERTO-RESUMEN (página abierta; WebFetch devolvió un resumen generado, no el texto crudo; ningún PDF completo leído), METADATOS (OpenAlex/Crossref), SNIPPET (solo resultado de búsqueda), NO VERIFICADO. Fallaron: ScienceDirect, MDPI, preprints.org, SSRN, Springer (403/redirección); Semantic Scholar (429); PDF de arXiv llegaron ilegibles.
Nota: este archivo es una copia del contenido; el resumen accionable y la corrección sobre "novedad" están en `literature-scout-20261004.md`.

## A. KAN/PIKAN en piezoelectricidad y multifísica electromecánica

| # | Fuente | Qué prueba | Estado |
|---|---|---|---|
| A1 | Zhang, Chen, An, Peng, Yang, Peng, Song, Zhao (2025). PINN para deflexión no lineal de IPMC basado en KAN. *Eng. Appl. Artif. Intell.* 144, PII S0952197625001265 (doi 10.1016/j.engappai.2025.110126) | PINN+KAN para actuador IPMC (electroactivo iónico, **no piezoeléctrico**); 29 referencias, todas de IPMC más Karniadakis 2021, Raissi 2019, Liu 2024. Antecedente de PINN+KAN en actuador electromecánico. | METADATOS (Crossref); abstract y texto no leídos |
| A2 | Cong, Wan, Tan, Huang. Optimización de parámetros de transductores piezoeléctricos tubulares con redes neuronales. ASME OMAE | KAN como sustituto de FEM (COMSOL) para diseño inverso de transductor piezoeléctrico; no resuelve la EDP | ABIERTO-RESUMEN |
| A3 | FBG vs sensores piezoeléctricos con MLP y KAN para diagnóstico de fallas. *J. Lightwave Technol.* 2025 (doi 10.1109/jlt.2025.3634028) | KAN con datos de sensores; irrelevante para PDE | METADATOS |
| A4 | Ponomarev et al. (2025). Soluciones neuronales universales para BVP de la "ecuación piezoeléctrica". *Vestnik Tyumen State Univ.* 11(4) | Keywords PINN, KAN; **falso amigo**: es piezoconductividad (difusión de presión en medios porosos); inferencia del agente, cuerpo no leído | ABIERTO-RESUMEN |
| A5 | Xin, Wang, Shi, Sun (2026). PINN-Based KANs con RAR-D para problemas de interfaz elípticos. arXiv 2602.01876 | KAN con interfaces; no piezo | ABIERTO-RESUMEN |
| A6 | PIKAN en elasticidad multimaterial para empaquetado electrónico. arXiv 2508.16999 | sin acoplamiento eléctrico | SNIPPET+METADATOS |
| A7 | KAN-PINN para EMI/EMC. arXiv 2405.11383 | electrostática, no electromecánico | SNIPPET |
| A8 | Wang, Sun, Bai, Anitescu, Eshaghi, Zhuang, Rabczuk, Liu. KINN. *CMAME* 433 (2025), arXiv 2406.11045 | mecánica de sólidos; sin piezo | ABIERTO-RESUMEN (Crossref) |
| A9 | Mehtaj y Banerjee (2025). SciML para ondas guiadas y SAW. *Sensors* 25(5):1401 (PMC11902327) | revisión; **no menciona KAN**; PINN en SAW "limited in scope"; no hay frase de "future work PIKAN". Existe una segunda revisión *Sensors* 25(12):3588 (403) | ABIERTO-RESUMEN (PMC) |
| A10 | Shukla, Di Leoni, Blackshire, Sparkman, Karniadakis (2020). PINN para grietas superficiales por ultrasonido. *J. Nondestruct. Eval.* 39:61 (arXiv 2005.03596) | placa **metálica**, ondas superficiales; no piezoeléctrico | SNIPPET+METADATOS |
| A11 | Georgiou et al. (2026). Marco de respuesta en frecuencia para TMD piezoeléctrico. arXiv 2608.07255 | modelo físico directo + red inversa; no PINN de residuo ni KAN | SNIPPET |
| A12 | Boreiry, Bahrami, Hairi Yazdi (2026). Dual-Output PINNs para harvesters piezoeléctricos de baja frecuencia. *J. Vib. Eng. Technol.* (doi 10.1007/s42417-026-02641-2) | **cita a KINN** (único trabajo piezo en citantes de KINN); título dice PINN; no se pudo abrir | METADATOS (OpenAlex) |
| A13 | Boundary integrated NNs para fractura en materiales elásticos y piezoeléctricos. arXiv 2502.21253 | **no usa KAN**; cita a KINN | ABIERTO-RESUMEN |
| A14 | PINN para problemas flexoeléctricos. arXiv 2506.21810 | energía + DEM, sin KAN | SNIPPET |
| A15 | Citantes de KAN (W4396821501), Shukla 2024 y Toscano 2025 con filtros piezo/electromecánico/ferroeléctrico/SAW en OpenAlex | KAN: 1 resultado irrelevante; Shukla 2024 y Toscano 2025: 0. Cobertura incompleta (538 citas del ID arXiv de KAN; la versión ICLR tiene otro ID no consultado) | METADATOS |
| A16 | Citantes de KINN (205; ~198 vistos) | solo A12 y A13 con piezoelectricidad; ninguno PIKAN piezoeléctrico; resúmenes truncados | METADATOS |
| A17 | Búsquedas directas en arXiv (Kolmogorov-Arnold + piezoelectric/piezoelectricity/electromechanical/ferroelectric/acoustic wave) | 0 resultados cada una ("KAN piezoelectric": 2 falsos positivos por "Kaan") | ABIERTO-RESUMEN |
| A18 | OpenAlex "kolmogorov-arnold AND piezoelectric" (3: A3 y dos tangenciales) y "...AND (electromechanical OR ferroelectric OR surface acoustic OR flexoelectric OR magnetoelectric OR electroelastic)" (3: modelado de actuador y dos diagnósticos) | sin PIKAN piezoeléctrico | METADATOS |

## B. KAN/PIKAN y PINN en Navier–Stokes con cilindro / DFG 2D-1

| # | Fuente | Qué prueba | Estado |
|---|---|---|---|
| B1 | FEATFLOW DFG 2D-1 Re=20 | Cd 5.57953523384, Cl 0.010618948146, Δp 0.11752016697 (coinciden con el vault) | SNIPPET |
| B2 | Shah (2026). Efectos de interacción en PINN para desprendimiento de vórtices de NS. arXiv 2608.19632 | **CORREGIDO tras leer el PDF completo:** Cd 5.567 (0.2 % vs 5.579) es la **simulación OpenFOAM** (icoFoam, 50 184 celdas, Tabla 8), no el PINN. El PINN (MLP 8×128 tanh, AdamW 1e-3, 125k pasos, 8 000 puntos, formulación transitoria con IC en reposo) se evalúa a Re=20 en t ≥ 2 s contra OpenFOAM: L2 1.1 % (Ux), 2.1 % (Uy), 2.8 % (p); diferencia de presión "dentro de 1 %" vs OpenFOAM. Sin Cd/Cl del PINN contra FEATFLOW. Sin KAN. El resto del artículo es sobre Re=100 transitorio | LEÍDO COMPLETO (PDF, págs. 1–5 y 13–16) |
| B3 | Guo, Sun, Li, Yuan, Wang (2025). ChebPIKAN para mecánica de fluidos. *Phys. Fluids* (arXiv 2411.04516; doi 10.1063/5.0284999) | cilindro con dominio x∈[1,8], y∈[−2,2], D=1; Chebyshev grado 10; no es el canal DFG; sin comparación con QoI; sin muestreo adaptativo | ABIERTO-RESUMEN |
| B4 | Kolmogorov-Arnold PointNet (arXiv 2408.02950) y su versión con física (arXiv 2504.06327) | cilindro en cavidad con convección natural; no DFG | ABIERTO-RESUMEN (2504.06327) |
| B5 | Cui, Cao, Liao, Wu (2025). PI-KAN para NS. *Phys. Fluids* (doi 10.1063/5.0257677) | NS forward, MSE ~1e-5; problemas no identificados | METADATOS |
| B6 | Lee y Lee (2026), PI-GLaK (doi 10.1063/5.0341290) | NS estacionario en geometrías vasculares | METADATOS |
| B7 | Wavelet Fourier para dinámica multiescala. arXiv 2607.09071 | cilindro 2D wake; sin Re ni DFG en abstract | ABIERTO-RESUMEN |
| B8 | SIMPLE-PINN. arXiv 2603.24013 | cilindro Re=100 inestacionario, MLP | ABIERTO-RESUMEN |
| B9 | Chuang y Barba (2023). Limitaciones de PINN en desprendimiento de vórtices. arXiv 2306.00230 | cilindro vs PetIBM; no DFG | ABIERTO-RESUMEN |
| B10–B11 | OpenAlex: PINN + (Schäfer/Turek/DFG benchmark) → solo B2; Kolmogorov-Arnold + (cylinder/NS/vortex shedding) → 67 registros, ninguno DFG 2D-1 por título | | METADATOS |

## C. Muestreo adaptativo, focalización y bases adaptativas

| # | Fuente | Qué hace | Estado |
|---|---|---|---|
| C1 | **Leong y Sim (2026). RA-PIKAN: densificación de grilla B-spline dirigida espacialmente. SSRN 7022851 (jun-2026)** | Entrenamiento de tres ciclos: densificación de grilla B-spline por diagnóstico del residuo, extensión uniforme, probe-gated training y aumento adaptativo de puntos de colocación por residuo. Burgers, Allen–Cahn, advección-difusión, Duffing; Allen–Cahn 4.80e-4 y 4.91e-4, mejora 1.26× y 1.83× sobre extensión uniforme. **B-spline, no RBF.** Preprint sin arbitraje | SNIPPET; SSRN 403 |
| C2 | Rigas et al. (2024). Adaptive Training of Grid-Dependent PIKANs. *IEEE Access* (arXiv 2407.17611) | extensión de grilla con transición de estado adaptativa; disparada por la grilla, no por el residuo; hasta 43 % menos error | ABIERTO-RESUMEN |
| C3 | Rigas et al. (2026). Dynamic Framework for Grid Adaptation in KANs. arXiv 2601.18672 (IJCNN 2026) | nudos según densidad de importancia por curvatura; Helmholtz −23.3 %; adapta la grilla, no los puntos | ABIERTO-RESUMEN |
| C4 | Xin et al. (2026). KAN + RAR-D. arXiv 2602.01876 | RAR-D sobre puntos con doble KAN por subdominios; no adapta la base | ABIERTO-RESUMEN |
| C5 | Chiu, Cheung, Braga-Neto, Lee, Li (2026). Free-RBF-KAN. arXiv 2601.07760 (v3, 6-feb-2026; LLNL OSTI 10.2172/3389234) | formas y grillas RBF entrenables; aproximación universal; calor y Helmholtz; sin muestreo por residuo | ABIERTO-RESUMEN |
| C6 | PINN para difusión transitoria con interfaces: KAN vs MLP. *CMAME* 2026 (doi 10.1016/j.cma.2026.119219) | RBF-KAN con centros entrenables + enriquecimiento polinómico + dominio temporal progresivo; RBF-KAN solo no supera uniformemente al MLP; no se sabe si usa muestreo por residuo | ABIERTO-RESUMEN (iMechanica)+METADATOS |
| C7 | Noorizadegan, Wang, Ling. Making Gaussian KANs Reliable and Accurate. arXiv 2604.21174; Partition-of-Unity Gaussian KANs, arXiv 2604.23599 | regla de diseño de la escala; centros fijos | ABIERTO-RESUMEN |
| C8 | Cavoretto, De Rossi, Haider, Noorizadegan. Adaptive RBF-KAN. arXiv 2605.21534 | parámetros de forma dinámicos con LOOCV; sin PINN en el resumen | ABIERTO-RESUMEN |
| C9 | Ma, Ren, Larsson. PINN RBF adaptativas para option pricing. arXiv 2601.12704 | posición de neuronas RBF refinada por residuo (red RBF, no KAN) | ABIERTO-RESUMEN |
| C10 | Shao, Pieper, Tian. Sparse RBF Networks para PDE. arXiv 2601.17562 | inserción/borrado de kernels con centros y radios entrenables; disparador no preciso | SNIPPET+parcial |
| C11 | Wu, Zhu, Tan, Kartha, Lu (2023). Estudio de muestreo adaptativo basado en residuo. *CMAME* 403:115671 (arXiv 2207.10289) | define RAD y RAR-D | ABIERTO-RESUMEN |
| C12 | Toscano, Chen, Oommen, Darbon, Karniadakis (2026). vRBA. *npj Artif. Intell.* 2(1) (doi 10.1038/s44387-026-00084-4) | contenido tomado de la nota del vault | METADATOS (Crossref) |
| C13 | Ouyang et al. (2026). RAMS. *Adv. Intell. Discov.* 2(3) (doi 10.1002/aidi.202500214) | muestras movidas por gradiente adversarial; descripción del vault | METADATOS (Crossref) |
| C14 | Peng et al. (2022). RANG. arXiv 2205.01051; R3 (Daw et al.) y Nabian et al. | RANG: muestreo guiado por residuo con densidad casi uniforme tipo RBF-FD; R3 y Nabian: **NO VERIFICADO** | RANG ABIERTO-RESUMEN |
| C15 | Chen, Howard, Stinis (2025), arXiv 2511.05452; PACMANN arXiv 2411.19632; AB-PINNs (*MLST* 2026, 10.1088/2632-2153/ae8638); Wang, Sankaran, Perdikaris causal (*CMAME* 2024, 10.1016/j.cma.2024.116813) | ponderación+muestreo; puntos movidos por gradiente del residuo; base adaptativa guiada por residuo; causalidad | arXiv/Crossref resumen; causal solo METADATOS |
| C16 | Otros títulos en citantes de KINN: Residual-Based Attention in KANs; adaptive-grid KAN Czochralski; Adaptive-Grid Residual KAN BSDE; Weak-form evolutionary KANs; Sinc KAN; Multilevel Training for KANs (arXiv 2603.04827) | campo activo de grilla adaptativa en KAN; solo 2603.04827 con resumen | METADATOS |

## D. PINN en piezoelectricidad

| # | Fuente | Detalle | Estado |
|---|---|---|---|
| D1 | Tandis y Tandis (2025). Placas piezoeléctricas. *EAAI* 160 (doi 10.1016/j.engappai.2025.111847) | 5 ecuaciones de conservación + 9 constitutivas; adimensional; error relativo medio espacial vs COMSOL 0.36 %–1.23 % (mayor en eléctricas); arquitectura/optimizador no en el abstract | ABIERTO-RESUMEN |
| D2 | González, Higueros, Shaw, Barrientos (2026-09-22). Mixed-Variable PINN para piezoelectricidad directa y conversa en bimorfo en voladizo. Preprints.org 202609.1914 | red mixta de 8 salidas, pérdida de primer orden, validación con FEM acoplado, PVDF. **VERIFICADO en el PDF: converso 0.107 / 0.150 / 0.045; directo 0.117 / 0.153 / 0.067** (u_x, u_y, potencial; el subagente los había puesto al revés); inconsistencias admitidas entre tensiones predichas y reconstruidas | ABIERTO-RESUMEN (abstract reconstruido desde OpenAlex; página 403) |
| D2b | Mismos autores. Enhanced Multiphysics Simulation via a Tailored PINN Architecture... Preprints.org 202606.0462 (2026-06-05) | L2 1.96e-1, 1.93e-1, 1.03e-3 | ABIERTO-RESUMEN (OpenAlex) |
| D3 | Mouratidou, Daraki, Drosopoulos, Foutsitzi, Larbi, Deü, Ohayon, Stavroulakis. PINN para sistemas piezoeléctricos con shunt. *Acta Mechanica* (en línea 2026-02-05; doi 10.1007/s00707-025-04619-9) | multi-PINN, Galerkin, partición temporal, validado con MATLAB FEM; **no confundir con** Acta Mech. 235 (2024), que no es piezo | ABIERTO-RESUMEN (OpenAlex) |
| D4 | Bharadwaj y Thovelil. arXiv 2602.13811 | **RETIRADO** (v2 2026-07-18 y v3 2026-08-20) por fallas críticas en el código de simulación; no citar sus L2 (2.34 %, 4.87 %) | ABIERTO-RESUMEN |
| D4b | Bharadwaj (2026-08-10). Neural solvers termodinámicamente admisibles. *J. Appl. Phys.* (10.1063/5.0345312) | 1D electro-elastodinámica; BC/IC exactas; cuasi-Newton; L2 del potencial O(1e-5); mismo autor que D4, relación no afirmable | ABIERTO-RESUMEN |
| D5 | Nguyen, Torri, Rochus (2024). Micro-bender piezoeléctrico. *J. Micromech. Microeng.* 34(11):115004 | PINN con datos experimentales; detalles inaccesibles | ABIERTO-RESUMEN (imec) |
| D7 | Solo títulos/DOI: PINN identificación de parámetros en harvester (SPIE 12946); semiconductores piezoeléctricos (Sci. China 2025; J. Appl. Phys. 2025); Spectral Integrated NNs; control de vibración con PINN (IEEE Access 2025); multi-network con Fourier features (MSSP 2026); Hybrid-PINN para histéresis de actuadores (CMES 2026); Physics-informed ELM en vigas flexoeléctrico-piezoeléctricas (Phys. Scr. 2026); metamateriales piezoeléctricos; harvesting en concreto metamaterial (MAMS 2025); neural operators para biosensores (2026) | sin arquitectura ni errores | METADATOS |

OpenAlex: 107 registros para (physics-informed AND piezoelectric); se vieron 96; ninguno decía "Kolmogorov" en el título.

## Conclusiones redactables propuestas por el scout (con la corrección del investigador aplicada)

- **A (media):** "En la búsqueda realizada (consultas del anexo; arXiv, OpenAlex, Crossref y búsqueda web general) no se encontraron trabajos que resuelvan las ecuaciones acopladas de la piezoelectricidad lineal con un PIKAN; los PINN piezoeléctricos identificados usan MLP." Conservar las salvedades: no escribir "primer uso de KAN en electromecánica" (A1, A2); abrir A12; el vault afirma un "gap PIKAN en SAW" que A9 no respalda.
- **B (media-baja):** "No se encontraron trabajos con PIKAN/KAN ni PINN que reporten Cd, Cl y Δp del propio modelo contra los valores de FEATFLOW para DFG 2D-1 estacionario." (Shah, arXiv 2608.19632, reporta errores de campo contra OpenFOAM, no contra FEATFLOW; ver corrección en B2.)
- **C (baja-media): sin declaración de novedad** (retirada por el investigador el 2026-10-04: el mecanismo aún no está definido). Solo la lista de trabajos cercanos (C1–C10).
- **D (media):** lista de D1–D5; no citar D4.

## Cobertura que falta
Scopus, Web of Science, IEEE Xplore, ScienceDirect (texto completo), Google Scholar real, Semantic Scholar (429), citantes de la versión ICLR de KAN (otro ID en OpenAlex), CNKI/Wanfang, ASME/IEEE UFFC/IUS, SSRN/Research Square y lectura de PDF completos.

## Anexo — consultas ejecutadas

**WebSearch (42):** Kolmogorov-Arnold network piezoelectric physics-informed; KAN piezoelectric energy harvester Kolmogorov-Arnold; PIKAN electromechanical coupling piezoelectricity PDE; Kolmogorov-Arnold network surface acoustic wave; ...ferroelectric domain switching phase-field; Kolmogorov-Arnold flexoelectric physics-informed neural network; physics-informed neural network piezoelectric 2026 arXiv electro-elastodynamics PINN; Chebyshev KAN piezoelectric; "Kolmogorov-Arnold" "piezoelectric" (extended); preprints.org 202609.1914 physics-informed piezoelectric; Mouratidou physics-informed neural network piezoelectric Acta Mechanica; Georgiou ... arXiv 2608.07255; Mehtaj Banerjee "Scientific Machine Learning for Surface Acoustic Wave Devices" Sensors 2025 (y variantes de la revisión SAW); Shukla Di Leoni Blackshire Sparkman Karniadakis PINN ultrasound cracks; Mouratidou Drosopoulos Stavroulakis shunted piezoelectric (2); Ponomarev et al. piezoelectric equation; IPMC Kolmogorov-Arnold EAAI (4 variantes y la PII S0952197625001265); Semantic Scholar citations de KAN + piezo; cites KINN + piezo/multiphysics; FAIR comparison MLP vs KAN + piezo; arXiv physics-informed KAN multiphysics; KAN electroelastic/magneto-electro-elastic; residual-based adaptive refinement RBF network adaptive centers; PIKAN adaptive collocation RAD RAR grid refinement; RBF-PINN trainable centers widths node insertion; PINN cylinder Re=20 Schäfer Turek DFG drag lift (4 variantes, incluida una con Kolmogorov-Arnold); KAN "flow around a cylinder" steady NS; 压电 物理信息 KAN; redes Kolmogorov-Arnold informadas por la física piezoeléctrico; KAN physics-informed bimorph cantilever PIKAN; KAN electromechanical coupling MEMS resonator/transducer surrogate; Residual-adaptive PIKAN Leong Sim.
**arXiv search directo:** las 8 consultas piezo/KAN de A17 más: physics-informed piezoelectric; PIKAN; physics-informed Kolmogorov-Arnold (118 resultados, vistos 42); Kolmogorov-Arnold cylinder; ...Navier-Stokes; physics-informed cylinder Schafer Turek; PINN DFG benchmark cylinder; Kolmogorov-Arnold adaptive sampling / residual-based adaptive / radial basis / adaptive grid.
**OpenAlex:** las de A15–A18, B10–B11 y búsquedas por título (RA-PIKAN, J-PIKAN, Chebyshev PIKAN, Legend-KINN, PIKAN power system, KKANs, AC-PKAN, R3 y Nabian sin resultado, causalidad, DeepXDE, Free-RBF-KAN, KINN); registros por DOI de 10 artículos; citantes (filter=cites) de W4396821501, W4399447630|W4403579390 y W4404468356 (3 páginas).
**Crossref:** 10.1038/s44387-026-00084-4; 10.1002/aidi.202500214; 10.1088/2632-2153/ae8638; 10.1016/j.cma.2024.117518; 10.1007/s44379-024-00002-6 (404); 10.1016/j.cnsns.2025.109011 (otro artículo); 10.1016/j.cma.2024.117290; 10.1016/j.engappai.2025.110126.
**Aperturas directas (WebFetch):** ~35 páginas arXiv abs/HTML, PMC11902327, Aston, AICCOMAS 2025, Lancashire, imec, Vestnik UTMN, ASME OMAE, iMechanica. **Fallaron:** mdpi.com (2), sciencedirect.com, papers.ssrn.com, preprints.org, link.springer.com, api.semanticscholar.org, PDF de arXiv 2608.07255 / 2502.21253 / 2603.24013, SPIE.

## Inconsistencias del vault (sin corregir; requieren aprobación humana para modificar `PIKAN-Vault`)
1. `Shukla_2020_Piezoelectric_NDE.md`: título/contenido equivocados (placa metálica, no piezo; arXiv 2005.03596); `00_INDEX.md` lo presenta como "único PINN en SAW".
2. `Mehtaj_2025_SAW_Review.md`: título equivocado; la revisión no menciona KAN ni "cierra el gap"; falta la segunda revisión (*Sensors* 25(12):3588).
3. `domains/Piezoelectricidad_y_SAW.md`: "Zero papers PIKAN en SAW" y "la review señala el gap" sin sustento.
4. `Wang_2024_KINN.md`: autores incorrectos; es *CMAME* 433 (2025); URL vacía.
5. `Xiong_2026_JPIKAN.md`: DOI de otro artículo (correcto probable 10.1016/j.cnsns.2025.109414); año 2025.
6. `Zhang_2026_LegendKINN.md`: *Expert Syst. Appl.* 2025, DOI 10.1016/j.eswa.2025.129839.
7. `Heravifard_2024_HWF_PIKAN.md`: es arXiv 2512.12001 (dic-2025); el título habla de Boltzmann sin colisiones.
8. `Anagnostopoulos_2025_RGA_KAN.md`: autores/venue incorrectos (Rigas et al., "Training Deep Physics-Informed KANs", arXiv 2510.23501, *CMAME* 452, 2026).
9. `Toscano_2025_KKANs.md`: título real "KKANs: Kurkova-Kolmogorov-Arnold Networks and Their Learning Dynamics" (arXiv 2412.16738); DOI pendiente.
10. `Toscano_2025_PIKANsReview.md`: autores incompletos; DOI 404 en Crossref; verificada arXiv 2410.13228.
11. `Wang_2024_CausalPINNs.md`: la nota da SISC 10.1137/22M1514275; OpenAlex da *CMAME* 2024, 10.1016/j.cma.2024.116813.
12. `Guo_2025_cPIKAN_Fluids.md`: DOI 10.1063/5.0284999; autores y título completos.
13. `Shuai_2025_PowerSystems.md`: venue *IEEE Open Access J. Power Energy* 2025, DOI 10.1109/oajpe.2025.3529928 (arXiv 2408.06650).
14. `AC_PKAN_2025.md` y `Wang_2024_PirateNet.md`: no verificados.
15. `frontiers/2026-08-20_Barrido_Marco_Teorico_Foco_Adaptativo.md`: enlaza a ~12 notas inexistentes (Dorfler 1996, Verfürth 2013, Jagtap 2020, McClenny 2023, Moseley 2023, Cuomo 2022, Dzimah 2026, He 2026, Jnini 2024, Chen 2025, Duanmu 2026, Georgiou 2026, Botvinick-Greenhouse 2025); su conclusión de hueco frente a Rigas 2024 queda debilitada por C1 y C3; "Mouratidou 2026" frente a "2025" (la fecha en línea es 2026-02-05); "WCCM-ECCOMAS 2026 piezoelasticidad con capas cuánticas" no verificado.
16. `Chiu_2026_Free_RBF_KAN.md` y `Comparadores_RBF_Adaptividad_2026.md`: coherentes con arXiv 2601.07760; omisión del informe LLNL/OSTI 10.2172/3389234.
17. `Wu_2022_Adaptive_Sampling_PINNs.md` (contenido CMAME 2023) y `Liu_2025_KAN.md` (ICLR 2025; arXiv 2024): cosméticos; decidir la convención de años antes de citar.
18. `PDE_Contracts_Burgers_AllenCahn_DFG.md`: los valores FEATFLOW coinciden con lo visto; el informe 411 de Turek y NNFM 48/52 no se verificaron.
