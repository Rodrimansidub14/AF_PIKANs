---
type: index
tags: #index #pikan #thesis
updated: 2026-09-23
---

# PIKAN Research Vault

> Vault de investigación para tesis sobre Physics-Informed Kolmogorov-Arnold Networks.
> Cada nodo es un archivo. Los `[[wikilinks]]` construyen el graph.

---

## Capas del campo

| Capa | Qué significa | Tag |
|------|--------------|-----|
| `FOUND` | Fundamento teórico — leer antes de todo | `#found` |
| `ARCH` | Contribución arquitectural — variantes y mecanismos | `#arch` |
| `APP` | Aplicación nueva — mismo método, dominio inexplorado | `#app` |
| `FUTURE` | Frontera abierta — gap no resuelto | `#future` |

---

## 0. Fundamentos

- [[Raissi_2019_PINNs]] — el paper que abre el campo
- [[Karniadakis_2021_NatureReviews]] — survey SciML
- [[Lu_2021_DeepXDE]] — framework principal
- [[Grossmann_2024_FEMvsPINN]] — benchmark honesto FEM vs PINN

---

## 1. Arquitectura Core

- [[Liu_2025_KAN]] — KAN fundacional. B-splines en aristas.
- [[Toscano_2025_PIKANsReview]] — review del campo. Leer antes de elegir variante.
- [[Shukla_2024_cPIKAN_Benchmark]] — comparación MLP/KAN publicada; registrar DOI y límites por problema
- [[Wang_2024_KINN]] — comparación más sistemática por dominio

---

## 2. Variantes Arquitecturales

### Polinomios ortogonales
- [[Guo_2025_cPIKAN_Fluids]] — cPIKAN fluidos. Baseline de facto.
- [[Xiong_2026_JPIKAN]] — Jacobi. 50% menos parámetros.
- [[Zhang_2026_LegendKINN]] — Legendre. 1-3 órdenes más rápido.

### Escalado dimensional
- [[Jacob_2025_SPIKANs]] — separable. 287× speedup. Para 3D+.

### Resolución múltiple
- [[Heravifard_2024_HWF_PIKAN]] — wavelet + Fourier. Para discontinuidades.

### Profundidad y estabilidad
- [[Anagnostopoulos_2025_RGA_KAN]] — residual-gated. Solución a KANs profundos.
- [[Wang_2024_PirateNet]] — precedente de RGA

### Atención y acoplamiento
- [[AC_PKAN_2025]] — atención + cPIKAN. Full-rank Jacobian.

### Aproximación universal
- [[Toscano_2025_KKANs]] — base-agnostic. Extracción simbólica.

---

## 3. Aplicaciones Existentes

- [[Shuai_2025_PowerSystems]] — sistemas eléctricos. Inverso.
- [[Shukla_2020_Piezoelectric_NDE]] — NDE piezoeléctrico. Único PINN en SAW.
- [[Mehtaj_2025_SAW_Review]] — review SAW + señal de gap explícita

---

## 4. Neural Operators (contexto)

- [[Li_2021_FNO]] — FNO. ~1000× speedup. Necesita datos.
- [[Lu_2021_DeepONet]] — DeepONet. Flexible en geometría.
- [[Kovachki_2023_NeuralOperator]] — unificación teórica del área
- [[Rahman_2023_LatentNeuralOperator]] — encoder → latent → decoder

---

## 5. Fronteras Arquitecturales (ideas de tesis)

- [[Adaptive_Multi-Resolution_Spline_Grids]] — AMR interno en KAN
- [[Causal_Physics-Informed_Attention]] — máscara de atención gobernada por CFL
- [[Latent_Space_Neural_Operators_KAN_edition]] — encoder-KANO-decoder
- [[Continual_Learning_Sistemático_en_PIKANs]] — el gap más subestimado

### Papers que soportan estas ideas
- [[Wang_2024_CausalPINNs]] — causalidad en la pérdida (precedente)
- [[Vaswani_2017_Attention]] — mecanismo de atención base
- [[Kharazmi_2021_hpVPINNs]] — AMR dentro de física-informada
- [[Kirkpatrick_2017_EWC]] — EWC / catastrophic forgetting (lo que KAN dice no necesitar)

---

## 6. Labs Grandes / Panorama

- [[Lam_2023_GraphCast]] — DeepMind. Weather forecasting.
- [[Sanchez_2020_GNS]] — DeepMind. Learned simulators.
- [[Jumper_2021_AlphaFold2]] — DeepMind. El modelo de éxito que todo el campo cita.
- [[Yang_2024_KAT]] — KAN a escala de transformers. Cuello de botella en GPU.

## 7. Benchmarks y muestreo para AF_PIKANs

- [[Takamoto_2022_PDEBench]] — referencia de protocolo de benchmark SciML.
- [[Wei_2024_PDENNEval]] — comparación de métodos de redes para PDE.
- [[Hao_2024_PINNacle]] — benchmark amplio de PINNs, incluye calor y fluidos.
- [[Wu_2022_Adaptive_Sampling_PINNs]] — RAD/RAR-D; artículo CMAME 2023 y versión arXiv 2022.
- [[Chiu_2026_Free_RBF_KAN]] — RBF-KAN adaptable, candidato distinto de RBF fija.
- [[Allen_Cahn_EXP-0003_References_2026-09-23]] — fuentes del contrato 1D, señales de energía y límites de la comparación PIKAN.

## 8. Fuentes de visualización y contratos PDE

- [[Visualization_Toolchain_References_2026-09-23]] — documentación oficial de PyVista, Plotly, Panel, VisPy, Taichi, Makie, napari y WebGPU.
- [[PDE_Contracts_Burgers_AllenCahn_DFG]] — contratos Burgers/Allen–Cahn/DFG; FEATFLOW 2D-1 y Schäfer–Turek 1996, con el informe 3D Turek 2010 distinguido.

---

## Dominios de aplicación

- [[Piezoelectricidad_y_SAW]] — candidato principal de tesis
- [[Neurociencia_Computacional]]
- [[Semiconductores_y_Quantum]]
- [[Fluidos_y_Mecánica_Estructural]]

---

## Conceptos transversales

- [[Spectral_Bias]] — por qué MLP falla en frecuencias altas
- [[Inverse_Problems]] — donde PIKANs documentadamente ganan
- [[Continual_Learning]] — el claim más citado y menos validado de KAN
- [[Grid_Extension]] — mecanismo único de KAN
- [[Symbolic_Regression]] — extracción de fórmulas de modelos entrenados
- [[Adaptive_Collocation]] — densificación de puntos de colocación
- [[Multiscale_Physics]] — predictor más fuerte de ventaja PIKAN
- [[Causal_Masking_and_Attention]] — imponer física en la arquitectura de atención
- [[Latent_Space_Operators]] — separar complejidad espacial de temporal

---

## Estado del proyecto

| Capa | Estado |
|------|--------|
| L1 — SOTA y dirección del campo | ✅ Completado |
| L2 — Gap analysis por dominio | 🔄 En progreso |
| L3 — Implementación PINN base (DeepXDE) | ⬜ Pendiente |
| L4 — Conversión a PIKAN | ⬜ Pendiente |
| L5 — Contribución arquitectural | ⬜ Pendiente |

## Bibliografía compilada para el brief de avance

- `sources/AF_PIKANs_Research_Brief_References_2026-09-23.bib` — fuentes científicas, benchmarks y documentación de herramientas citadas en el brief, listas para reutilizar en LaTeX/Zotero.
