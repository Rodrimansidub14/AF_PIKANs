---
type: frontier
tags: #frontier #arch #open-gap #thesis-direction
updated: 2026-08-24
status: proposed
promotion: human_approved
source_role: literature_scout
note: "Nota promovida al vault canónico con aprobación humana explícita."
---

# Barrido bibliográfico — Marco teórico reestructurado (2026-08-20)

> Origen: hilo de research automation orientado por
> `propuesta_reestructuracion_marco_teorico.md` (externa, no versionada).
> Objetivo del barrido: SOTA + referencias primarias/seminales para las
> secciones 6.1–6.7 del capítulo, sin modificar `i-marco_teorico.tex` ni
> `m-bibliografia.bib`.

---

## Resumen ejecutivo (síntesis, agregado 2026-08-24)

- **Cobertura**: alta en 6.1, 6.2, 6.3, 6.4 y 6.7; media en 6.5 (asignación
  de capacidad — probable aporte propio, no vacío bibliográfico) y 6.6
  (mecanismo — ver decisión abajo).
- **23 referencias base** verificadas en el primer barrido + **14 nuevas**
  en el segundo (6 canónicos faltantes en `m-bibliografia.bib`, 6 candidatos
  2026 en su mayoría sin arbitrar, 2 hallazgos de adyacencia).
- **Decisión de diseño recomendada**: inyectar `T` sobre la **densidad de
  grilla interna de la spline (PIKAN)**, no sobre reponderación del residual
  ni modulación del muestreo — ambos ya saturados en literatura reciente
  (SA-PINN, ReLoBRaLo, RAR/RAD/RAR-D, FI-PINN, PINNACLE, Chen 2025). Esta
  opción es la única que explota el soporte local de la base spline (la
  propiedad que justifica PIKAN sobre MLP). El precedente más cercano
  (Rigas 2024, grid-dependent PIKAN) usa cronograma de entrenamiento, no
  señal derivada del residual — ese contraste es el hueco de contribución.
  Diferenciar también de AB-PINNs (macro/subdominios) frente al enfoque
  propuesto (micro/nodos de control).
- **No invalida** la pregunta de investigación; **sí invalida** la
  formulación actual del mecanismo en `i-marco_teorico.tex` (línea 112,
  reponderación directa del residual).
- **Pendiente crítico derivado**: el análisis de costo por época
  (`\pendiente` líneas 128–130) pasa de opcional a crítico si se adopta la
  ruta de grilla adaptativa, por riesgo de inestabilidad numérica en
  remallado dinámico.
- **Estado**: `i-marco_teorico.tex` y `m-bibliografia.bib` sin modificar.
  Esta nota fue promovida a `PIKAN-Vault/PIKAN-Vault/frontiers/` con
  aprobación humana explícita; conserva el carácter propositivo del hueco
  arquitectural y no convierte la síntesis en evidencia experimental.

---

## Cadena conceptual validada

```
PDE → aproximación numérica → SciML → SOTA → limitación →
firma matemática → mecanismo → stress tests
```

La literatura revisada sostiene esta cadena. El punto de fricción real no
está en la cadena sino en el mecanismo (ver sección "Decisión de diseño"
abajo).

---

## 0. Fundamentos (ya cubiertos en `m-bibliografia.bib`)

- [[Raissi_2019_PINNs]] — residual, collocation, RAR.
- [[Karniadakis_2021_NatureReviews]] — survey SciML/PIML.
- [[Liu_2025_KAN]] — KAN fundacional.
- [[Kharazmi_2021_hpVPINNs]] — descomposición de dominio + refinamiento variacional.
- [[Grossmann_2024_FEMvsPINN]] — comparación honesta PINN vs FEM.

## 1. Canónicos faltantes identificados en este barrido

- [[Dorfler_1996_AdaptiveFEM]] — DOI `10.1137/0733054`. Marcado adaptativo
  clásico. Resuelve el `\pendiente` de línea 36 de `i-marco_teorico.tex`.
- [[Verfurth_2013_AposterioriErrorBook]] — OUP, ISBN 9780199679423. Libro de
  estimadores a posteriori, mismo pendiente.
- [[Jagtap_2020_AdaptiveActivations]] — DOI `10.1016/j.jcp.2019.109136`.
- [[McClenny_2023_SelfAdaptivePINN]] — DOI `10.1016/j.jcp.2022.111722`. SA-PINN,
  arbitrado (J. Comput. Phys. 474).
- [[Moseley_2023_FBPINNs]] — DOI `10.1007/s10444-023-10065-9`. Contraargumento
  directo a la sección de asignación de capacidad (6.5).
- [[Cuomo_2022_SciMLReview]] — DOI `10.1007/s10915-022-01939-z`. Review
  complementario a Karniadakis 2021.

## 2. Candidatos 2026 (heterogéneos, mayoría sin arbitraje)

- [[Dzimah_2026_PINNKANBenchmark]] — arXiv 2602.15068. Benchmark controlado
  MLP-PINN vs PIKAN a igual presupuesto de parámetros. **No arbitrado.**
- [[He_2026_SpectralPreconditioners]] — arXiv 2603.00904. Extiende sesgo
  espectral con preconditioners operator-aware. **No arbitrado.** Precursor
  arbitrado: [[Jnini_2024_PreconditioningPINN]] (arXiv 2402.00531).
- [[Chen_2025_AdaptiveWeightSampling]] — arXiv 2511.05452. Método híbrido
  muestreo+pesos — **amenaza directa** al mecanismo propuesto si éste termina
  siendo reponderación del residual. **No arbitrado.**
- [[Duanmu_2026_KANGraphAttention]] — DOI `10.1016/j.engappai.2026.115798`.
  Arbitrado. Variante PIKAN+GNN, evidencia de heterogeneidad arquitectural.
- [[Georgiou_2026_PiezoFreqResponsePINN]] — arXiv 2608.07255. Complementa
  Tandis 2025 y Mouratidou 2026 en el caso piezoeléctrico. **No arbitrado.**
- WCCM-ECCOMAS 2026, *Hybrid PINN con capas cuánticas para piezoelasticidad*
  — solo resumen de congreso, sin DOI ni autoría confirmada. **Tratar como
  nota de tendencia, no como evidencia primaria.**

## 3. Upgrade de estado editorial

- [[BotvinickGreenhouse_2025_ABPINNs]] — antes preprint (arXiv 2510.08924),
  ahora arbitrado: *Mach. Learn.: Sci. Technol.*, DOI `10.1088/2632-2153/ae8638`.
  Sube de evidencia secundaria a primaria. Es el precedente más cercano a
  "redistribuir capacidad" pero opera a nivel de subdominios (macro), no de
  densidad de nodos de control dentro de una sola red (micro).

---

## Decisión de diseño pendiente: punto de inyección del mecanismo

`i-marco_teorico.tex` (línea 112) deja explícitamente pendiente si `T` se
inyecta sobre:

1. **Reponderación del residual** (`w = T(s)` aplicado a la pérdida, como
   está escrito hoy) — territorio saturado: [[McClenny_2023_SelfAdaptivePINN]],
   ReLoBRaLo, [[Chen_2025_AdaptiveWeightSampling]].
2. **Modulación del muestreo** — también saturado: RAR, RAD/RAR-D, FI-PINN,
   PINNACLE.
3. **Modulación de la densidad de la grilla interna de la spline (PIKAN)** —
   el único punto de inyección que usa la propiedad que justifica elegir
   PIKAN sobre MLP (soporte local). Precedente más cercano: adaptive-grid
   PIKAN training (Rigas et al. 2024, arXiv 2407.17611) — pero disparado por
   cronograma de entrenamiento, no por una señal derivada del estado del
   modelo/residual. Ahí está el hueco de contribución real.

**Recomendación de literatura**: opción 3. Diferenciación limpia frente a
Rigas 2024 (señal-driven vs cronograma) y frente a
[[BotvinickGreenhouse_2025_ABPINNs]] (micro vs macro). Costo: mayor
dificultad de implementación y posible inestabilidad numérica en remallado
dinámico — el `\pendiente` de "Costo" (línea 128-130 de `i-marco_teorico.tex`)
se vuelve crítico bajo esta ruta, no opcional.

---

## Enlaces sugeridos para actualizar el índice del vault (`00_INDEX.md`)

Candidatos de inserción y enlace después de esta promoción:

- Sección **0. Fundamentos**: agregar Dörfler 1996 y Verfürth 2013 como
  precedente numérico clásico (actualmente el índice no tiene capa de
  numérica clásica adaptativa).
- Sección **2. Variantes Arquitecturales → Resolución múltiple**: agregar
  Rigas 2024 (grid-dependent PIKAN) si no está ya vinculado bajo otro nombre.
- Sección **5. Fronteras Arquitecturales**: el ítem
  `[[Adaptive_Multi-Resolution_Spline_Grids]]` ya existente es exactamente
  el concepto que este barrido identifica como el hueco de contribución —
  vincularlo explícitamente con la decisión de diseño de arriba.
- Sección **Conceptos transversales**: `[[Adaptive_Collocation]]` ya cubre
  la familia de muestreo; falta un concepto equivalente para "densidad de
  grilla adaptativa dentro de KAN" — podría crearse
  `[[Adaptive_Spline_Grid_Density]]` como nodo nuevo.

---

## Estado de verificación de metadatos

Todas las referencias de este documento fueron verificadas contra Crossref
y/o arXiv API (título, autores, fecha, DOI) durante el barrido. Las
marcadas "no arbitrado" son preprints de arXiv sin versión de journal
confirmada a la fecha de este barrido.
