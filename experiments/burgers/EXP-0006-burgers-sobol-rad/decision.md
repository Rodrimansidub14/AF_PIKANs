---
experiment_id: EXP-0006
decision: rejected
decided_at: 2026-10-04
decided_by: Rodrigo Mansilla (chat, 2026-10-04: "apruebo" / "cierro" / "aprobados")
supersedes:
---

# EDR — EXP-0006: Burgers: RAD contra Sobol (resultado negativo)

Cierre **de orden**: redactado por un agente y aprobado por el investigador. No cambia ningún número ni es un claim; los baselines vigentes son `evals/baselines/baseline_v1_exploratory.*` (2026-10-03) y `baseline_v2_exploratory.*` (2026-10-04).

- **Estado real:** pre-registro preparado pero **nunca firmado**; sus criterios (razón geométrica ≤ 0.8) eran "propuestos, NO activos". Se ejecutó bajo autorización exploratoria.
- **Resultado:** RAD sobre el RBF de esa campaña salió ~5.9× peor que Sobol (IC95 2.8–12.1; 5/5 parejas), con una base no convergida y un diseño frágil. Resultado negativo exploratorio, sin criterio activado.
- **Etiqueta:** cerrado como resultado negativo. Coincide con el patrón de EXP-0011 y EXP-0012 (RAD no ayuda sobre estos controles). El baseline vigente de Burgers es el RBF48 de v2.
