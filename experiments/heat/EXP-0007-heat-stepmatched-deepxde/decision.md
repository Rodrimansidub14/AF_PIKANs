---
experiment_id: EXP-0007
decision: closed_exploratory_record
decided_at: 2026-10-04
decided_by: Rodrigo Mansilla (chat, 2026-10-04: "apruebo" / "cierro" / "aprobados")
supersedes:
---

# EDR — EXP-0007: Heat step-matched (RBF vs B-spline) con presupuesto DeepXDE

Cierre **de orden**: redactado por un agente y aprobado por el investigador. No cambia ningún número ni es un claim; los baselines vigentes son `evals/baselines/baseline_v1_exploratory.*` (2026-10-03) y `baseline_v2_exploratory.*` (2026-10-04).

- **Estado real:** contrato `PENDING_HUMAN_SIGNATURE`, nunca firmado; se ejecutó bajo autorización exploratoria (campaña S2 y seguimiento).
- **Etiqueta:** registro exploratorio. El L-BFGS archivado de S2 (2.06e-6) estaba truncado por un defecto del arnés y **queda supersedido** por el L-BFGS completo (1.857e-6 ± 1.8e-7, `heat-lbfgs-redo-20260930`), que es lo que figura en v1. El B-spline de esta campaña no entra al baseline.
