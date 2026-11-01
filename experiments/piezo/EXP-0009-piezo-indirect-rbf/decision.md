---
experiment_id: EXP-0009
decision: closed_baseline_recorded
decided_at: 2026-10-04
decided_by: Rodrigo Mansilla (chat, 2026-10-04: "apruebo" / "cierro" / "aprobados")
supersedes:
---

# EDR — EXP-0009: Piezo indirecto: problema, referencia y control

Cierre **de orden**: redactado por un agente y aprobado por el investigador. No cambia ningún número ni es un claim; los baselines vigentes son `evals/baselines/baseline_v1_exploratory.*` (2026-10-03) y `baseline_v2_exploratory.*` (2026-10-04).

- **Estado real:** spec en borrador; el control está registrado en v2: receta de Daniel, n = 5, RBF16 u/v/φ 0.092/0.104/0.048 contra MLP 0.136/0.186/0.046; RBF gana 5/5 pareado en u y v, 2.3× el costo, no convergido.
- **Etiqueta:** cerrado como baseline. La reimplementación propia (`piezo_pinn.py`) queda como diagnóstico negativo.
- **Abierto (no es un cierre):** el FEM es del mismo grupo (sin contraste de terceros) y la licencia de Daniel no está confirmada. El mecanismo en piezo irá en un EXP nuevo, después de DFG.
