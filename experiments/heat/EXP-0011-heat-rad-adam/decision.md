---
experiment_id: EXP-0011
decision: rejected        # accepted | rejected | inconclusive | deferred
decided_at: 2026-10-04
decided_by: Rodrigo Mansilla (chat, 2026-10-04: "apruebo" / "cierro" / "aprobados")
supersedes:
---

# EDR — EXP-0011: RAD sobre el control RBF de Heat

**Decisión: rechazada, por la regla pre-registrada y firmada el 2026-10-03.** Aprobada por el investigador.

- **Criterio primario cumplido:** razón RAD/control al final de Adam, media geométrica 0.088 (umbral ≤ 0.80), 5/5 semillas ≤ 0.80.
- **Guardia violada:** error tras L-BFGS completo 4.9× el del control (límite 2.0×). La condición de falsación (c) rechaza la hipótesis.
- **Alcance:** se rechaza H-RAD-HEAT-1 con esta configuración (RAD potencia 2, 50 % uniforme, refresco 1 000), en este control, con n = 5. No es un resultado general sobre RAD.
- **Seguimiento:** EXP-0012 (cerrado, ver su decisión). Patrón resultante en `experiments/INDEX.md`.
- Resultados: `results.md`. Nada de esto es un claim (gate `scientific_claim`).
