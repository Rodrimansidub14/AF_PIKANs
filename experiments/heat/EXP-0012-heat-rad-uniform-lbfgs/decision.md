---
experiment_id: EXP-0012
decision: rejected        # accepted | rejected | inconclusive | deferred
decided_at: 2026-10-04
decided_by: Rodrigo Mansilla (chat, 2026-10-04: "apruebo" / "cierro" / "aprobados")
supersedes:
---

# EDR — EXP-0012: ¿el déficit tras L-BFGS venía de los puntos de L-BFGS?

**Decisión: rechazada por la regla pre-registrada (media geométrica 2.025 > 2.0; 2/5 semillas ≤ 2.0). El investigador cierra el hilo RAD en Heat (opción A) y no abre EXP-0013.**

## Patrón registrado explícitamente

> **En el control RBF de Heat, RAD mejora una etapa y empeora la otra.** Reduce el error al final de Adam ~11× (razón 0.088, 5/5 semillas) y deja un estado que L-BFGS completo no refina al piso del control: ~4.9× peor si L-BFGS corre sobre los puntos RAD, y ~2.0× peor (2.025) si corre sobre el pool uniforme. Corregir los puntos de L-BFGS recupera ~2.5× pero no cierra la brecha.

Límites del patrón (declarados): n = 5 semillas, un problema (Heat), una configuración de RAD; en las 5 corridas L-BFGS terminó por tope (2 de iteraciones, 3 de tiempo), así que no se probó si más iteraciones lo cierran. Eso no se hizo ni se hará dentro de este hilo.

## Consecuencias para el diseño del primer mecanismo (reglas, no resultados)
1. Un mecanismo se evalúa **en las dos etapas** (fin de Adam y tras L-BFGS); mejorar solo Adam no basta. La guardia de L-BFGS es vinculante.
2. L-BFGS se evalúa sobre un pool **fijo y declarado** de antemano (no sobre los puntos que el mecanismo concentra).
3. El mecanismo se pre-registra con esa guardia ya incorporada.

Resultados: `results.md`. Nada de esto es un claim.
