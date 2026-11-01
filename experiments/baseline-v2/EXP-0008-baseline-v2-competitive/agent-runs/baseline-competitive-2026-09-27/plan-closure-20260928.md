# Plan de cierre del baseline competitivo — 2026-09-28

Planificado por: Claude Opus 5.5. Ejecutor: Codex `gpt-6-luna`, reasoning effort `max`.
Estado de partida: [competitive-baseline-execution.md](competitive-baseline-execution.md), [scorecard-current.csv](scorecard-current.csv).

## Restricciones duras

- Cómputo **solo** en Kaggle T4×2. Nada de entrenamientos, notebooks, tests ni validadores locales. Localmente solo: editar notebooks/manifiestos, subir con Kaggle CLI, descargar y agregar artefactos (lectura/CSV).
- Cuota: 15.90 h disponibles (último registro). Reserva ≥ 12.00 h. **Presupuesto de este plan: 3.50 h de sesión** (margen 0.40 h). Refrescar la cuota con la CLI antes de cada lanzamiento; si `disponible − cap_de_la_sesión < 12.0 h`, no lanzar.
- Todo kernel lleva cap remoto de tiempo explícito (el que se indica abajo) y escribe checkpoint/JSON parcial antes de agotarse.
- `evals/` y `registry/claims.jsonl`: solo lectura. No mover umbrales, grids ni graders. Si un resultado lo exige, detenerse y pedir gate `eval_integrity`.
- `scientific_accepted=false` siempre. El agente no acepta referencias ni claims; solo reúne evidencia para el gate humano.
- Sin credenciales Kaggle disponibles → detenerse y reportar; no pedir tokens.

## Diagnóstico: qué impide llamar "comparable a literatura" a cada problema

| Problema | Bloqueo | Es resoluble con GPU en ≤3.5 h? |
|---|---|---|
| Burgers | Ninguno para PINNacle Adam→L-BFGS (0.013264 vs 0.0133). vRBA no reproducido. | Cerrado como *comparable a PINNacle*. vRBA queda fuera de presupuesto. |
| Heat | Contrato 500 s vs meta 1M pasos → `incomplete_validation`; comparación con DeepXDE no es isocosto. | Sí: corrida con presupuesto por pasos igual a la receta DeepXDE α=0.1. |
| Allen–Cahn | Referencia CUDA FD/RK4 no aceptada; AEH-PINN (8.12e-5) no reproducido. | Evidencia de referencia: sí (barato). Reproducir AEH-PINN (100k épocas × 10): no. Refinamiento L-BFGS de RBF: sí. |
| DFG Re=20 | Sin campo FEM convergido; Cd de la cohorte 301–305 a 8.2% tras L-BFGS (corregido 2026-09-30; el 73.8% usaba una fórmula de fuerza errónea). | Campo FEM Taylor–Hood: sí. Puntuar cohortes existentes contra él: sí. |

## Sesiones (orden de ejecución)

### S1 — Referencias independientes (cap 30 min, T4×2)

Un kernel, dos partes independientes; cada una escribe su JSON aunque la otra falle.

**S1a Allen–Cahn.** Solver independiente del FD/RK4 existente: pseudo-espectral Fourier/Chebyshev + ETDRK4 (Kassam–Trefethen) en float64, con la misma PDE/IC/BC del contrato. Reportar, en la malla de evaluación del contrato:
- ‖u_FD8192 − u_ETDRK4‖/‖u_ETDRK4‖ (relativo L2 espacio-tiempo ponderado) y máximo absoluto;
- diferencias contra `.mat` de DeepXDE interpolado a su malla 201×101;
- convergencia del propio ETDRK4 (N y Δt duplicados).
Criterio de *evidencia* (no aceptación): discrepancia FD vs ETDRK4 ≤ 1e-6 relativo, lo que haría despreciable el error de referencia frente a 1.65e-2. Re-puntuar los 10 checkpoints existentes (RBF y MLP) contra ETDRK4 como segunda referencia.

**S1b DFG 2D-1.** FEM estacionario Taylor–Hood P2/P1 (scikit-fem o FEniCSx vía pip/conda en el contenedor), Newton, outlet do-nothing, 3 niveles de malla con refinamiento cerca del cilindro. Criterio de evidencia: Cd, Cl, Δp del nivel fino dentro de 0.5% / 0.0005 abs / 0.5% de FEATFLOW y cambio entre los dos niveles más finos < 0.2%. Exportar u, v, p en una malla de evaluación fija (declararla en el manifiesto antes de puntuar; excluir el interior del cilindro). Puntuar field relative L2 (u, v, p por separado; p con constante gauge fijada por el outlet) de los checkpoints 301–305 (post-L-BFGS) y 501–505 (RBF y Tanh-MLP), sin reentrenar.

### S2 — Heat step-matched (cap 75 min, T4×2)

Contrato v5 nuevo en `experiments/` (v4 no se reescribe; el pre-registro lo firma el humano, registrar como pendiente de firma). RBF, B-spline y FNN, mismos 5 seeds y pools Sobol de v4, presupuesto por **pasos** igual a la receta DeepXDE α=0.1 de referencia (20 000 Adam + L-BFGS con los mismos criterios de parada), float64, mismo evaluador 512×256. Estado de completitud = pasos alcanzados, no tiempo. Si la estimación de tiempo por brazo × 15 excede 65 min, reducir a RBF + B-spline antes de lanzar (no a mitad). Reportar diferencias pareadas por seed contra DeepXDE α=0.1 (mismas seeds si el artefacto lo permite; si no, declarar no pareado).

### S3 — Allen–Cahn L-BFGS sobre checkpoints RBF existentes (cap 45 min, T4×2)

Solo si S1a termina. Continuación L-BFGS de los 5 checkpoints RBF-Sobol (y MLP como control, mismo presupuesto) con pools guardados, verificación de hashes antes de continuar, 240 s por brazo-seed. Puntuar contra FD8192 y ETDRK4. No es reproducción de AEH-PINN y así se reporta.

### Reserva: 60 min

Solo para relanzar una sesión que falle por infraestructura (no por resultado). Nada nuevo.

Total máximo: 30 + 75 + 45 + 60 = 210 min = 3.50 h.

## Salidas obligatorias

1. Artefactos descargados en `kaggle-outputs/<kernel>/…` con manifiesto (hashes, versiones, cap, duración real).
2. `scorecard-current.csv` ampliado (no borrar filas; arreglar las filas DFG de la cola que perdieron salto de línea).
3. `competitive-baseline-execution.md` actualizado: por problema, veredicto uno de `comparable_to_literature` / `screening_only` / `not_comparable`, con el contrato que coincide y el que no.
4. `closure-gate-evidence.md`: tabla para el gate humano `scientific_claim` con la evidencia de referencia (S1) y lo que falta.
5. Cuota final leída por CLI y horas consumidas por sesión.

## Qué NO significa terminar este plan

No hay SOTA en los cuatro problemas: Allen–Cahn está ~200× sobre AEH-PINN y Burgers no se comparó con vRBA. El resultado esperado honesto es: Burgers comparable a PINNacle; Heat comparable a DeepXDE bajo presupuesto por pasos (si S2 lo muestra); Allen–Cahn y DFG con referencias mejor soportadas pero sin comparador SOTA reproducido.
