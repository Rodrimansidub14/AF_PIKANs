# EXP-0023 — prueba corta en T4 (pruebas previas 6 y 7), 2026-10-20

Kernel `rodrimandub/af-pikans-exp-0023-shorttest` (T4×2, 1 127 s de sesión, semilla 1). Resultados en `results/shorttest-gpu/exp0023_shorttest/` (`report.json`). Autorizada en el chat («Autorizo una prueba corta…, sin lanzar el bloque T1»). No se lanzó el bloque. Criterios fijados antes de correr (docstring de `kaggle/shorttest/exp0023_shorttest_gpu.py`).

## Prueba 7a — equivalencia CPU/GPU de la reparametrización (estado compartido, float64, N = 512): **cumple**
Diferencia relativa máxima de P, de c' y de la salida del modelo entre CPU y T4 (criterio < 1e-6):

| | J | E | I |
|---|---|---|---|
| Burgers | 2.8e-16 | 1.5e-12 | 5.1e-11 |
| Allen–Cahn | 2.1e-16 | 1.6e-12 | 2.3e-11 |

## Prueba 7b — corridas cortas (Adam 100, L-BFGS 20, recálculo cada 50, N = 512, float32): **NO cumple**
| | B (criterio 5 % / 10 %) | E, I (criterio 20 % / 20 %) |
|---|---|---|
| Burgers | GPU/CPU: 0.0 % y 0.11 % — cumple | E: GPU `nonfinite_loss`; CPU llega con pérdida 85.8 y error 3.39 → 0.60. I: no finito en ambos |
| Allen–Cahn | 0.0 % y 0.01 % — cumple | E: GPU pérdida 57 034 (L-BFGS rechazado), CPU pérdida 1.19, error 0.665 vs 0.707 (5.9 %) y 0.665 vs 0.767 (13.3 %). I: no finito en CPU; en GPU L-BFGS lanza excepción y termina en NaN |

**Lectura: no es un problema CPU/GPU; los brazos precondicionados divergen o se vuelven inestables en los primeros 100 pasos de Adam, en los dos dispositivos.** B reproduce CPU↔GPU a ≤ 0.11 %.

Causa probable (medida en el estado de 7a, no probada causalmente): P = (B + εI)^(-1/2) no está normalizada. Valores singulares de P: 0.2–3.4 e3 (J, E, I; capa 1 hasta 3.4e3; capas 0 y 2 hasta 1.5e2 y 71). Adam da pasos de tamaño ≈ lr = 1e-3 en c' sin importar la curvatura, así que en c el paso es hasta ≈ lr · σ_max(P) ≈ 3 en las direcciones mal condicionadas, 3 órdenes de magnitud sobre el paso del baseline. J (diagonal) tiene la misma escala, así que no es un asunto propio de los bloques completos. El brazo J no se corrió en 7b.

## Prueba 6 — presupuesto en T4 (float32, 200 pasos, 3 repeticiones; N = 2 048)
Medido con 4 procesos simultáneos (2 en GPU + 2 CPU), así que los s/paso son algo mayores que los 0.039/0.021 aislados del T1.

| s por paso de Adam | B | J | E | I | F |
|---|---|---|---|---|---|
| Burgers | 0.054 | 0.055 | 0.059 | 0.056 | 0.051 |
| Allen–Cahn | 0.060 | 0.058 | 0.058 | 0.053 | 0.050 |

| s por recálculo (N = 2 048) | J | E | I |
|---|---|---|---|
| Burgers | 3.3 | 3.7–3.9 | 5.1–5.2 |
| Allen–Cahn | 1.8–2.7 | 2.2–2.8 | 2.3–3.5 |

- Sobrecosto por paso de la reparametrización: dentro del ruido (≤ ≈ 9 %).
- Un recálculo ≈ 60–95 pasos de Adam (Burgers, E/I). Control completo: 21 recálculos Burgers y 41 AC → ≈ 80–110 s y ≈ 90–140 s por corrida; convención T1 (3 y 5 recálculos) ≈ 11–16 s y ≈ 11–17 s. Menos del ≈ 10 % de una corrida en ambos casos.
- La corrida completa por problema (prueba 6 literal) no se hizo: no estaba autorizada y los brazos precondicionados divergen con la receta actual. El bloque de 50 corridas sigue en el orden de ≈ 2 h (convención T1) o ≈ 20 h (corridas completas), proyectado.

---

# Repetición tras la Enmienda 1 (P normalizado), 2026-10-21

Kernel `rodrimandub/af-pikans-exp-0023-shorttest` v2 (T4×2 + CPU, 1 232 s). Resultados: `results/shorttest-gpu-amend1/exp0023_shorttest/report.json`. Verificación previa del rango de P: `results/verify_sv.json` (máximo 31.6; mínimo 0.3–0.6, hasta 0.05 en I; aceptado por el investigador en el chat). Con los P de la T4 el máximo es 31.62 en ambos problemas (confirma que corrió la versión normalizada).

**Criterio de la Enmienda 1 (perdida finita y, en el paso 100, ≤ 2× la de B): 20 de 20 corridas pasan** (5 brazos × 2 problemas × CPU y GPU). **7a: cumple** (≤ 5.0e-11).

Razón de la pérdida en el paso 100 respecto de B (GPU; la CPU coincide salvo I en Burgers: 0.725 frente a 0.871):

| | J | E | I | F |
|---|---|---|---|---|
| Burgers | 1.063 | 0.815 | 0.871 | 0.645 |
| Allen–Cahn | 0.586 | 0.112 | 0.077 | 0.724 |

Esto es **solo estabilidad**: la prueba no mide error contra la referencia ni es evidencia de mejora (el error L2 relativo tras 100 pasos de Adam de J, E, I, F en Allen–Cahn es 0.74–0.94 frente a 0.90 de B). La divergencia del spec v2 desaparece. Nota: I en Burgers difiere entre CPU y GPU (razón 0.725 frente a 0.871; error tras L-BFGS 0.161 frente a 0.175), dentro de lo que ya se vio de sensibilidad numérica; no hay criterio de CPU↔GPU en esta repetición.
