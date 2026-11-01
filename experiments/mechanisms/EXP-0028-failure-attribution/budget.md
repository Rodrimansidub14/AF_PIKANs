# EXP-0028 Fase 1 — presupuesto (2026-10-23)

## Qué se midió y qué es proyección
- **Prueba corta en CPU** (1 semilla 4001 por brazo y problema; Adam 6–10 pasos, L-BFGS 5, recálculo cada 5 pasos, N = 64): todos los brazos corren de punta a punta en los 4 problemas (B, ID, F, J, E, I, S, R, C, W en `ac`; B, F, E, S, R, C, W en `kpp`; B, S, J en `ac2e4`; B, S, I, E en `burgers`), también desde el bundle plano (`BUNDLE_DIR`, incluido W vía `run_ceiling.py` del bundle).
- **B == ID: coinciden bit a bit** (Allen-Cahn d=1e-3, semilla 4001: error Adam 0.6965690575750904 y final 0.9233319811316515 en ambos). W parte del mismo estado de Adam que B (error Adam idéntico, como debe ser).
- **Tiempos de CPU no sirven para estimar T4:** la máquina estaba al 100 % de carga por otros procesos (de 0.55 s/paso medido en EXP-0023 a ~2 s/paso aquí; 85–250 s por corrida de 10 pasos). Se declara y no se extrapola.
- **La estimación de T4 usa tiempos reales de T4 con la misma receta y el mismo arnés** (EXP-0023 bloques 1-B/1-C, EXP-0025, EXP-0026; 5–20 corridas por celda): segundos por corrida (Adam + L-BFGS, 2 ranuras en paralelo):

| Problema | B | F | J | E | I | R | C |
|---|---|---|---|---|---|---|---|
| Allen-Cahn d=1e-3 | 95–101 | 97–101 | 102 | 103–112 | 104 | 105 | ~ B |
| Allen-Cahn d=2e-4 | 96–97 | 101–103 | — | 105–107 | — | — | — |
| Fisher-KPP | 97 | 101 | — | 102–103 | — | — | — |
| Burgers RBF48 | 106–109 | 107 | 116 | 112–117 | 122 | 107 | ~ B |

  Sin medición en T4 (proyección): **C** ≈ B + una reubicación (< 2 s); **S** ≈ B + 19 (Burgers) o 38 (resto) remuestreos de 32 768 candidatos, cada uno ≈ 3–4 pasos de Adam: ≈ 1.05 × B; **W** ≈ solo el L-BFGS (≈ 20–25 s) más el cálculo de A* porque reutiliza el estado de Adam de B.

## Total en T4×2 (2 ranuras)
Por problema: 50 corridas principales (5 brazos × 10 semillas) ≈ 5 150 s (Burgers ≈ 5 680 s) + palancas 5 semillas × (S 112 + R 107 + C 105 + W 25) ≈ 1 750 s. Total ≈ 28 200 s de GPU-segundos, 260 corridas + 20 de S.

| | GPU-s | Pared con 2 ranuras | + calentamientos y arranques |
|---|---|---|---|
| Un solo kernel (4 problemas) | ≈ 28 200 | ≈ 3.9 h | **≈ 4.1 h > tope de 4 h** |
| phase1a (ac, ac2e4) | ≈ 13 800 | ≈ 1.9 h | **≈ 2.0 h** |
| phase1b (burgers, kpp) | ≈ 14 400 | ≈ 2.0 h | **≈ 2.1 h** |

**Propuesta: partir en dos lanzamientos** (`phase1a`, `phase1b`), cada uno bajo el tope con holgura. Topes de cada kernel: sin trabajo nuevo después de 3.4 h, kill a 3.9 h; el orden de despacho pone primero las semillas 4001–4005 con todos los brazos y después 4006–4010, de modo que un corte afecta solo semillas tardías de los brazos principales. Las corridas son reanudables (un JSON por corrida), así que un kernel cortado se relanza y completa lo que falta. El kernel único (`...-phase1`) queda construido pero **excede el tope**; no se recomienda.

## Notas
- Reloj por corrida: `adam_seconds`, `lbfgs_seconds`, `total_seconds` (el costo de recalcular el precondicionador entra en el reloj, como en 0023). Para W el reloj de Adam es el de B.
- Calendario de S escalado a la convención T1 (calentamiento 200, cada 100); ver `missing_arms.md`.
- Estas cifras son proyecciones de corridas previas; la primera ejecución real las confirma o corrige.
