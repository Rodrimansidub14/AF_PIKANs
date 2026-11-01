# EXP-0025 — presupuesto medido (condición previa 3) — 2026-10-22

Medido con `tier1.py` en **Kaggle CPU, sin GPU** (kernel `rodrimandub/af-pikans-exp-0025-tier1-cpu`, 2 procesos × 2 hilos, float32): Allen–Cahn d = 1e-3, RBF16, pool 8 000+400+800, Adam 4 000, K = 500, n_p = 25, observador con `aR` y `aG` (variante E), H_d fresco de 8 192 puntos por decisión. B y O en el mismo hardware y en el mismo proceso, B primero. Solo etapa Adam (el L-BFGS es idéntico en B y O). Semillas 2901 y 2902, fuera de T1/V. Datos: `results/tier1_seed2901.json`, `results/tier1_seed2902.json`.

| Semilla | B (s) | O (s) | Razón O/B | Pasos extra / pasos | Segundos de sondeo |
|---|---|---|---|---|---|
| 2901 | 922 | 1 231 | **1.336** | 0.150 (= 3·n_p/K) | 325 |
| 2902 | 924 | 1 244 | **1.347** | 0.150 | 319 |

- **Límite del spec (reloj de O ≤ 1.5× B): se cumple** (1.34–1.35). El sondeo cuesta ~26 % del reloj de O; el 15 % de pasos extra es exacto.
- **Estimación del bloque (T4):** 50 corridas × ~57 s × factor de brazos (B 1.00, R ≈ 1.0, E ≈ 1.08, O ≈ 1.34, O− ≈ 1.0 ⇒ promedio ≈ 1.1) ≈ 52–57 min por bloque, **dentro del tope de 1.5 h por bloque y 3 h en total**. Es proyección, no medición en T4: el reloj en T4 no se midió.
- Se añade O−m si O resulta inerte (ver abajo): +10 corridas por bloque ≈ +10 min; sigue dentro del tope.

## Prueba 9: acciones e inercia (Tier 1, 2 semillas, solo frecuencia, sin error contra palancas fijas)

| Semilla | Acciones elegidas en las 8 decisiones | Inerte (> 90 % a0) |
|---|---|---|
| 2901 | a0 ×8 | **sí** |
| 2902 | aG, a0 ×7 | no (87.5 %), pero a0 > 50 % ⇒ el spec exige el brazo O−m |

- Agregado: a0 en 15 de 16 decisiones (93.8 %).
- **Decisión 0 (paso 0, red inicial):** mejora de `aG` sobre a0 = +0.35 (2901) y +0.52 (2902); la de `aR` +0.13 y −0.08. En 2901 `aG` queda **vetado por la guarda γ** (1 celda crítica empeora > 25 %); en 2902 pasa (0 celdas).
- **Decisiones 1–7 (red ya entrenada, pérdida ~1e-4 a 1e-3):** casi todas las mejoras medidas son **negativas y enormes** (hasta −1 607 para `aR`, −10.6 para `aG`): cambiar de palanca a mitad de Adam desajusta los momentos de Adam y el sondeo de 25 pasos paga solo el transitorio. Es la amenaza «sesgo de horizonte corto» del spec, ahora observada.
- No hubo `null` ni pérdidas no finitas.

## Condiciones previas al bloque (estado)
1. Pruebas 1–9: **cumplidas** (tests 1–8 pasan, 19/19 en `test_observer_v0.py`; prueba 9 aquí).
2. Ronda 1-C puntuada y variante fijada: **cumplida** (variante E).
3. Presupuesto medido y bajo el tope: **cumplido** (este archivo).
4. Política de campañas de `context/agent-policy.toml`: **pendiente** para EXP-0025 (la aprobación del 2026-10-22 cubre solo los bloques de 1-C).
