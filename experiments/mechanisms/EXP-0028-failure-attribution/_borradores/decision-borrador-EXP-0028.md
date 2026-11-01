---
experiment_id: EXP-0028
decision: BORRADOR_SIN_FIRMAR   # propuesta del agente: accepted_as_exploratory_campaign
claims: []
decided_by: pendiente (Rodrigo Mansilla)
date: pendiente
---

# Borrador de decisión, EXP-0028 (atribución de fallos del mecanismo)

**Estado: borrador del agente, sin firmar.** Para que sea decisión, copiar a `decision.md`, ajustar `decision`, `decided_by` y `date`, y firmar. Marcado **[verificar]** donde no releí la fuente completa.

## Qué se pre-registró
`spec.md` v1.1 a v1.6 (firmada el 2026-10-23) y `predictions.md` (firmada el 2026-10-25 con «mejora modesta» para F y E). Hipótesis: H-ATTR-A, H-ATTR-B, H-ATTR-C, H-LEV-1, H-PRED-1. Banco: Allen–Cahn, Burgers, Fisher–KPP, NS2d-CG, DFG, más piezo (converso y directo).

## Resultado por hipótesis

| Hipótesis | Lectura | Fuente |
|---|---|---|
| H-ATTR-A (la ganancia de E viene de la base, no del residuo) | **Mixta, por problema.** Respaldada en NS-flat y DFG (receta corta) y en algunos casos de AC; no respaldada en AC y KPP a receta completa (D5: E mejor que F en KPP 0.24, AC 0.72) | `results/phase1*_scored.md`, `predictions.md` §1 |
| H-ATTR-B (el daño viene de la parte adaptada al residuo) | **No concluyente [verificar]:** en lo que revisé, la condición de aplicación (E empeora con mediana > 1.1) casi no se dio | `results/phase1*_scored.md` |
| H-ATTR-C | **[verificar]** no revisé su veredicto | `spec.md` |
| H-LEV-1 | **[verificar]** no revisé su veredicto | `spec.md` |
| H-PRED-1 (predicción para piezo, v1.3: no se gana por tasa base) | **No respaldada (provisional, n = 3 por objetivo).** Converso y directo: mejor brazo observado E (como se previó) pero el peor fue F (se previó J) y las direcciones acertadas no superan la tasa base | `results/piezo_vs_predictions.md` |

## Hallazgos que sostiene la campaña (descriptivos)
- A receta completa, F y E mejoran en AC, KPP y Burgers (n = 2) y se diluyen en NS (n = 3 a 5, todo ≈ 0.95); la dilución con presupuesto suficiente **no** es universal (AC con 40 000 pasos sigue en E 0.21).
- En Burgers, E es lotería por semilla; tras L-BFGS, F daña (2.05, n = 2).
- Piezo: E mejor que F en converso y directo; F el peor brazo (contra la predicción).
- El voltaje flotante no explica el error del directo (ya es el minimizador exacto, `analisis_piezo/check_Vf.md`).

## Decisión propuesta
**Aceptar como campaña exploratoria** (mismo precedente que EXP-0013): produce el mapa de dónde sirve E y la atribución parcial, y la predicción para piezo no se sostiene. **Sin claims confirmatorios**, por n pequeño, hardware mixto y recetas cortas en parte del banco. Vale decir también que lo que cierra el hilo de piezo es EXP-0029/0030, no esta decisión.

## Pendiente de cerrar por el investigador
- Verificar H-ATTR-B, H-ATTR-C y H-LEV-1 contra el spec antes de firmar.
- Burgers con receta completa (2 semillas) y DFG (cortado) siguen incompletos; decidir si se declaran como límite.
