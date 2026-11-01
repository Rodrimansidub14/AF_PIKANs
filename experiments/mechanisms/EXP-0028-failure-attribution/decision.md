---
experiment_id: EXP-0028
decision: accepted_exploratory
claims: [C1, C2, C3, C4, C5, C7]
decided_by: Rodrigo Mansilla (firma en el chat, 2026-10-26: «firmo todo»)
date: 2026-10-26
---

# Decisión, EXP-0028 (atribución de fallos del mecanismo)

**Estado: firmada por el investigador el 2026-10-26** («firmo todo»), sobre el texto redactado por el agente. Los claims aceptados están en `registry/claims.jsonl`.

## Qué se pre-registró
`spec.md` v1.1 a v1.6 (firmada el 2026-10-23) y `predictions.md` (firmada el 2026-10-25 con «mejora modesta» para F y E). Hipótesis: H-ATTR-A, H-ATTR-B, H-ATTR-C, H-LEV-1, H-PRED-1. Banco: Allen–Cahn, Burgers, Fisher–KPP, NS2d-CG, DFG, más piezo (converso y directo).

## Resultado por hipótesis

| Hipótesis | Lectura | Fuente |
|---|---|---|
| H-ATTR-A (la ganancia de E viene de la base, no del residuo) | **Mixta, por problema.** Respaldada en NS-flat y DFG (receta corta) y en algunos casos de AC; no respaldada en AC y KPP a receta completa (D5: E mejor que F en KPP 0.24, AC 0.72) | `results/phase1*_scored.md`, `predictions.md` §1 |
| H-ATTR-B (el daño viene de la parte adaptada al residuo) | **No concluyente:** en las tablas puntuadas revisadas la condición de aplicación (E empeora con mediana > 1.1) casi no se dio, y donde se dio (Burgers, KPP) E es dependiente de la semilla | `results/phase1*_scored.md` |
| H-ATTR-C | **Sin criterio propio en el spec** (solo se nombra en el encabezado; el eje C es descriptivo, fase 1b, «sin umbral propio»). No se emite veredicto | `spec.md` |
| H-LEV-1 | **Sin criterio propio en el spec** (solo se nombra en el encabezado; la fase 2 por propiedades es descriptiva, «sin regresión con p»). No se emite veredicto | `spec.md` |
| H-PRED-1 (predicción para piezo, v1.3: no se gana por tasa base) | **No respaldada (provisional, n = 3 por objetivo).** Converso y directo: mejor brazo observado E (como se previó) pero el peor fue F (se previó J) y las direcciones acertadas no superan la tasa base | `results/piezo_vs_predictions.md` |

## Hallazgos que sostiene la campaña (descriptivos)
- A receta completa, F y E mejoran en AC, KPP y Burgers (n = 2) y se diluyen en NS (n = 3 a 5, todo ≈ 0.95); la dilución con presupuesto suficiente **no** es universal (AC con 40 000 pasos sigue en E 0.21).
- En Burgers, E es lotería por semilla; tras L-BFGS, F daña (2.05, n = 2).
- Piezo: E mejor que F en converso y directo; F el peor brazo (contra la predicción).
- El voltaje flotante no explica el error del directo (ya es el minimizador exacto, `analisis_piezo/check_Vf.md`).

## Decisión propuesta
**Aceptar como campaña exploratoria** (mismo precedente que EXP-0013): produce el mapa de dónde sirve E y la atribución parcial, y la predicción para piezo no se sostiene. **Sin claims confirmatorios**, por n pequeño, hardware mixto y recetas cortas en parte del banco. Vale decir también que lo que cierra el hilo de piezo es EXP-0029/0030, no esta decisión.

## Límites declarados
- Verificado el 2026-10-26: H-ATTR-C y H-LEV-1 no tienen regla de decisión en el spec; quedan como descriptivas y no se les atribuye veredicto.
- Burgers con receta completa (2 semillas) y DFG (cortado por un apagón) siguen incompletos y se declaran como límite.
