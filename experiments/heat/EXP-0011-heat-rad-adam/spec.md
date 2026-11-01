---
experiment_id: EXP-0011
slug: heat-rad-adam
status: decided         # draft | preregistered | running | scored | decided
suite: tier2_candidate
problem_id: problem.heat_1d (u_t = 0.1 u_xx, EXP-0007 contract)
model_id: model.rbf_pikan (rbf_fixed, EXP-0008 control)
mechanism_id: mechanism.rad (src/mechanisms/rad.py, power 2, uniform_fraction 0.5)
hypothesis_id: H-RAD-HEAT-1
preregistered_at: 2026-10-03
preregistered_by: Rodrigo Mansilla (investigador; firma en el chat: "firmo EXP-0011")
---

# EXP-0011 — RAD sobre el control RBF-PIKAN de Heat (primer experimento de mecanismo)

Redactado el 2026-10-03 por un agente a partir de lo que el investigador aprobó en el chat ("apruebo todo esto": Heat como caso, etapa Adam como métrica, mejora relativa ≥ 20 %, n = 5). **Firmado el 2026-10-03** por el investigador; desde ese momento los umbrales no se mueven.

## Pregunta

¿El muestreo residual-adaptativo (RAD) de los puntos de colocación, a igual número de puntos y de pasos de Adam, reduce el error de campo del control RBF-PIKAN de Heat al final de Adam?

## Hipótesis (falsable)

H-RAD-HEAT-1: con la receta del control (EXP-0008, semillas pareadas 901–905, float64, 20 000 pasos de Adam, lote completo de 4 096 puntos interiores), reemplazar el pool fijo de puntos interiores por RAD reduce el L2 relativo en la malla analítica 512×256 **al final de Adam** en al menos 20 % (razón por semilla RAD/control ≤ 0.80, media geométrica de las razones ≤ 0.80).

El mecanismo actúa solo en la etapa Adam. L-BFGS se reporta como prueba de estrés, no como criterio de éxito.

## Control (congelado)

RBF-PIKAN `rbf_fixed`, semillas 901–905, pools y checkpoints de S2: fin de Adam `2.197e-4 ± 1.80e-4`; tras L-BFGS completo `1.857e-6 ± 1.8e-7` (baseline v1 propuesto, `baseline-v1-freeze-proposal-20261001.md`). El brazo RAD reutiliza pesos iniciales, semilla y puntos iniciales y de borde del control; solo cambia la selección de puntos interiores.

## Mecanismo

`src/mechanisms/rad.py::sample_rad` (ya existente; no ve valores de la solución ni la malla de evaluación, no importa `evals`). Candidatos: pool Sobol de 16 384 puntos interiores de la semilla; cada 1 000 pasos de Adam se evalúa el residuo PDE |r| sobre los candidatos y se re-muestrean 4 096 puntos con probabilidad `0.5/N + 0.5·|r|²/Σ|r|²`; sin corrección por importancia en la pérdida (como en el módulo). Mismo número de puntos por paso que el control; el costo adicional (evaluaciones del residuo sobre candidatos) se reporta en segundos de GPU.

## Métrica primaria

| Campo | Valor |
|---|---|
| Métrica | L2 relativo de campo, malla analítica 512×256, **fin de Adam (20 000 pasos)** |
| Dirección | minimizar |
| Comparación | pareada por semilla contra el control, n = 5 |
| **Delta que cuenta** | razón RAD/control ≤ 0.80 en media geométrica y en ≥ 4 de 5 semillas |

## Métricas de guardia

| Métrica | Restricción |
|---|---|
| L2 relativo tras L-BFGS completo | no peor que 2× el control (`≤ 3.7e-6`); se reporta aparte |
| Tiempo de GPU por corrida | ≤ 1.5× el control, incluyendo el re-muestreo |
| Estabilidad | 5/5 corridas terminan con pérdida finita |

## Condición de falsación

La hipótesis se rechaza si **cualquiera** de estas ocurre: (a) la media geométrica de las razones RAD/control es > 0.80; (b) menos de 4 de 5 semillas tienen razón ≤ 0.80; (c) se violan las guardias. Si la razón pareada es > 1.0 en la mayoría de las semillas, RAD queda como *no beneficioso en este control*. Un resultado a favor **no** es un claim: pasa al gate `scientific_claim` del investigador.

**Advertencia de potencia (declarada antes de correr):** el control tiene un coeficiente de variación del 82 % entre semillas al final de Adam (2.2e-4 ± 1.8e-4). Con n = 5, un efecto del 20 % solo es detectable de forma fiable por la comparación pareada, no por la media absoluta. Un resultado ambiguo (razones entre 0.8 y 1.0, mezcladas) se reporta como "no concluyente", no como éxito ni fracaso. Precedente local: en Burgers (EXP-0006) RAD no mejoró el control.

## Presupuesto

| Campo | Valor |
|---|---|
| Suite / tier | tier2_candidate / 2 |
| Backend | Kaggle T4×2, privado |
| Semillas | 901, 902, 903, 904, 905 (pareadas con el control) |
| Corridas | 5 (solo brazo RAD; el control ya existe) |
| Costo máximo | 1.5 h de sesión, tope duro 5 400 s; cuota ≥ 6 h de reserva |

## Superficie de edición declarada

- [x] `src/mechanisms/rad.py` (y su adaptador de entrenamiento en `src/mechanisms/rad_callback.py`)
- [ ] ningún otro archivo; no se toca `evals/`, el modelo base ni la PDE.

## Amenazas de validez

- **Leakage:** el residuo se evalúa solo sobre candidatos de entrenamiento; el error de referencia no entra en la política de puntos. Verificar con `tests/test_mechanism_isolation.py` antes de lanzar (Tier 0).
- **Pools:** el brazo RAD usa los mismos puntos iniciales y de borde que el control; el pool interior inicial es el del control.
- **Comparación a presupuesto igual:** pasos de Adam iguales, pero RAD gasta cómputo extra en el re-muestreo; por eso se reporta segundos de GPU y no solo pasos.
- **Piso de L-BFGS:** el control llega a un piso de discretización de ~1.9e-6 con 4 096 puntos; un mecanismo no puede mostrar mejoras por debajo de ese piso tras L-BFGS.

## Firma

El pre-registro entra en vigor cuando el investigador cambia `status` a `preregistered` y añade fecha y nombre. A partir de ese momento los umbrales no se mueven.

- [x] Firmado por el investigador — fecha: 2026-10-03 — nombre: Rodrigo Mansilla (mensaje de chat "firmo EXP-0011", tras leer el texto completo del pre-registro)
