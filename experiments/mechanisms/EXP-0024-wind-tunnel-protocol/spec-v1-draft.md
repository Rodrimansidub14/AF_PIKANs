---
experiment_id: EXP-0024
slug: wind-tunnel-protocol
version: v1-draft
status: draft          # protocolo: se firma una vez; autoriza la exploración dentro de los topes
suite: tier0_protocol
problem_id: problem.burgers_pinnacle_1d, problem.allen_cahn_1d (túnel); confirmación: los mismos
model_id: model.rbf_pikan
mechanism_id: none (protocolo de descubrimiento; se aplica a EXP-0023 y a cualquier mecanismo futuro)
hypothesis_id: ninguna (no contrasta hipótesis; define cómo se explora y cómo se confirma)
---

# EXP-0024 — Túnel de viento: separar descubrir de confirmar

Redactado el 2026-10-20 por un agente. **No está firmado.**

## Problema que resuelve

Hasta ahora cada idea candidata pasó por un preregistro firmado con n = 3 a 5 semillas y GPU. Así se usó un protocolo de **confirmación** como motor de **búsqueda**: cada negativo cuesta una firma, horas de GPU y poca información (p mínimo 1/32). El protocolo separa las dos funciones y conserva el rigor donde importa.

## Niveles

| Nivel | Qué es | Costo | Quién lo corre |
|---|---|---|---|
| **T0** | Ajuste supervisado sin PDE (el arnés del paso 0 del EXP-0015) con el mecanismo candidato | segundos a minutos, CPU | agente, sin firma por corrida |
| **T1** | Corrida corta de la PIKAN-PDE (propuesto: 10 % del presupuesto del baseline v2) en Burgers y Allen–Cahn | minutos a pocas horas | agente, sin firma por corrida |
| **V** | Validación: el finalista contra el baseline en semillas disjuntas de las de T0/T1, con el presupuesto y los criterios que el agente propone | presupuesto reducido | agente propone, el investigador aprueba |
| **C** | Confirmación: preregistro firmado como los anteriores, semillas nuevas, presupuesto completo | el actual | investigador firma |

Los resultados de T0, T1 y V son exploratorios: `scientific_accepted=false`. Solo C puede sostener una afirmación.

## Reglas (vinculantes al firmar)

1. **Libro de diseños.** Cada variante probada en T0 o T1 se anota en `registry/designs.jsonl`: id, fecha, descripción, nivel, semillas, métrica. No se descarta nada sin anotar. El número de diseños probados queda disponible para quien lea el resultado de C.
2. **Semillas disjuntas.** T0/T1 usan semillas 1–99 (propuesto) y V otro bloque; ninguna semilla se reutiliza entre niveles ni con las del EXP-0016 y 0017. Las de C se fijan en su preregistro.
3. **Selección del finalista fijada antes de explorar.** Métrica de selección: razón de error final frente al baseline, mediana sobre semillas en T1; el finalista debe además mejorar al baseline en V. Un solo finalista por ronda a C.
4. **Topes por ronda.** Hasta 30 diseños y hasta 8 h de cómputo de exploración (propuesto). Si se agotan sin finalista, la ronda se cierra y se reporta.
5. **Registro de transferencia.** Para cada finalista se anota si pasó C. Si dos finalistas seguidos pasan T1 y V pero fallan C, el túnel se declara no informativo para esa clase de mecanismo y la exploración sube de nivel (T1 con más presupuesto) en lugar de seguir con T0.
6. **El túnel no cambia los criterios de C.** Los umbrales de C no se ajustan a lo visto en la exploración.

## Qué autoriza la firma

Que el agente explore dentro de los topes en T0/T1 sin firma por corrida, **sin usar la referencia como entrada de ningún mecanismo candidato**. Un mecanismo que use la referencia (oráculo) sigue necesitando la aprobación explícita del investigador y la etiqueta `oracle_ceiling`.

## Límites declarados

- T0 mide aspectos de representación y optimización de un ajuste supervisado, no el paisaje de una pérdida con PDE. Un mecanismo puede ganar en T0 y no transferir; las reglas 5 y la existencia de T1 existen por eso.
- La exploración con muchas variantes infla los tamaños de efecto del finalista (maldición del ganador). Lo corrigen las semillas nuevas en V y C, no un ajuste estadístico.
- Un tope de diseños limita la búsqueda; si el mecanismo bueno está más lejos, la ronda no lo encuentra.
- No sustituye a los preregistros de los experimentos con hipótesis (EXP-0018, 0022, 0023); los complementa.

## Pruebas previas

1. El arnés de T0 reproduce la tabla n = 3 del paso 0 del EXP-0015 (mismos brazos C, U41, O11).
2. `registry/designs.jsonl` no es leído por ningún verificador ni escrito por `evals/` (el archivo es nuevo y de la misma naturaleza que `proposals.jsonl`).
3. Las semillas de T0, T1, V y C están en bloques disjuntos (prueba que lo comprueba).

## Superficie de edición declarada

Nuevo: `registry/designs.jsonl`, `experiments/mechanisms/EXP-0024-wind-tunnel-protocol/` (arnés T0/T1 reutilizando el paso 0 del EXP-0015), `tests/test_wind_tunnel_seeds.py`. Sin cambios en `evals/` ni en `registry/claims.jsonl`.

## Firma

- [ ] Protocolo: niveles, reglas 1–6, topes (30 diseños, 8 h) y qué autoriza — Rodrigo Mansilla — fecha: ____
