# EXP-0028 Fase 1 — brazos y celdas que NO se corren (sin reescribir código existente)

Regla del spec v1.1: un brazo×problema solo se corre si el código existente ya lo soporta. Estos no, y quedan en el mapa de palancas como «sin datos».

| Brazo | Problemas | Motivo |
|---|---|---|
| **H** (traspaso Adam→L-BFGS, EXP-0016) | los 4 | El disparador `stall_handoff.handoff_step` exige `k_max >= k_min = 4000` y 3 bloques consecutivos de 500 pasos tras `k_min`. Con la receta T1 de 1-B/1-C (Adam 2000 en Burgers, 4000 en el resto) levanta `ValueError` (Burgers) o no puede dispararse (con 4000 solo termina un bloque en `k >= k_min`, y exige 3): `k_t = k_max`, es decir H ≡ B por construcción. Correrlo con efecto requiere el presupuesto completo del control (Adam 20 000/40 000), unas 10× el cómputo y una receta distinta de la del resto del banco. **Mitigación sin costo:** cada corrida de B guarda `handoff_probe` (lo que decidiría el disparador con su historia de pérdida), así el mapa puede decir «H no se dispara a este presupuesto» con datos. |

## Brazos incluidos con salvedad
- **S** (muestreador autogobernado, EXP-0014): incluido por instruccion del investigador (2026-10-23) con un adaptador delgado (`adam_with_sampler` en `run_phase1.py`) sobre `self_gated_sampler.sample(mode='gated')`: 32768 candidatos Sobol, gate Spearman(|r|, ||grad u||), potencia 2, piso 0.5, L-BFGS sobre el pool uniforme inicial. **Desviacion declarada:** el calendario del spec 0014 (calentamiento 2000, cada 1000) es para un control de 20000/40000 pasos; con el presupuesto T1 de esta campana (Adam 2000 en Burgers) nunca remuestrearia (S == B). Se escalo por la convencion T1 (10 %): calentamiento 200, cada 100 (`--s-warmup 2000 --s-every 1000` restaura los literales).
- **W** (foco por responsabilidad, EXP-0018, rama P): corre en los 4 problemas con el código de `run_ceiling.py` sin tocarlo, pero es un **techo oráculo** (A* usa el error real contra la referencia; JSON con `label: oracle_ceiling`). Parte del estado final de Adam de B de la misma semilla (guardado en `*-B-seed*-adam.pt`), así que su reloj de Adam es el de B y solo difiere el L-BFGS. No es un mecanismo practicable; se reporta aparte.
- **R** (RAD, EXP-0025): `observer_v0.Observer` en modo `fixed:aR` (K = 500, 8192 candidatos, piso uniforme 0.5, presupuesto fijo), igual que el brazo R de 0025; extendido a Allen-Cahn d=2e-4 y Fisher-KPP porque `residual_at` lee `prob.pde_residual` (ya presente en el arnés 1-C).
- **C** (regrid, idea A de 0024): diseño D1 de `t1_trainer.DESIGNS` (cuantiles puros una vez al 25 % de Adam), usando `adam_with_hook` tal cual; es genérico en el problema.

## Fuera de esta ejecución (por spec)
NS2d-CG y DFG (Fase 1c) y piezo (Fase 4): requieren el mecanismo en residuo vectorial; este bloque es solo la Fase 1a/1b + ampliación v1.1 en los cuatro problemas escalares.
