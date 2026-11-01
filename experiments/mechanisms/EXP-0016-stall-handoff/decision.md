# Decisión pendiente — EXP-0016 v2, tras las etapas A y B

Preparado por un agente el 2026-10-19. **No decide el agente.** Detalle y límites en `results.md`.

## Resumen de la evidencia

| | Burgers | Allen–Cahn |
|---|---|---|
| H-HANDOFF-1 (techo de error), criterio firmado | se sostiene (GM 0.43; 4/5), pero **no se distingue de ruido** (permutación p = 0.23) | no (GM 0.85) |
| H-HANDOFF-T (techo de costo) | se sostiene (4/5) | 2/3 |
| H-AWARE-1 (el mapa interno localiza el error) | **no** (Spearman +0.145, solape 0.004; residuo +0.455 / 0.393) | **no** (+0.190 / 0.000) |
| H-HANDOFF-2 (el disparo cae donde conviene) | **no** (el disparo interno no dispara en 5/5) | **no** |

Regla firmada: si H-AWARE-1 falla, el "dónde" interno se cierra en esta construcción y no se pasa a la etapa C con el brazo T. Comprobación posterior: la estructura |∇u| tampoco le gana al residuo.

## Opciones

1. **Cerrar el EXP-0016 como resultado negativo del "dónde" interno** (asentamiento de las bases de la capa 0) y escribirlo en la tesis: entre las señales sin referencia probadas, el residuo es la mejor localizadora del error en Burgers durante Adam. Sin costo de GPU.
2. **Abrir un experimento corto de "cuándo / costo"**: el traspaso temprano (k = 0.25–0.40) no empeoró el error mediano en Burgers y ahorra 60 %–75 % del tiempo de Adam, pero con n = 5 no es distinguible de ruido y en Allen–Cahn empeora. Haría falta más semillas (p. ej. 15–20 solo en k = 0.40 y 1.00, ≈ 3–4 h de T4×2) y un criterio de no inferioridad firmado antes.
3. **Probar otra señal interna para el "dónde"** con otra parte de la representación (capas ocultas, o la capa 0 sin separabilidad por dimensión), con su ceiling en lazo abierto sobre los mismos checkpoints antes de cualquier corrida. Evidencia previa débil: el residuo ya localiza la zona del frente en Burgers.

- [x] Decisión del investigador: **opción 1** — se cierra el EXP-0016 v2 como resultado negativo del "dónde" interno basado en el asentamiento de las bases de la capa 0; no se corre la etapa C con el brazo T — fecha: 2026-10-20 (respuesta en el chat: "Opción 1: cerrar"; registrado por el agente).
  - Etapa A: el criterio firmado de H-HANDOFF-1 y H-HANDOFF-T se cumple en Burgers, pero el techo **no se distingue de ruido** (permutación p = 0.23; análisis no preregistrado). No se reporta como efecto.
  - La opción 3 queda absorbida por el EXP-0017 (batería de señales y atribución en el espacio de la representación, solo CPU).
