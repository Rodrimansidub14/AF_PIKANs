# Handoff 2026-10-18 — confirmar el paso 0 y elegir la dirección del EXP-0016

## Estado al 2026-10-20 (leer primero)

- **EXP-0016 v2, etapas A y B corridas** (`experiments/mechanisms/EXP-0016-stall-handoff/results.md`). A: H-HANDOFF-1 y T se cumplen en Burgers por la letra del criterio, pero el techo no se distingue de ruido (permutación p = 0.23). B: **H-AWARE-1 y H-HANDOFF-2 fallan**; el residuo localiza mejor que el mapa de asentamiento. `decision.md` del EXP-0016 sigue sin decidir (recomendación del agente: opción 1, con la nota de la permutación).
- **Restricciones del investigador:** el título no cambia y el mecanismo se diseña sí o sí.
- **EXP-0016 cerrado** (opción 1, 2026-10-20). **EXP-0017 firmado el 2026-10-20: listo para correr.** Empezar por las pruebas previas del spec (aislamiento, unitarias de A y Δ, y reproducir el residuo del EXP-0016: Burgers 0.455 / 0.393).
- **Siguiente:** `experiments/mechanisms/EXP-0017-representation-battery/spec.md` (firmado, solo CPU, sobre los estados de la etapa A). Parte 1: batería de señales que no son de actividad (incluida la geometría de resolución) contra el residuo. Parte 2: atribución del residuo a las bases a través de la representación, contrastada con lo que L-BFGS movió. Su árbol de decisión siempre termina en un mecanismo para el EXP-0018; la rama principal es la focalización en el espacio de la representación (precondicionamiento por base), específica de KAN.

## Estado al 2026-10-19

- **Tarea 1: hecha.** Paso 0 con n = 3 en DFG, Burgers y Heat (`step0/results.md`, `table_n3.md`). H-BASIS-2 no se sostiene en Burgers (O11/C 1.23); U41 mejora también Heat (control negativo). Decisión del investigador (2026-10-18, `step0/decision.md`): **bloque GPU del EXP-0015 cancelado**; rama A.
- **Tarea 2: en curso.** El EXP-0016 pasó a **v2-draft** (`experiments/mechanisms/EXP-0016-stall-handoff/spec.md`; la v1 de Claude Code queda en `spec-v1-draft.md`). Cambio pedido por el investigador: alinear el mecanismo con el título y los objetivos del protocolo.
  - La v1 usaba la pérdida total como señal (externa) y solo decidía *cuándo*. La v2 usa el **asentamiento de los coeficientes de cada base RBF de la capa 0** (interna) para decidir *cuándo* traspasar y *dónde* concentrar la fase de refinamiento (pérdida ponderada por el mapa de bases no asentadas, sobre el pool fijo, sin agregar capacidad).
  - La señal de pérdida de la v1 (`stall_handoff.py`, 17 pruebas) queda como **línea base**; el ejecutor `sweep.py` y sus 21 pruebas se conservan.
  - Hipótesis nuevas: **H-HANDOFF-T** (mismo error con ≤ 0.6 del reloj) y **H-AWARE-1** (el mapa interno localiza el error mejor que el residuo). Etapa C con brazos C, E, L, T, R y guardia del §4.5 del protocolo (no mejorar la región crítica empeorando el global).
- **Terminología:** "PIKAN-PDE" = RBF-PIKAN entrenada con la pérdida física. "PINN" solo para el MLP.

### Siguiente trabajo para Claude Code (antes de pedir la firma de A y B) — estado 2026-10-19
1. **Hecho.** `src/mechanisms/settling_focus.py` (señal de asentamiento por base, mapa m(x), disparo q* = 0.80, pesos de media 1): 33 pruebas sintéticas (`tests/test_settling_focus.py`) y aislamiento sin `evals`.
2. **Hecho.** `sweep.py` registra coeficientes de la capa 0 y r² por punto cada 500 pasos de Adam (y en el paso 0) y calcula el disparo interno sin referencia. 31 pruebas (`tests/test_stall_handoff_sweep.py`): k = 1.00 sigue reproduciendo bit a bit el control original **con el registro activado**, y el registro no cambia el entrenamiento.
3. **Hecho el 2026-10-19 (GPU).** Equivalencia en GPU: Allen–Cahn determinista; en Burgers la primera corrida de un proceso difería (arranque en frío) y con `sweep.warmup` el control, su réplica y k = 1.00 son idénticos bit a bit (`results/precheck-gpu/`, `results/warmup-check-gpu/`). **Sigue pendiente** el ejecutor de Heat (adaptador al código de la campaña S2; es lo primero que se recorta).
4. **Hecho en lo posible.** Vista previa del mapa interno: solo Heat tiene varios instantes archivados. El mapa no es constante (rango/media 0.5–0.9) y su forma se repite en parte entre semillas (Spearman mediano 0.45). Burgers y Allen–Cahn no se pueden ver hasta la etapa A. Detalle en el spec ("Vista previa del mapa interno").

Lo que queda para pedir la firma de A y B: ítem 3 (la prueba en GPU se corre antes de lanzar el bloque grande, no antes de firmar).

---

Para continuar en Claude Code. Estado: EXP-0015 v2 **firmado** (2026-10-18) con la enmienda del paso 0; paso 0 corrido con **una semilla por brazo** en los seis problemas (`experiments/mechanisms/EXP-0015-layer0-basis-refinement/step0/results.md`). El bloque GPU del EXP-0015 sigue bloqueado hasta cerrar la tarea 1.

## Tarea 1 — Confirmar el paso 0 con 2 semillas más (CPU, ~3 h)

**Por qué.** Con n = 1 no se puede aplicar la tabla de decisión del spec. Los tres problemas donde los brazos se separan son DFG, Burgers y Heat.

**Qué correr** (desde `experiments/mechanisms/EXP-0015-layer0-basis-refinement/step0/`, misma receta que la semilla 1):
```
python equiv_test.py
python fit_step0.py --problem dfg     --arms C,O11,U41     --seeds 2,3 --adam 2000 --lbfgs 2000 --ntrain 8000 --out results
python fit_step0.py --problem heat    --arms C,O11,U41     --seeds 2,3 --adam 2000 --lbfgs 2000 --ntrain 8000 --out results
python fit_step0.py --problem burgers --arms C,O11,U41,X11 --seeds 2,3 --adam 2000 --lbfgs 2000 --ntrain 8000 --out results
```
Tiempos de referencia en 2 núcleos de CPU: DFG y Heat ~3 min por corrida; Burgers ~20 min por corrida (ancho 48). Opcional si sobra tiempo: NS2d-CG (~27 min por corrida) y Allen–Cahn/piezo (~2–3 min).

**Cómo leerlo.** Por problema y brazo: razón brazo/C por semilla y media geométrica. Usar los umbrales del spec: mejora si media geométrica ≤ 0.80 y ≥ 2/3 semillas ≤ 0.80. Reportar también "PINN / ajuste C".

**Qué decide** (tabla de la enmienda en `../spec.md`):
- Ajuste C ≪ PINN (≥ 10×) en un problema → la capacidad no limita ahí; el bloque GPU de ese problema se recorta.
- Si O11 no mejora en Burgers con n = 3 → H-BASIS-2 (hipótesis primaria) no se sostiene en Burgers.
- Si U41 sigue mejorando en Heat (control negativo) → el efecto de más bases es en parte capacidad; cualquier brazo de bases futuro necesita un control con los parámetros igualados (p. ej. ancho de capa oculta aumentado hasta igualar a U41).

**Registro al terminar.** Agregar las corridas a `registry/runs.jsonl`, actualizar `step0/results.md` con la tabla n = 3, y dejar la decisión sobre el bloque GPU del EXP-0015 al investigador (`decision.md`).

## Tarea 2 — Elegir la dirección del mecanismo (EXP-0016)

Depende de la tarea 1. La evidencia actual (n = 1) apunta a la rama A.

**Rama A (principal si se confirma): el error de los PINN es de optimización.** Hoy lo indican piezo (274×), DFG (30×), NS2d-CG (22×) y Allen–Cahn (7×). Antecedentes propios que el spec debe usar:
- EXP-0011/0012: lo ganado en Adam se pierde con L-BFGS (RAD en Heat: 0.088× al final de Adam, 2–5× peor tras L-BFGS).
- En Burgers todas las corridas de L-BFGS terminaron por tope de tiempo: medir error contra tiempo de reloj, no solo el valor final.
- En piezo L-BFGS destruyó la solución en todas las variantes propias; la receta de Daniel no usa L-BFGS.
Candidatos de mecanismo a evaluar en el spec (elegir uno, con su señal interna):
1. **Traspaso Adam → L-BFGS decidido por la red**: cambiar de fase cuando una señal interna (p. ej. estabilización de los coeficientes por base, o caída del ritmo de mejora del residuo) lo indique, en lugar de un número fijo de pasos.
2. **Condicionamiento por capa**: reescalar o precondicionar los coeficientes RBF por base según su ocupación y su presión de gradiente.
En ambos: una señal sin referencia validada primero en lazo abierto (índice de efectividad, como en EXP-0013) antes de cerrar el lazo; evaluación en las dos etapas; curvas de error contra tiempo; seis problemas en la prueba de estrés; Burgers como problema de desarrollo.

**Rama B (secundaria): las bases en DFG.** Solo si la tarea 1 confirma O11 y U41 en DFG. Tendría que separar resolución de capacidad (control con parámetros igualados) antes de cualquier mecanismo de r-adaptividad de centros.

**Pendiente del investigador:** decidir si se pospone el EXP-0014 (propuesta registrada en `registry/proposals.jsonl` el 2026-10-18: su gate λ = Spearman(|r|, ‖∇u‖) solo pesa cuando las señales coinciden).

## Material relacionado
- Brief para Daniel (Claude Docs, privado hasta compartirlo): resultados de baselines, configuraciones de piezo, hallazgos, paso 0 y 6 preguntas.
- Figuras 3D y de campos por problema: `var/agent_runs/brief-figs-20261018/` (`make_fields.py`, `plot_fields.py`).

## Estado al 2026-10-20 (tarde)

- EXP-0017 calculado y cerrado en negativo; decisión del investigador: opción 1 (techo oráculo A*), registrada en `EXP-0017/decision.md` y `registry/proposals.jsonl`.
- `experiments/mechanisms/EXP-0018-responsibility-ceiling/spec.md` está en **borrador, sin firmar**. No correr nada de GPU hasta que el investigador firme (preregistro y uso del oráculo). Antes de firmar conviene calibrar los umbrales 0.85/0.90 con el ruido pareado del baseline v2.
- Pendiente en paralelo: pase de LaTeX normal del proyecto (referencias cruzadas de las secciones nuevas de `j2-resultados.tex` y `j3-discusion.tex`).

### Specs al 2026-10-20 (tarde)

- EXP-0018 `responsibility-ceiling` (ruta 3, techo oráculo A*): **firmado v2 el 2026-10-20** (criterios y uso de la referencia como oráculo). Es el único listo para correr; sigue lo que dice su spec y su `decision.md`.
- EXP-0022 `conditioning-diagnosis` (ruta 2, CPU, estados existentes): **firmado v2 el 2026-10-20**. Se corre con `python3 run_exp0022.py` (reanudable, un JSON por estado en `results/states/`) y se puntúa con `python3 analyze_exp0022.py`. Decide si EXP-0023 se firma.
- EXP-0023 `gram-preconditioning` (ruta 1, mecanismo derivado): borrador condicionado a H-COND-1 y H-COND-2 del 0022. No firmar antes.
- EXP-0024 `wind-tunnel-protocol` (ruta 5, protocolo): borrador; se firma una vez y autoriza explorar en T0/T1 con topes y registro en `registry/designs.jsonl`.
- Nota de numeración: el spec firmado del EXP-0018 llama «0019» al experimento que sustituiría A* por un estimador sin referencia; ese número queda reservado para eso y los tres borradores nuevos usan 0022 a 0024.

- EXP-0022 corrido y puntuado el 2026-10-20 (40 estados, CPU, ~25 min): H-COND-1 y H-COND-2 no se sostienen por la letra (ver `results.md`); decisión pendiente del investigador en `decision.md`. EXP-0023 no se firma sin decisión explícita.

- **2026-10-20 (noche):** el investigador firmó EXP-0023 v2 y EXP-0024 v2 (chat: «firmo 0023 y 0024»).
  - EXP-0023: **no lanzar GPU** hasta pasar las pruebas previas 1–7 del spec y registrar el presupuesto. Implementar `src/mechanisms/gram_preconditioning.py` reutilizando `EXP-0022.../gn.py` para los bloques.
  - EXP-0024: pendiente la prueba previa 1 (arnés T0 que reproduzca la tabla n=3 del paso 0 del EXP-0015) y la prueba 2 (`registry/designs.jsonl` fuera del alcance de los verificadores). La prueba de semillas pasa. Ronda 1: regridding de centros ocultos por cuantiles de activación (A) y variantes de precondicionamiento (B). Cada variante probada se anota en `registry/designs.jsonl`.
- **EXP-0024 ronda 1, nivel T0 corrido (2026-10-20):** ver `experiments/mechanisms/EXP-0024-wind-tunnel-protocol/round1-T0.md`. Idea A (reubicar centros ocultos por cuantiles): Allen–Cahn D1/D2 ≈ 0.68×, 5/5; Heat D1 empeora tras L-BFGS. Pendiente: entrenador T1 con pérdida de PDE y gancho `regrid.regrid_net` para D1, D2, D3 (Burgers y Allen–Cahn, semillas del bloque T), luego V (bloque 100–199). Falta además la prueba previa 1 completa (reproducir la tabla n=3 a presupuesto completo) y la prueba 2.
- **EXP-0024 T1 (2026-10-20):** `t1_trainer.py` y `tests/test_wind_tunnel_t1.py` listos (3 pasan). Plan y costos en `experiments/mechanisms/EXP-0024-wind-tunnel-protocol/T1-plan.md`. Tareas para Claude Code: (1) bundle de Kaggle usando `EXP-0018.../kaggle/build_bundle.py` como plantilla; (2) prueba corta en T4 (CTRL de T1 frente a CPU, segundos por paso); (3) pedir al investigador la autorización de lanzar el bloque (40 corridas: CTRL, D1, D2, D3 × semillas 1–5 × Burgers y Allen–Cahn). No lanzar sin esa autorización.
- **2026-10-20 17:32 — autorización de lanzamiento del bloque T1 del EXP-0024** (40 corridas en T4), condicionada a que la prueba corta en T4 pase. Condiciones y topes en `EXP-0024.../T1-plan.md`, sección «Autorización de lanzamiento». Orden para Claude Code: bundle → prueba corta en T4 → si pasa y la estimación ≤ 4 h, lanzar; si no, volver a consultar al investigador.
- **2026-10-20 (noche) — EXP-0024 T1, prueba corta en T4:** bundle (`experiments/mechanisms/EXP-0024-wind-tunnel-protocol/kaggle/`, builder basado en el del 0018) y prueba hechos; **no pasó por 0.11 pp en un punto** (Burgers D1 frente a la CPU, 20.11 % > 20 %); el control y Allen–Cahn sí; estimación del bloque 0.71 h. El bloque **no se lanzó** (condición 1 de la autorización). Pendiente de decisión: aceptar y lanzar, repetir Burgers D1 en GPU, o fijar otro criterio. Detalle en `T1-plan.md`.
- **2026-10-20 (noche) — EXP-0024 T1, diagnóstico de Burgers D1 (opción 2):** la GPU es idéntica a sí misma; float64 no elimina la diferencia con la CPU (7.6 %); una perturbación de 1e-7 cambia el error de D1 12.3 % y el del control 7.5e-6: sensibilidad numérica del diseño, no ruido de GPU ni error de código. Queda sin explicar la persistencia en float64. Bloque T1 **sin lanzar** (estimación 0.7 h); decisión pendiente. Detalle en `T1-plan.md`.
- **2026-10-20 (noche) — EXP-0024 T1, reubicación CPU vs GPU:** sobre un estado idéntico en float64 la diferencia máxima es 4.2e-11 (< 1e-9): la reubicación es inocente; la brecha de Burgers D1 es sensibilidad del entrenamiento posterior (salvedad en `T1-plan.md`). Por la regla del investigador se lanza el bloque T1 (40 corridas, estimación 0.7 h).
- **2026-10-20 (noche) — EXP-0024 T1 corrido (40/40, 36 min):** Allen–Cahn D1/D2/D3 pasan (medianas 0.695/0.675/0.765, 5/5); **Burgers: los tres empeoran** (21×/4.5×/11.5×, 0/5). Regla 3: ningún diseño pasa a V. Informe y opciones en `experiments/mechanisms/EXP-0024-wind-tunnel-protocol/round1-T1.md`. Decisión pendiente.
- **2026-10-20 (22:01) — decisiones firmadas por el investigador sobre EXP-0024:** (1) idea A cerrada como especifica de Allen-Cahn; sin mas diagnosticos ni variantes. (2) Regla de salida: si la ronda 1-B no pasa T1 (criterio ya firmado), termina la busqueda de mecanismo en Burgers/Allen-Cahn. **Orden para Claude Code:** (a) implementar `src/mechanisms/gram_preconditioning.py` y pasar las pruebas previas 1-7 del EXP-0023 (sin GPU hasta pasarlas), registrar el presupuesto; (b) un unico bloque T1 de las variantes B con los criterios del spec, sin variantes nuevas; pedir autorizacion de lanzamiento al investigador con el presupuesto medido; (c) en paralelo, listar lo que falta para pasar a NS/DFG y piezo (solver, datos de referencia) y proponerlo, sin lanzar nada.
- **2026-10-20 (23:54) — EXP-0023, enmienda 1 firmada** (`experiments/mechanisms/EXP-0023-gram-preconditioning/amendment-1.md`). **Orden para Claude Code:** (1) implementar P = (B/m + 1e-3 I)^(-1/2), m = traza(B)/11, y **verificar que los valores singulares de P caen en ~1-32** (si no, parar y consultar); (2) repetir la prueba 7b (5 brazos, 2 problemas, CPU y GPU, 100 pasos de Adam; criterio: perdida finita y <= 2x B en el paso 100); (3) si pasa, pedir autorizacion para el bloque T1 (50 corridas, semillas 1-5, convencion T1, ~2 h); si no pasa, registrar fallo de la ronda 1-B (regla de salida) y NO enmendar de nuevo.
- **2026-10-21 — EXP-0023:** el investigador acepta el minimo de sv de P (0.05-0.6; fe de erratas en amendment-1.md). Orden: repetir la 7b en T4 (5 brazos, 2 problemas, CPU y GPU, 100 pasos; finita y <= 2x B). Si pasa, pedir autorizacion para el bloque T1; si no, fallo de la ronda 1-B, sin nueva enmienda.
