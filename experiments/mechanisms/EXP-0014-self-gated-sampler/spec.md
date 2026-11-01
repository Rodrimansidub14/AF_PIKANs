---
experiment_id: EXP-0014
slug: self-gated-sampler
status: draft            # draft | preregistered | running | scored | decided — el investigador lo cambia a `preregistered`
suite: tier2_candidate
problem_id: problem.burgers_pinnacle_1d (primario); problem.allen_cahn_1d (réplica)
model_id: model.rbf_pikan (RBF48 en Burgers; RBF16 en Allen–Cahn), controles de baseline v1
mechanism_id: mechanism.self_gated_sampler (src/mechanisms/self_gated_sampler.py)
hypothesis_id: H-MECH-1
preregistered_at:
preregistered_by:
---

# EXP-0014 — Muestreador autogobernado: primera iteración de un mecanismo de *awareness*

Redactado el 2026-10-05 por un agente a partir de la decisión del investigador (opción A de EXP-0013). **No está firmado**: entra en vigor cuando el investigador cambie `status` a `preregistered` y ponga fecha y nombre. Hasta entonces no se ejecuta nada en GPU.

**Esta es la iteración 1 de un loop (propuesta → verificación → registro).** Su resultado, sea cual sea, abre las iteraciones siguientes (ver "Camino de iteraciones").

## Pregunta central y alcance

¿Y si el mecanismo supiera dónde poner su esfuerzo de manera automática? Esta iteración prueba el escalón más pequeño: que el modelo **decida solo qué señal propia seguir** para colocar sus puntos de colocación, a partir de la concordancia que mide entre dos señales sin referencia (residuo y estructura de su propia solución). No cubre *cuándo* cambiar de fase ni dónde poner *bases*; eso va en otras iteraciones (EXP-0015 mira bases).

## De dónde sale (EXP-0013, exploratorio)

- Las señales de estructura de la solución del propio modelo (|∇u|, |Δu|, anchura del frente) dominan la importancia en Burgers; el residuo se alinea con el error en DFG y por zona en Burgers, pero no en Heat, Allen–Cahn, NS2d-CG, y está anticorrelacionado en piezo.
- Patrón de EXP-0011/0012: RAD (residuo solo) mejora una etapa y empeora la otra.
- **Reglas heredadas, incorporadas aquí:** (1) se evalúa en las dos etapas; (2) L-BFGS corre sobre un pool **fijo y declarado de antemano** (el pool uniforme de la semilla, no los puntos del mecanismo); (3) la guardia de L-BFGS es vinculante.

## Mecanismo (fijado antes de correr)

`src/mechanisms/self_gated_sampler.py` (20 pruebas pasan, incluida la de aislamiento: no importa `evals`).

- Conjunto candidato fijo por semilla: 32 768 puntos Sobol del dominio (flujo independiente del pool de entrenamiento).
- Calentamiento: 2 000 pasos de Adam con el pool uniforme inicial. Después, **cada 1 000 pasos** (hasta el paso 19 000) se evalúan sobre los candidatos el residuo |r| y la estructura s = ‖∇u‖ del modelo actual (autograd), y se re-muestrean los puntos de colocación.
- **Gate (la "awareness"):** λ = clip(Spearman(|r|, s), 0, 1). Densidad: p = u/N + (1−u)·[λ·w(|r|) + (1−λ)·w(s)], con w(x) = x²/Σx² (potencia 2), u = 0.5 (piso uniforme, igual que RAD). Muestreo con reposición, sin corrección por importancia (misma convención que RAD).
- Mismo número de puntos de colocación y de pasos de Adam que el control; solo cambian los puntos. L-BFGS: sobre el pool uniforme inicial de la semilla (el del control), sin tocar.

## Brazos y comparador

| Brazo | Qué es |
|---|---|
| **C** control | receta v1 sin mecanismo, **réplica en la misma sesión** (comparador pareado) |
| **S** estructura fija | `mode='structure'` (λ = 0): ablación que fija la señal; descriptiva |
| **G** autogobernado | `mode='gated'`: **brazo primario** |

Comparador pre-registrado: la **réplica del control en la misma sesión**, pareada por semilla (en EXP-0011 las réplicas no reprodujeron por semilla el control archivado, razones 0.46–1.96×; por eso el archivado queda como paridad, no como comparador). Control archivado de v1 para paridad: Burgers RBF48 tras L-BFGS 3.87e-4 ± 2.2e-4; Allen–Cahn RBF16 2.686e-3 ± 9.5e-4.

## Problemas y recetas (idénticas a v1)

| | Burgers (primario) | Allen–Cahn (réplica) |
|---|---|---|
| Modelo | RBF-PIKAN ancho 48 (29 376 par.), float32 | RBF-PIKAN ancho 16, float32 |
| Puntos | 8 192 Sobol por semilla | 9 200 de residuo (8 000 + 400 + 800) por semilla |
| Adam | 20 000 pasos, lr 1e-3 | 40 000 pasos, lr 1e-3 |
| L-BFGS | strong-Wolfe, tope 600 s, parada por plateau | tope 300 s |
| Referencia | Cole–Hopf 1025×129 (evaluador del verificador) | `Allen_Cahn.mat` (aceptado 2026-10-01) |
| Semillas | 301–305 | 601–605 |

## Métrica primaria y criterio (vinculante al firmar)

**Primaria (Burgers, brazo G):** L2 relativo contra Cole–Hopf **tras L-BFGS**. Razón por semilla G/C (misma semilla, misma sesión). **El criterio se cumple si** la media geométrica de las razones es ≤ 0.80 **y** ≥ 4 de 5 semillas tienen razón ≤ 0.80.

**Guardias (todas deben cumplirse):**
- g1: L2 al final de Adam, razón G/C en media geométrica ≤ 1.25 (no empeorar la etapa Adam).
- g2: tiempo de GPU por corrida ≤ 1.5× el del control (incluye el costo del mecanismo).
- g3: 5/5 corridas con pérdida finita.
- g4: réplica en Allen–Cahn: razón G/C tras L-BFGS en media geométrica ≤ 1.25 (no peor).

**Secundarias (se reportan, no deciden):** Allen–Cahn con el mismo criterio de 0.80; brazo S frente a C y frente a G; razón en la etapa Adam; trayectoria de λ (¿el gate fue inerte?).

## Condición de falsación

H-MECH-1 se rechaza si: (a) la media geométrica de G/C tras L-BFGS en Burgers es > 0.80; (b) menos de 4 de 5 semillas tienen razón ≤ 0.80; o (c) se viola alguna guardia. **Banda no concluyente:** media geométrica entre 0.80 y 1.00, o razones mezcladas: se reporta "no concluyente", ni éxito ni fracaso. Un resultado favorable **no** es un claim (gate `scientific_claim`).

## Límites declarados antes de correr

- **Potencia:** la dispersión entre semillas del control tras L-BFGS es alta (CV ≈ 0.6 en Burgers); con n = 5 solo se detecta con fiabilidad un efecto grande en la comparación pareada.
- **El gate puede ser inerte:** en EXP-0013 la alineación residuo–estructura en Burgers fue baja (≈ 0.0–0.17 en Adam), por lo que se espera λ ≈ 0 y G ≈ S. Si ocurre, el efecto atribuible es el de la señal de estructura, no el del gate; esa inercia es en sí un resultado.
- **Atribución parcial:** G vs S aísla el gate; S vs C aísla la señal de estructura; no hay brazo de residuo solo (RAD ya dio negativo en EXP-0006/0011/0012 sobre estos controles).
- Sin corrección por importancia en la pérdida; muestreo con reposición (duplicados).

## Presupuesto

| Campo | Valor |
|---|---|
| Backend | Kaggle T4×2, privado; **se lanza después de la renovación de cuota (2026-10-10)** |
| Corridas | Burgers 15 (C, S, G × 5) + Allen–Cahn 15 = 30 |
| Costo estimado | Burgers ≈ 2.1 h + Allen–Cahn ≈ 1.9 h de sesión, +15 % de margen por el mecanismo; **tope duro 5 h**; reserva ≥ 12 h tras el lanzamiento |
| Pruebas previas (Tier 0/1, CPU) | pruebas unitarias del mecanismo (hechas, 20 pasan); humo del ejecutor de 200 pasos en CPU; prueba de aislamiento (hecha) |

Si se lanza junto con EXP-0015 en la misma sesión, el brazo de control se comparte y el costo conjunto baja (ver EXP-0015).

## Superficie de edición declarada

- [x] `src/mechanisms/self_gated_sampler.py` y su adaptador de entrenamiento en `experiments/mechanisms/EXP-0014-self-gated-sampler/` (el ejecutor puede llamar a `evals`? **No**: el entrenamiento no lee referencias; el puntaje se hace en un paso aparte, como en v1).
- [ ] No se toca `evals/`, el modelo base ni la PDE.

## Camino de iteraciones (por qué esto "abre la puerta")

| Resultado de EXP-0014 | Siguiente iteración |
|---|---|
| Cumple (G < C, guardias OK) | EXP-0016: ablaciones (potencia, piso uniforme, cadencia), NS2d-CG con el mismo mecanismo (orden acordado), más semillas |
| No concluyente | EXP-0016: más semillas o un segundo cue (p. ej. señal de bases de EXP-0015) |
| Rechazado | se registra el patrón en `registry/failures.jsonl`; se pasa a otra familia de EXP-0013: *cuándo* (disparador de fase por tasa de mejora del residuo) o *bases* |
| G ≈ S (gate inerte) | EXP-0016 con un gate que sí discrimine (p. ej. selección de cue por sonda de mejora del residuo retenido) |

## Firma

- [ ] Firmado por el investigador — fecha: ____ — nombre: ____
