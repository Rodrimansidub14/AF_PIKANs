---
experiment_id: EXP-0022
slug: conditioning-diagnosis
version: v2             # firmada el 2026-10-20; v1 (borrador) archivada en spec-v1-draft.md
status: preregistered   # draft | preregistered | running | scored | decided
suite: tier1_regression  # solo CPU, sobre estados ya guardados; sin entrenamiento nuevo
problem_id: problem.burgers_pinnacle_1d (principal), problem.allen_cahn_1d (réplica)
model_id: model.rbf_pikan (RBF48 en Burgers; RBF16 en Allen–Cahn; 11 bases, ancho 1.75 × espaciado), pares (Adam, L-BFGS) de la etapa A del EXP-0016
mechanism_id: none (diagnóstico; decide si el EXP-0023 se firma)
hypothesis_id: H-COND-1 (el refinamiento se mueve por direcciones más planas de lo que su magnitud implica), H-COND-2 (el bloque por arista captura el mal condicionamiento)
preregistered_at: 2026-10-20
preregistered_by: Rodrigo Mansilla (mensaje de chat "firmo", 2026-10-20 16:01, sobre esta v2; registrado por el agente. Los cambios de la v1 a la v2 listados abajo no los había visto al firmar: si objeta alguno antes de que empiece la corrida, se enmienda antes de ver datos)
---

# EXP-0022 — ¿Qué estructura del problema de optimización frena a la PIKAN-PDE? Diagnóstico de condicionamiento

> **Terminología.** "PIKAN-PDE" = RBF-PIKAN entrenada con la pérdida física. Ninguna cantidad de este experimento lee la referencia: el código del cálculo solo importa `math`, `numpy` y `torch` (prueba por AST).

## Cambios respecto de la v1 (borrador)

1. **H-COND-1 reemplazada.** El criterio v1 (ρ_d ≤ 0.05) era trivial: GN tiene rango ≤ N mucho menor que el número de parámetros, así que casi cualquier dirección es plana. Medido en un estado de Burgers: ρ de una dirección gaussiana aleatoria = 9.3e-5, ρ_d = 4.2e-5. El criterio nuevo compara el desplazamiento con **el mismo desplazamiento con signos aleatorios** (misma magnitud por parámetro).
2. **Datos y método.** Subconjunto fijo de **2 048 puntos del pool de entrenamiento** de cada semilla (la pérdida que se optimiza de verdad; en v1 era una submalla de 8 192 puntos de la referencia). El espectro sale **exacto** de la Gram K = J Jᵀ / N (N × N); no se usa Lanczos ni SLQ. Con N = 2 048 hay a lo sumo 2 048 autovalores no nulos.
3. **m = 10 y 100.** m = 500 se reporta pero no decide: en Allen–Cahn el autovalor 500 cambia 19 % entre float32 y float64.
4. **Arista = (capa ℓ, salida o, entrada i)**, con sus 11 coeficientes (2 448 aristas en Burgers).
5. **Fila de decisión intermedia** (ver abajo), añadida porque dos estados de Burgers ya vistos quedaron cerca del umbral de Jacobi.

## Pregunta

En 4 de 6 problemas el error de la PIKAN-PDE es de optimización, no de capacidad. Los EXP-0015 a 0018 probaron remedios (dónde y cuánto refinar) sin nombrar la patología. **¿El problema de optimización está mal condicionado, y esa mala condición vive en una estructura que da la propia representación (los bloques por arista)?** Si sí, existe un mecanismo derivado y no buscado (EXP-0023).

## Cantidades (fijadas antes de correr)

- **J** = ∂r/∂θ en el estado de Adam, con r el residuo de la PDE por punto (la misma expresión que `prob.loss`) y θ todos los parámetros; doble precisión, J guardada en float32. **GN** = JᵀJ / N.
- **Espectro**: autovalores de K = J Jᵀ / N con `eigvalsh`. **κ_eff(m) = λ₁ / λ_m**, m = 10 y 100 (decide); m = 500 descriptivo.
- **Desplazamiento** d = θ_L-BFGS − θ_Adam en cada par. **ρ_d** = (dᵀ GN d / dᵀd) / λ₁. **ρ_sr** = media de ρ para 20 copias de d con signos aleatorios por parámetro (semilla fija). **Razón** ρ_d / ρ_sr.
- **Precondicionadores** (se aplican como P^(-1/2) GN P^(-1/2); se recalcula κ_eff):
  - *Crudo*: sin precondicionar.
  - *Jacobi*: P = diag(GN).
  - *Por arista (primario)*: P = bloque diagonal con el bloque 11×11 exacto de GN de cada arista, más ε·I con ε = 1e-3 × (traza/11). La rama `base` y el resto entran con su valor Jacobi.
  - *Por entrada* (descriptivo): un bloque por entrada (ℓ, i) con todos los coeficientes que la leen.
- **Gram de la base por arista** G_ℓi = E[φφᵀ]: número de condición y rango efectivo por capa, con las activaciones reales frente a entradas uniformes en [−1, 1] (separa el solape de las gaussianas de la concentración de las activaciones entrenadas). Descriptivo.
- **Localización**: fracción de la energía de d en el 20 % de aristas con más energía, por capa. Descriptivo.

## Hipótesis y criterios (vinculantes)

Mediana sobre los 40 pares (Burgers 25, Allen–Cahn 15), en **ambos** problemas.

- **H-COND-1.** ρ_d / ρ_sr ≤ 0.1 (propuesto): L-BFGS se mueve por direcciones al menos 10× más planas que las que implica la magnitud de su desplazamiento. Es la firma de combinaciones de bases solapadas que se cancelan.
- **H-COND-2.** κ_eff(100) con el precondicionador *por arista* ≤ 0.1 × el crudo **y** ≤ 0.33 × el de Jacobi. Lo segundo es el control importante: Adam ya hace algo parecido a Jacobi.

Se reporta además la comparación pareada por semilla con permutación con signo exacta; con 5 y 3 semillas el p solo informa, manda la mediana.

## Regla de decisión (vinculante)

| Resultado | Consecuencia |
|---|---|
| H-COND-1 y H-COND-2 se sostienen | El EXP-0023 pasa a la fila de firma. Mecanismo candidato derivado |
| H-COND-2 cumple contra el crudo pero no contra Jacobi, con razón a Jacobi entre 0.33 y 0.6 (mediana) | Ganancia moderada sobre Jacobi. El EXP-0023 solo se firma por decisión explícita del investigador, con Jacobi como brazo principal de comparación |
| H-COND-1 se sostiene, H-COND-2 no (razón a Jacobi > 0.6 o no cumple el crudo) | Hay mal condicionamiento pero no lo captura el bloque por arista. Se mira el reporte descriptivo (por entrada, por capa); el EXP-0023 se rediseña o se cierra la ruta |
| H-COND-1 no se sostiene en algún problema | El desplazamiento no es especialmente plano. Se reporta como hallazgo; la ruta de condicionamiento se cierra para ese problema y se discute si vale para el otro |

## Exposición previa a datos y calibración (declarada)

Antes de firmar se calcularon **tres estados** de los 40 (siguen en el análisis): Burgers 301 f=0.25 (presupuesto), Burgers 302 f=0.6 y Allen–Cahn 602 f=0.6 (calibración, con dos mitades disjuntas del pool y J en float64).

- Los umbrales de **H-COND-2** (0.1 y 0.33) estaban en el borrador v1, antes de cualquier cálculo. En los tres estados vistos: edge/crudo = 0.005–0.017; edge/Jacobi = 0.33 y 0.40 en Burgers y 0.16–0.17 en Allen–Cahn. Es decir, el criterio de Jacobi está en el borde en Burgers; no se movió.
- El umbral de **H-COND-1** (0.1) se fijó viendo un solo valor, Burgers 301 (0.04), y sin ver Allen–Cahn. Después se midió: Burgers 302 = 0.022; Allen–Cahn 602 = 0.14–0.15, es decir **por encima del umbral**. No se movió.
- **Ruido del cálculo** (log-diferencia entre las dos mitades del pool, un estado por problema): edge/crudo 0.26 y 0.006; edge/Jacobi 0.06 y 0.04; razón ρ 0.011 y 0.066. El ruido de muestreo es chico frente a la distancia de los umbrales; lo que manda es la variación entre estados.
- **Precisión**: float32 frente a float64 en los autovalores, error < 1e-4 para m = 10 y 100 en ambos problemas (por eso m = 500 no decide).
- Datos de la calibración: `calibration.json`.

## Límites declarados antes de correr

- GN es una aproximación local en el estado de Adam; no describe la trayectoria entera.
- Solo Burgers y Allen–Cahn (los estados existentes). Nada se generaliza a Heat, DFG ni piezo sin la prueba de estrés.
- N = 2 048 limita el rango de GN y subestima la cola del espectro; por eso se comparan razones en m ≤ 100.
- Un desplazamiento plano no prueba que precondicionar mejore la optimización; eso lo mide el EXP-0023.
- ρ_sr conserva la magnitud por parámetro pero no la estructura de signos; mide cuánto de lo plano se debe a combinaciones coherentes y no a la escala.

## Presupuesto (medido)

Solo CPU (2 núcleos): Burgers ≈ 52 s por estado, Allen–Cahn ≈ 14 s. 25 × 52 s + 15 × 14 s ≈ 25 min.

## Pruebas previas (hechas, `tests/test_exp0022_gn.py`, 7 pasan)

1. El residuo funcional coincide con el de la campaña (1e-10).
2. J contra diferencias finitas centrales en una dirección aleatoria.
3. El espectro por la Gram coincide con el de GN densa; el resto es nulo.
4. El bloque por arista coincide con el bloque de la matriz densa.
5. (B + εI)^(-1/2) blanquea el bloque (S B S = I con ε = 0).
6. El operador precondicionado tiene espectro finito y positivo.
7. Aislamiento por AST: el módulo solo importa `math`, `numpy` y `torch`, y no contiene referencias a la solución.

## Superficie de edición declarada

`experiments/mechanisms/EXP-0022-conditioning-diagnosis/` (gn.py, run_exp0022.py, analyze_exp0022.py, calibrate_thresholds.py, resultados), `tests/test_exp0022_gn.py`. Sin cambios en `evals/`, `registry/claims.jsonl`, baselines ni en experimentos anteriores.

## Firma

- [x] Preregistro: cantidades, umbrales (0.1 de H-COND-1; 0.1 y 0.33 de H-COND-2; 0.6 de la fila intermedia), m = 10 y 100, N = 2 048 y regla de decisión — Rodrigo Mansilla — fecha: 2026-10-20 (chat: "firmo")
