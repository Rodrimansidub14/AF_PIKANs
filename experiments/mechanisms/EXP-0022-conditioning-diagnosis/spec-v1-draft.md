---
experiment_id: EXP-0022
slug: conditioning-diagnosis
version: v1-draft
status: draft          # draft | preregistered | running | scored | decided — solo el investigador lo pasa a `preregistered`
suite: tier1_regression  # solo CPU, sobre estados ya guardados; sin entrenamiento nuevo
problem_id: problem.burgers_pinnacle_1d (principal), problem.allen_cahn_1d (réplica)
model_id: model.rbf_pikan (RBF48 en Burgers; RBF16 en Allen–Cahn), pares (Adam, L-BFGS) de la etapa A del EXP-0016
mechanism_id: none (diagnóstico; decide si el EXP-0023 se corre)
hypothesis_id: H-COND-1 (el refinamiento se mueve por direcciones planas), H-COND-2 (el bloque por arista captura el mal condicionamiento)
---

# EXP-0022 — ¿Qué estructura del problema de optimización frena a la PIKAN-PDE? Diagnóstico de condicionamiento

> **Terminología.** "PIKAN-PDE" = RBF-PIKAN entrenada con la pérdida física. Ninguna cantidad de este experimento lee la referencia.

Redactado el 2026-10-20 por un agente. **No está firmado.** Los números marcados «propuesto» se pueden mover antes de firmar.

## Origen y pregunta

Sabemos que en 4 de 6 problemas el error de la PIKAN-PDE es de optimización, no de capacidad. Los EXP-0015 a 0017 probaron remedios (dónde y cuándo refinar) sin nombrar la patología. **Pregunta:** ¿el problema de optimización está mal condicionado, y esa mala condición vive en una estructura que la propia representación nos da (los bloques por arista)? Si sí, existe un mecanismo derivado y no buscado: precondicionar con la estructura de la representación (EXP-0023).

## Cantidades (fijadas antes de correr)

Sobre una malla fija de 8 192 puntos de colocación (la misma submalla del EXP-0017), con la pérdida del baseline v2:

- **GN** = JᵀJ / N, con J = ∂r/∂θ, r el residuo de la PDE por punto, θ todos los parámetros. Se aplica sin formarla (productos J·v y Jᵀ·w con `torch.func`).
- **Espectro.** Lanczos con reortogonalización completa, 300 iteraciones. Se reportan los m = 10, 100 y 1 000 mayores valores propios con su residuo de convergencia (se acepta un valor si el residuo relativo es < 1e-3; si no, se reporta hasta el último convergido). Más una densidad espectral por cuadratura de Lanczos estocástica (SLQ, 10 vectores).
- **κ_eff(m) = λ_1 / λ_m.** Se usa λ_m y no λ_min porque con más parámetros que puntos efectivos GN tiene núcleo, y λ_min no dice nada.
- **Desplazamiento** d = θ_L-BFGS − θ_Adam en cada par. **Cociente de Rayleigh relativo** ρ_d = (dᵀ GN d / dᵀd) / λ_1 en el estado de Adam.
- **Precondicionadores comparados** (se aplican a GN como P^(-1/2) GN P^(-1/2), y se recalcula κ_eff):
  - *Jacobi*: P = diag(GN).
  - *Por arista (primario)*: P = bloque diagonal con el bloque 11×11 exacto de GN de los coeficientes de cada arista, más ε·I con ε = 1e-3 × media de sus valores propios (propuesto). La rama `base` de cada arista y el resto de parámetros entran con su valor Jacobi.
  - *Por capa* (descriptivo): bloque por capa de entrada, sin selección posterior.
- **Gram de la base por arista** G_ℓi = Σₓ φ(h_ℓi(x)) φ(h_ℓi(x))ᵀ (11×11), número de condición y rango efectivo por capa, contra el mismo cálculo con entradas uniformes en [−1.05, 1.05]. Separa el solape de las gaussianas (propiedad de la base) de la concentración de las activaciones (propiedad de la representación entrenada).
- **Localización**: fracción de la energía de d que cae en el 20 % de las aristas con más energía, por capa.

## Hipótesis y criterios (vinculantes al firmar)

Mediana sobre los 40 pares (Burgers 25, Allen–Cahn 15), en ambos problemas.

- **H-COND-1.** ρ_d ≤ 0.05 (propuesto): L-BFGS mueve la red por direcciones de curvatura menor al 5 % de la máxima. Si L-BFGS se mueve por direcciones de curvatura alta, el cuello de Adam no es el condicionamiento.
- **H-COND-2.** κ_eff(m = 100) con el precondicionador *por arista* ≤ 0.1 × el crudo **y** ≤ 0.33 × el de Jacobi (propuesto). Lo segundo es el control importante: Adam ya hace algo parecido a Jacobi, así que una mejora que Jacobi también da no justifica un mecanismo nuevo.

Comparaciones pareadas por estado con permutación con signo dentro de semilla (10 000), reportadas junto al criterio de mediana. Con 5 y 3 semillas el p por semilla solo informa; manda la mediana.

## Regla de decisión (vinculante)

| Resultado | Consecuencia |
|---|---|
| H-COND-1 y H-COND-2 se sostienen | El EXP-0023 pasa a la fila de firma: el mal condicionamiento está y el bloque por arista lo captura. Mecanismo candidato derivado |
| H-COND-1 se sostiene, H-COND-2 no | Hay mal condicionamiento pero no lo captura el bloque por arista. Se mira el reporte descriptivo por capa; el EXP-0023 se rediseña con otra estructura de bloque o se cierra la ruta |
| H-COND-1 no se sostiene | La ruta de condicionamiento se cierra. Adam se frena por otra causa; se reporta como hallazgo y el hilo pasa al EXP-0018 y a lo que él decida |

## Límites declarados antes de correr

- GN es una aproximación local en el estado de Adam; no describe la trayectoria entera.
- Los estados existentes cubren Burgers y Allen–Cahn; sin Heat, DFG ni piezo. Un resultado de condicionamiento aquí no se generaliza a los otros problemas sin la prueba de estrés.
- La ancha de las gaussianas depende de la configuración real de cada experimento; hay que leerla del código del experimento, no de este spec.
- Lanczos recupera bien los extremos y mal el interior; por eso κ_eff usa λ_m con m fijo y se reporta la convergencia.
- Correlación entre curvatura y desplazamiento no prueba que precondicionar mejore la optimización; eso lo mide el EXP-0023.

## Presupuesto

Solo CPU. Estimación 1–3 h (productos de GN dominan en Burgers, 29 376 parámetros); medir con un solo estado antes de lanzar los 40.

## Pruebas previas (Tier 0/1)

1. Producto GN·v contra la matriz densa de GN en una red pequeña (error relativo < 1e-8).
2. Lanczos contra `numpy.linalg.eigh` en esa misma red.
3. El bloque 11×11 por arista coincide con el bloque de la matriz densa.
4. Aislamiento: nada importa `evals/` ni lee la referencia.

## Superficie de edición declarada

Nuevo: `experiments/mechanisms/EXP-0022-conditioning-diagnosis/` (scripts y resultados), `tests/test_exp0022_gn.py`. Sin cambios en `evals/`, `registry/claims.jsonl`, baselines ni en el EXP-0016 a 0018.

## Firma

- [ ] Preregistro: cantidades, umbrales (0.05, 0.1, 0.33, m = 100) y regla de decisión — Rodrigo Mansilla — fecha: ____
