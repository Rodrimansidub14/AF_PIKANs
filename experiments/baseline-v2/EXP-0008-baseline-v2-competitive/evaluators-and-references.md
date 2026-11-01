# Evaluadores y referencias (puntos 2 y 5)

Estado: borrador de EXP-0008. Propone; no acepta referencias ni modifica
`evals/`. Toda pieza nueva del verificador la escribe el agente **como
borrador en `var/agent_runs/exp0008/scorers/`** y la incorpora el humano a
`evals/` (gate `eval_integrity`).

## 1. Tres capas de evaluación

| Capa | Qué es | Quién la ve | Uso |
|---|---|---|---|
| **V** | grilla de validación fija 129×65 (`evals.checkpoint_validation`), separada de las mallas finales | etapa A y diagnósticos | selección de receta y rung |
| **E** | malla final del problema (`eval_grid` de `problem.yaml`): Heat 512×256, Burgers CH 1025×129, AC 512×256, DFG 440×82 + fronteras | job de scoring tras congelar | confirmación (gates G1/G2) |
| **Externa** | evaluador de otra fuente que no intervino en nada | job de scoring tras congelar | comparabilidad con la literatura |

Reglas: (1) el entrenamiento no importa `evals`; (2) la selección usa solo V y
semillas dev; (3) E y externos se puntúan una vez por checkpoint declarado, en
un job separado; (4) ningún gate se evalúa sobre el resultado de V.

**Transparencia sobre E.** E se ha usado para puntuar v1 y las campañas de
cierre. No se ha seleccionado ninguna receta viendo un ranking de E, pero la
exposición existe (ver `spec.md §10`). Los evaluadores externos son la
defensa: se fijan de antemano y no dependen de nada del proyecto.

| Problema | V | E | Externo(s) |
|---|---|---|---|
| Heat | 129×65 | analítica 512×256 | malla tutorial DeepXDE 256×201 |
| Burgers | 129×65 | Cole–Hopf (Gauss–Hermite 128) 1025×129 | FEM PINNacle 101×11 (`burgers1d.dat`) |
| Allen–Cahn | 129×65 | ETDRK4 y FD8192 candidatas 512×256 | `.mat` DeepXDE 201×101 (interpolado) |
| DFG | QoI + residuales | 440×82 + 512 pts/segmento | FEATFLOW (Cd, Cl, Δp) y, cuando exista, FEM Taylor–Hood |

## 2. WP-E1 — puntuar el RBF en el FEM 101×11 de PINNacle (CPU local)

**Objetivo (H6).** Hacer comparable el número de Burgers con la tabla del
paper y cuantificar cuánto discrimina ese evaluador.

**Hallazgo previo que condiciona el uso.** En la reproducción, el mismo MLP
puntuado con dos evaluadores da: Adam solo → CH 9.08e-3 vs FEM 1.72e-2;
Adam→L-BFGS → CH 5.66e-4 vs FEM 1.3264e-2 (DE entre semillas 2.9e-5). El FEM
101×11 se comporta como un **piso ≈ 1.3e-2** (discrepancia entre el archivo y
Cole–Hopf): con `FEM² ≈ CH² + piso²`, ese evaluador no distingue modelos por
debajo de ~1.3e-2. Consecuencia para los gates: la métrica **vinculante** de
Burgers es CH; el FEM 101×11 sirve para la comparabilidad con la tabla
publicada y se reporta junto con su piso. (WP-E1 mide el piso directamente:
`‖u_FEM − u_CH‖/‖u_CH‖` en los 1 111 puntos.)

**Resultado medido (WP-E1, 2026-09-28, CPU local; `var/agent_runs/exp0008/wp-e1/`).**
Hash del archivo verificado (`3b6d31d8…`). Piso `‖u_FEM − u_CH‖/‖u_CH‖` en los 1 111
puntos = **0.013298**, prácticamente igual al 0.0133 que publica PINNacle para su variante L-BFGS:
ese número publicado es esencialmente el error del archivo de referencia, no del modelo.

| Conjunto de checkpoints (Adam v1) | n | CH 1025×129 | FEM 101×11 |
|---|---:|---|---|
| RBF Sobol 201–205 | 5 | 3.954e-2 ± 2.56e-2 | 4.97e-2 ± 2.53e-2 |
| RBF RAD-v1 1201–1205 | 5 | 0.2124 ± 0.0752 | 0.1948 ± 0.0782 |
| RBF EXP-0005 101–103 | 3 | 5.40e-2 ± 3.93e-2 | 5.99e-2 ± 3.45e-2 |
| MLP 58 EXP-0005 101–103 | 3 | 0.1157 ± 0.0241 | 0.1073 ± 0.0201 |

H6 se cumple: para v1 el cociente FEM/CH es ≈ 1.26 (dentro de 1×–30×). Los CH reproducen
los valores de `paired-runs.csv` a 1e-15, lo que valida el scorer.

**Procedimiento.**

1. Lectura externa (solo lectura, se anuncia antes de ejecutar):
   `https://raw.githubusercontent.com/i207M/PINNacle/6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf/ref/burgers1d.dat`,
   hash esperado `3b6d31d895022b397330091ace59220167f0918c2ed616ed2ae5b1458eede5eb`.
   Si el hash no coincide, se detiene.
2. Evaluar el modelo en float64 en cada `(x,t)` del archivo; `L2RE = ‖u−u_ref‖₂/‖u_ref‖₂`
   (definición del evaluador de PINNacle).
3. Puntuar con CPU los checkpoints ya existentes: Sobol v1 201–205 y RAD
   1201–1205 (`checkpoint-020000.pt`, presentes localmente en
   `baseline-competitive-2026-09-27/kaggle-cli-fetch-20260927-02/burgers-rerun-v4/…`),
   y los MLP/B-spline de EXP-0005 si sus checkpoints están disponibles.
4. Calcular en los mismos 1 111 puntos el error CH (consistencia con
   `evals`) y el piso FEM-vs-CH.
5. Prueba de sanidad (equivalente a `evals/meta`): el scorer aplicado al propio
   archivo da 0; aplicado a CH da el piso; una perturbación conocida de la
   solución exacta da el error esperado.

**Salida:** tabla por semilla y agregada (CSV + JSON), `reference_accepted=false`,
hashes de entrada. Coste: segundos de CPU.

## 3. WP-E3 — cerrar el scorer σn de DFG (CPU local)

**Problema observado.** Las cohortes 301–305 (RBF, post-L-BFGS: Cd 1.463, error
73.8 %) y 501–505 (RBF: Cd 5.236, error 6.2 %; MLP 5.29) usan la misma
arquitectura y residuales held-out casi iguales (~0.018–0.019), pero Cd
difiere 12×. El scorer de fuerzas se corrigió el 2026-09-28 (tracción `σn`,
`σ = ν∇u − pI`, normal radial saliente; antes se usaba una reducción
tangencial no coincidente con FEATFLOW) y los workers de 501–505 son del
2026-09-27. No consta cuál scorer produjo cada cifra.

**Procedimiento.**

1. Puntuar **los mismos checkpoints** de ambas cohortes (pre y post L-BFGS, RBF
   y MLP) con el `dfg.py` actual, en CPU/float32, y comparar con las cifras
   archivadas. Checkpoints de 501–505 y 301–305 se toman de
   `kaggle-outputs/dfg-lbfgs-refinement/…` y `dfg-rbf-current-cohort-refinement/…`.
2. Ejecutar la prueba analítica de regresión de `dfg.py` (campo con `σn`
   conocido) y una prueba de **signo/orientación** de la normal.
3. Descomponer Cd en aportes de presión y viscoso por cohorte y comprobar el
   balance de caudal entrada/salida (no requiere referencia).
4. Verificar que la cuadratura 512/1 024 no cambia el resultado (ya medido:
   1.6e-7 en Cd).
5. Entregar tabla `cohorte × arm × scorer → (Cd, Cl, Δp, error)`.

**Decisión que habilita (H5).** Si con un solo scorer las cohortes coinciden
(≤2×), el resultado bueno de 501–505 se vuelve el baseline v1 de DFG y v2 se
corre para cerrar G2. Si siguen difiriendo, aparece una causa de
entrenamiento a aislar antes de gastar GPU en DFG.

## 4. Referencia FEM de DFG (sin GPU)

El FEM de referencia es cálculo **de CPU**; lanzarlo en sesiones GPU gastó cuota
sin necesidad (S1b: 94 s de GPU en dos intentos fallidos + agotamiento de la
reserva). Se proponen tres rutas, en orden de preferencia; la decisión es del
investigador (aprobación #6 del `spec.md`).

| Ruta | Herramientas | Ventaja | Riesgo |
|---|---|---|---|
| **A (recomendada)** | scikit-fem + `gmsh` (wheel de pip, Windows) + `scipy.sparse`; Taylor–Hood P2/P1, geometría del cilindro isoparamétrica P2 | probada en **local**, sin contenedor; falla visible en D0 | la geometría curva exige mapeo isoparamétrico; con malla recta el cilindro poligonal sesga Cd |
| B | kernel Kaggle **CPU** con `micromamba`, `dolfinx` + `python-gmsh` | DOLFINx maduro | el módulo Python de gmsh parece ser un paquete conda distinto (`python-gmsh`) y el fallo `No module named 'gmsh'` es consistente con eso: **hipótesis a verificar en D0/D5**, no hecho |
| C | WSL2/Docker + DOLFINx | idem B, reproducible | requiere instalar WSL/Docker |

**Contrato FEM (ya declarado en `evals/problems/dfg_2d/problem.yaml`):**
`ν=1e-3`, `ρ=1`, canal 2.2×0.41, cilindro `r=0.05` en `(0.2,0.2)`, entrada
parabólica máx 0.3, no-slip en paredes/cilindro, salida *do-nothing*. QoI:
`Cd`, `Cl` por integral de `σn` sobre el cilindro (coeficientes con `2/(ρ Ū² D)`,
`Ū=0.2`, `D=0.1`), `Δp = p(0.15,0.2) − p(0.25,0.2)`. Exportar `u,v,p` en la
malla fija 440×82 sin el interior del cilindro; `p` con la constante fijada
por la salida.

**Convergencia (evidencia, no aceptación).** Tres niveles como mínimo con
refinamiento cerca del cilindro. Criterios de `problem.yaml`: cambio de campo
entre los dos niveles finos `< 0.005`, cambio de QoI `< 0.001`, cambio
siguiente `< 10 %` del efecto. Comparación con FEATFLOW
(`Cd 5.57953523384`, `Cl 0.010618948146`, `Δp 0.11752016697`): dentro de
0.5 % / 5e-4 abs / 0.5 % como evidencia adicional.

**Meta-pruebas del scorer FEM** (como `evals/meta`): solución de Poiseuille
sin cilindro (QoI conocido), y verificación de que el mismo scorer de tracción
sobre el FEM da los QoI reportados.

## 5. Aceptación de referencias (humana)

El agente reúne evidencia; el humano decide. Para cada referencia se entrega
una ficha con procedencia, hashes, convergencia y un **criterio de
adecuación** propuesto: *la incertidumbre de la referencia (discrepancia entre
solvers independientes) debe ser ≤ 1 % del umbral más estricto que se pretende
resolver con ella*.

| Referencia | Evidencia disponible | Adecuada si… | Pendiente |
|---|---|---|---|
| Burgers CH (Gauss–Hermite 128) | analítica; ya en `evals/` y verificada por `evals/meta` | — | ninguno |
| Allen–Cahn ETDRK4 | S1a: vs FD8192 `3.84e-7` (rel. ponderado), máx. abs `5.5e-6`; auto-convergencia `9.3e-13` (N) y `1.3e-11` (dt) | `3.84e-7 ≤ 1 %·2e-3 = 2e-5` ✔ | revisión independiente de procedencia y del código |
| Allen–Cahn FD/RK4 8192 | idem; referencia de los scores S3 | idem | idem |
| Allen–Cahn BDF AC-L4 | cambio espacial `4.6e-6`, temporal `2.4e-10` | idem | cierre del criterio de "cambio vs efecto" |
| `.mat` DeepXDE 201×101 | interpolado a 512×256: diferencia `1.57e-4` con ETDRK4 | como **evaluador externo**, con piso `1.6e-4` (incluye interpolación) | ninguno (se reporta el piso) |
| DFG FEM Taylor–Hood | inexistente | criterios de §4 | implementación y ejecución |
| Heat analítica | `evals/` | — | ninguno |

`reference_accepted` lo escribe el humano; ningún artefacto del agente lo
pone en `true`.

## 6. Lo que este documento no hace

No cambia grillas, umbrales ni graders existentes; no incorpora un scorer a
`evals/`; no usa el FEM 101×11 para seleccionar receta; no declara aceptado
ningún score de referencia candidata.
