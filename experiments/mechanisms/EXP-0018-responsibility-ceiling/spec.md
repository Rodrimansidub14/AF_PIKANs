---
experiment_id: EXP-0018
slug: responsibility-ceiling
version: v2                # firmada el 2026-10-20; v1 (borrador) archivada en spec-v1-draft.md
status: scored          # draft | preregistered | running | scored | decided — solo el investigador lo pasa a `preregistered`
suite: tier2_candidate
problem_id: problem.burgers_pinnacle_1d (desarrollo), problem.allen_cahn_1d (réplica)
model_id: model.rbf_pikan (RBF48 en Burgers; RBF16 en Allen–Cahn), baseline v2
mechanism_id: mechanism.responsibility_focus (src/mechanisms/responsibility_focus.py, escrito y probado el 2026-10-20)
hypothesis_id: H-CEIL-1 (el techo existe), H-CEIL-2 (la responsabilidad informa más que un escalado arbitrario)
preregistered_at: 2026-10-20
preregistered_by: Rodrigo Mansilla (chat 2026-10-20: «Firmo el EXP-0018 v2 (criterios, umbrales 0.85/0.90 con 4 de 5, beta=1, q=0.5, guardias g1-g3) y apruebo el uso de la referencia como oráculo (oracle_ceiling)»; registrado por el agente)
decision_origin: EXP-0017 decision.md, opción 1, firmada por Rodrigo Mansilla el 2026-10-20 (chat: "Firmo opción 1")
---

# EXP-0018 — Techo causal del foco por responsabilidad (A* oráculo)

> **Terminología.** "PIKAN-PDE" = RBF-PIKAN entrenada con la pérdida física. Los brazos que usan A* leen la referencia: están etiquetados `oracle_ceiling`, no son candidatos y nunca entran a `claims.jsonl`.

Redactado el 2026-10-20 por un agente; revisado el mismo día por Claude Code (v2, ver «Cambios de la v2»). **No está firmado.** Los números marcados «propuesto» son del agente y el investigador los puede mover antes de firmar; después de firmar no se mueven.

## Origen y pregunta

El EXP-0017 no sostuvo H-LOC-1, H-ATTR-1 ni H-ATTR-2 (fila «nada pasa» del árbol firmado). Esa fila manda a medir **primero el techo oráculo**: A\* con el error real. La correlación descriptiva entre A\* y lo que L-BFGS movió (+0.26 y +0.28) no responde la pregunta causal, porque Δ es lo que L-BFGS movió y no lo que convenía mover.

**Pregunta.** Si desde el traspaso el paso de cada base se reparte según su responsabilidad *real* A\*, ¿baja el error final a igual presupuesto, y baja más que con un reparto que ignora la responsabilidad?

## Diseño

Partida común: el **estado final de Adam** de cada semilla (k = 1.00 de la etapa A del EXP-0016: 20 000 pasos en Burgers, 40 000 en Allen–Cahn), el mismo para todos los brazos. Burgers semillas 301–305 (estados ya guardados). Allen–Cahn semillas 601–605: 601–603 ya guardados; 604 y 605 se entrenan con la receta (790 s de Adam cada una en T4, determinista tras el calentamiento). Cada brazo corre solo la fase de refinamiento (L-BFGS) desde ese estado, con el mismo presupuesto de iteraciones y de reloj que el baseline. Pareado por semilla.

A\*_ℓig = Σ_x |u(x) − u_ref(x)| · φ_ℓig(h_ℓi(x)) / Σ_x φ_ℓig(h_ℓi(x)), calculada **una vez** en el estado de partida (estática), con la definición de unidad probada en el EXP-0017. El costo de calcularla no cuenta en el reloj (oráculo).

| Brazo | Qué hace | Papel |
|---|---|---|
| **B** | L-BFGS sin cambios | línea base |
| **P** | reparametriza los coeficientes por base, θ_ℓig = s_ℓig · θ′_ℓig, con s = (A\*/mediana A\*)^(β/2), β = 1 (equivale a precondicionar con diag(A\*^β)); s recortado a [0.1, 10] (propuesto, fijo) | techo del foco *suave* |
| **H** | L-BFGS solo sobre el 50 % de bases con mayor A\*; el resto congelado (q = 0.5, propuesto, fijo) | techo del foco *duro* |
| **N** | igual que P pero con A\* permutado al azar entre las bases de la misma capa (misma distribución de escalas, sin información) | control: ¿importa *quién* recibe la escala? |
| **O** | igual que P con la ocupación O en lugar de A\* | exploratorio, sin hipótesis |

**Por qué hay un brazo H.** L-BFGS es casi invariante a un escalado diagonal una vez que acumula historia: P podría dar nulo solo porque el método deshace la reparametrización, no porque la responsabilidad no sirva. H no es invariante. Si P falla y H pasa, el hallazgo es «el foco sirve pero no como precondicionamiento»; si ambos fallan, la familia se cierra.

Un solo β y un solo q, fijados aquí: no hay barrido ni se elige el mejor valor (lección del permutation p = 0.23 del EXP-0016).

## Hipótesis y criterios (vinculantes al firmar)

Métrica: error L2 relativo final tras L-BFGS (la misma que el baseline v2). Razón pareada r = error(brazo) / error(B) por semilla.

- **H-CEIL-1 (el techo existe).** En Burgers **y** Allen–Cahn, para P o para H: mediana de r ≤ 0.85 (propuesto: baja ≥ 15 %) y el brazo mejora en ≥ 4 de 5 semillas. Se prueban 2 brazos × 2 problemas con Holm; p por permutación con signo dentro de semilla (mínimo posible 1/32).
**Guardias (todas deben cumplirse; agregadas en la v2 por el §4.5 del protocolo):**
- g1: error L2 relativo en la **región crítica** (frente de Burgers: |x| ≤ 5ν, t ≥ 1/π; interfaz de Allen–Cahn: |u_ref| < 0.9; definida con la referencia, solo para puntuar) del brazo frente a B: razón en media geométrica ≤ 1.10. Mejorar el error global empeorando el frente no respalda la hipótesis.
- g2: reloj de L-BFGS del brazo ≤ 1.5 × el de B (por semilla, media geométrica).
- g3: todas las corridas con parámetros finitos.

- **H-CEIL-2 (la responsabilidad informa).** Para el brazo que pasó H-CEIL-1: su mediana de error frente a N (P frente a N, H frente a un H con 50 % de bases elegidas al azar, brazo N-H) cumple mediana de razón ≤ 0.90 y mejora en ≥ 4 de 5 semillas en ambos problemas.

(El brazo N-H se corre solo si H pasa H-CEIL-1; se declara aquí para que no sea una decisión posterior.)

**Regla de decisión estadística (aclaración de la v2).** Con n = 5, el p exacto por permutación con signo dentro de semilla vale como mínimo 1/32, y con la corrección de Holm sobre 4 pruebas el mínimo ajustado es 0.125: **ningún p puede ser < 0.05**. Por eso los criterios de arriba (mediana de la razón y 4 de 5 semillas) son la regla; los p se reportan, no deciden. La calibración de abajo mide qué tan seguro es ese procedimiento.

## Regla de decisión (vinculante)

| Resultado | Consecuencia |
|---|---|
| H-CEIL-1 y H-CEIL-2 se sostienen | La responsabilidad real guía el refinamiento. El experimento siguiente: sustituir A\* por un estimador sin referencia (residuo, y estimador de error) y medir cuánto del techo se conserva. Ese es el mecanismo candidato |
| H-CEIL-1 se sostiene, H-CEIL-2 no | El efecto es del escalado heterogéneo, no de la responsabilidad. Se reporta; el mecanismo, si existe, es de condicionamiento y no de representación |
| H-CEIL-1 falla en ambos brazos | Se cierra la familia «foco por responsabilidad por base» (estática, β = 1, q = 0.5). Se reporta como techo negativo; la ocupación (brazo O) queda como única señal sin explorar |
| Se sostiene en un problema y no en el otro | Se reporta como dependiente del problema; no se avanza al 0019 sin decisión del investigador |

## Cambios de la v2 respecto de la v1 (para aceptar o revertir al firmar)

1. Estado de partida precisado (final de Adam, k = 1.00) y semillas 604–605 de Allen–Cahn entrenadas aquí.
2. Guardias g1–g3 (región crítica, reloj, finitud) agregadas por el §4.5; la v1 no las tenía.
3. Aclaración de que ningún p puede decidir con n = 5 (Holm) y de que la regla es el tamaño del efecto y 4 de 5.
4. Calibración del ruido y de la tasa de falsos positivos (abajo); los umbrales de la v1 se mantienen.
5. Detalles de implementación fijados: mediana y umbral de H **globales** (todas las capas juntas); los parámetros `base` de la vía lineal de cada arista **no se escalan ni se congelan** en ningún brazo; A\* y la ocupación se calculan sobre la misma submalla fija de 8 192 puntos del EXP-0017; N usa una permutación reproducible por semilla y capa; N-H elige al azar el mismo número de bases que H.
6. Presupuesto con tiempos medidos y topes.

## Calibración (2026-10-20)

Ruido pareado medido en la etapa A del EXP-0016: desviación típica del log de la razón entre puntos finales de L-BFGS desde estados cercanos (k = 0.6 y 0.8 frente a k = 1.00): **0.71 en Burgers** (0.44–0.97 según k) y **0.38 en Allen–Cahn**. Es una cota superior del ruido de cambiar el precondicionador desde el *mismo* estado (aquí los estados difieren); el ruido desde el mismo estado no se ha medido. Simulación (`calibrate_thresholds.py`, 40 000 repeticiones, brazos independientes con ese ruido, 5 semillas por problema):

| Umbrales H-CEIL-1 / H-CEIL-2 | Falso positivo (techo sostenido) | Potencia si la razón verdadera es 0.6 / 0.7 / 0.8 |
|---|---|---|
| **0.85 / 0.90 (v1, se mantiene)** | **0.9 %** (3.7 % solo H-CEIL-1) | 53 % / 28 % / 11 % |
| 0.80 / 0.85 | 0.6 % | 50 % / 24 % / 9 % |
| 0.75 / 0.85 | 0.4 % | 47 % / 21 % / 7 % |

Lectura: endurecer los umbrales no compra nada (casi no baja los falsos positivos y sí la potencia). Con 5 semillas el experimento solo ve con fiabilidad reducciones ≳ 40 %; una reducción del 20 % se detecta con probabilidad ≈ 10 %. Un nulo no descarta un efecto pequeño. Con 10 semillas por problema y exigencia 8/10, la potencia *baja* (37 % a 0.6), porque la regla 8/10 es más dura que 4/5.

**Expectativa declarada antes de correr.** Efecto pequeño o nulo: en el EXP-0017 A\* correlacionó menos con lo que L-BFGS movió (+0.26 y +0.28) que la ocupación (+0.46 y +0.51), y en Burgers el error real no se concentra por base (el 20 % superior de A\* concentra 0.36). La pregunta informativa es si P o H bajan el error más que sus controles permutados (N, N-H).

## Límites declarados antes de correr

- A\* es estática: se calcula una vez en el estado de partida y L-BFGS mueve el estado mucho después. Un nulo no descarta una A\* que se refresque.
- n = 5 por problema: el p exacto por semilla solo alcanza 0.031 si las cinco semillas favorecen al brazo. El criterio descansa en el tamaño del efecto y en 4 de 5, no solo en p.
- Los umbrales 0.85 y 0.90 son propuestos; se calibraron con ruido medido (ver «Calibración») y **no se cambian**: la calibración muestra que ya son prudentes frente a falsos positivos (0.9 %) y que lo que limita es la potencia, no la exigencia. Solo se detectan con fiabilidad reducciones de error del orden de 40 % o más.
- Solo capas 0–2 de las que existen; solo dos problemas. La prueba de estrés en los seis problemas ocurre solo si hay candidato (0019).
- No corrige la causalidad de Δ del EXP-0017; la sustituye por una medición de efecto sobre el error.

## Presupuesto

Kaggle T4×2, privado. Estimación **con tiempos medidos** del control (L-BFGS desde el estado final de Adam, la misma fase de cada brazo): Burgers 131–208 s por corrida (2 280–3 558 iteraciones, siempre por meseta), Allen–Cahn 98–127 s (2 596–3 406 iteraciones); Adam de Allen–Cahn ≈ 790 s por semilla nueva.

| Concepto | Corridas | Tiempo estimado (T4) |
|---|---|---|
| Burgers, 5 semillas × 5 brazos (B, P, H, N, O) | 25 | ≈ 4 200 s |
| Allen–Cahn, 5 semillas × 5 brazos | 25 | ≈ 2 800 s |
| Adam de Allen–Cahn, semillas 604 y 605 | 2 | ≈ 1 600 s |
| **Subtotal (sin N-H)** | | **≈ 8 600 s ≈ 2.4 h-GPU ≈ 1.2 h de sesión en dos GPU** |
| N-H, solo si H pasa H-CEIL-1 (10 corridas) | 10 | ≈ 1 400 s más |

Los brazos con foco pueden tardar más que B en llegar a la meseta (no medido): con un factor 2 el total sería ≈ 2.4 h de sesión. **Tope duro 3 h**, mismo mecanismo que la etapa A del EXP-0016 (no se despacha trabajo nuevo después de 2.5 h; se matan los trabajos a las 3 h). Orden de la cola: Burgers, luego Allen–Cahn.

## Pruebas previas (Tier 0/1, CPU) — estado al 2026-10-20

| # | Prueba | Estado |
|---|---|---|
| 1 | La reparametrización de P es exacta: con s ≡ 1 reproduce a B **bit a bit** (L-BFGS completo); el gradiente respecto de θ′ es s · gradiente respecto de θ; la función no cambia al reparametrizar | **hecha** (`tests/test_responsibility_focus.py`). Hallazgo: la implementación con `torch.nn.utils.parametrize` **no** da igualdad bit a bit, porque cambia el orden de los parámetros y con él el de las sumas de L-BFGS; se reimplementó reemplazando el forward de la arista, que conserva el orden |
| 2 | H congela de verdad: los coeficientes de las bases congeladas no cambian tras L-BFGS y los demás sí | **hecha** |
| 3 | N conserva la distribución de s de P (mismos valores, permutados dentro de capa, reproducible por semilla) | **hecha** |
| 4 | A\* del ejecutor coincide con el calculado en el EXP-0017 para los mismos estados | **hecha** para Burgers 301 y Allen–Cahn 601 (diferencia ≤ 1e-12 con el mismo número de hilos; con otro número de hilos aparece 1e-9 por el orden de las sumas en float32) |
| 5 | Aislamiento: el mecanismo no importa `evals/`; ninguna función recibe una referencia; la referencia entra solo por `oracle_responsibility` y por la puntuación | **hecha** |
| 6 | Ejecutor de punta a punta en CPU (red chica): seis ramas, B idéntico al L-BFGS de la receta, ramas distintas entre sí | **hecha** (`tests/test_ceiling_executor.py`, 4 pruebas) |
| 7 | En T4 tras el calentamiento, B reproduce bit a bit el k = 1.00 de la etapa A del EXP-0016 | **hecha el 2026-10-20** (`results/precheck7/`): Burgers 301 y Allen–Cahn 601, hash del estado final idéntico al de la etapa A y mismas iteraciones (2 972 y 2 940); el error global coincide salvo 1.4e-16 relativo en el último bit de la norma. Primer intento fallido por dos supuestos de CPU en el ejecutor (tensor del error del oráculo y tensores de escala/máscara creados en CPU); corregidos con pruebas de regresión |

## Superficie de edición declarada

Nuevo: `src/mechanisms/responsibility_focus.py`, `tests/test_responsibility_focus.py`, `experiments/mechanisms/EXP-0018-responsibility-ceiling/` (scripts y resultados). Sin cambios en `evals/`, `registry/claims.jsonl`, baselines ni en el EXP-0016/0017.

## Firma

- [x] Preregistro: criterios, umbrales (0.85, 0.90, 4/5), β = 1, q = 0.5, guardias g1–g3 y regla de decisión — Rodrigo Mansilla — fecha: 2026-10-20
- [x] Uso de la referencia como oráculo (brazos P, H, N, N-H con A\*), solo con etiqueta `oracle_ceiling` — Rodrigo Mansilla — fecha: 2026-10-20
