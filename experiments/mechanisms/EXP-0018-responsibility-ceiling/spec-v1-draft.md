---
experiment_id: EXP-0018
slug: responsibility-ceiling
version: v1-draft
status: draft          # draft | preregistered | running | scored | decided — solo el investigador lo pasa a `preregistered`
suite: tier2_candidate
problem_id: problem.burgers_pinnacle_1d (desarrollo), problem.allen_cahn_1d (réplica)
model_id: model.rbf_pikan (RBF48 en Burgers; RBF16 en Allen–Cahn), baseline v2
mechanism_id: mechanism.responsibility_focus (src/mechanisms/responsibility_focus.py, por escribir y probar)
hypothesis_id: H-CEIL-1 (el techo existe), H-CEIL-2 (la responsabilidad informa más que un escalado arbitrario)
decision_origin: EXP-0017 decision.md, opción 1, firmada por Rodrigo Mansilla el 2026-10-20 (chat: "Firmo opción 1")
---

# EXP-0018 — Techo causal del foco por responsabilidad (A* oráculo)

> **Terminología.** "PIKAN-PDE" = RBF-PIKAN entrenada con la pérdida física. Los brazos que usan A* leen la referencia: están etiquetados `oracle_ceiling`, no son candidatos y nunca entran a `claims.jsonl`.

Redactado el 2026-10-20 por un agente. **No está firmado.** Los números marcados «propuesto» son del agente y el investigador los puede mover antes de firmar; después de firmar no se mueven.

## Origen y pregunta

El EXP-0017 no sostuvo H-LOC-1, H-ATTR-1 ni H-ATTR-2 (fila «nada pasa» del árbol firmado). Esa fila manda a medir **primero el techo oráculo**: A\* con el error real. La correlación descriptiva entre A\* y lo que L-BFGS movió (+0.26 y +0.28) no responde la pregunta causal, porque Δ es lo que L-BFGS movió y no lo que convenía mover.

**Pregunta.** Si desde el traspaso el paso de cada base se reparte según su responsabilidad *real* A\*, ¿baja el error final a igual presupuesto, y baja más que con un reparto que ignora la responsabilidad?

## Diseño

Partida común: los estados de Adam ya guardados de la etapa A del EXP-0016 en el traspaso del baseline v2 (el mismo k para todos los brazos). Burgers semillas 301–305. Allen–Cahn semillas 601–605: 604 y 605 requieren correr Adam (solo 601–603 existen). Cada brazo corre solo la fase de refinamiento (L-BFGS) desde ese estado, con el mismo presupuesto de iteraciones y de reloj que el baseline. Pareado por semilla.

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
- **H-CEIL-2 (la responsabilidad informa).** Para el brazo que pasó H-CEIL-1: su mediana de error frente a N (P frente a N, H frente a un H con 50 % de bases elegidas al azar, brazo N-H) cumple mediana de razón ≤ 0.90 y mejora en ≥ 4 de 5 semillas en ambos problemas.

(El brazo N-H se corre solo si H pasa H-CEIL-1; se declara aquí para que no sea una decisión posterior.)

## Regla de decisión (vinculante)

| Resultado | Consecuencia |
|---|---|
| H-CEIL-1 y H-CEIL-2 se sostienen | La responsabilidad real guía el refinamiento. EXP-0019: sustituir A\* por un estimador sin referencia (residuo, y estimador de error) y medir cuánto del techo se conserva. Ese es el mecanismo candidato |
| H-CEIL-1 se sostiene, H-CEIL-2 no | El efecto es del escalado heterogéneo, no de la responsabilidad. Se reporta; el mecanismo, si existe, es de condicionamiento y no de representación |
| H-CEIL-1 falla en ambos brazos | Se cierra la familia «foco por responsabilidad por base» (estática, β = 1, q = 0.5). Se reporta como techo negativo; la ocupación (brazo O) queda como única señal sin explorar |
| Se sostiene en un problema y no en el otro | Se reporta como dependiente del problema; no se avanza al 0019 sin decisión del investigador |

## Límites declarados antes de correr

- A\* es estática: se calcula una vez en el estado de partida y L-BFGS mueve el estado mucho después. Un nulo no descarta una A\* que se refresque.
- n = 5 por problema: el p exacto por semilla solo alcanza 0.031 si las cinco semillas favorecen al brazo. El criterio descansa en el tamaño del efecto y en 4 de 5, no solo en p.
- Los umbrales 0.85 y 0.90 son propuestos y no vienen de la varianza medida del baseline. **Antes de firmar**, el investigador (o Claude Code) puede calibrarlos con el ruido pareado entre semillas del baseline v2 que ya existe.
- Solo capas 0–2 de las que existen; solo dos problemas. La prueba de estrés en los seis problemas ocurre solo si hay candidato (0019).
- No corrige la causalidad de Δ del EXP-0017; la sustituye por una medición de efecto sobre el error.

## Presupuesto

GPU. Estimación: del orden de la etapa A del EXP-0016 (1.85 h), 5 brazos × 10 semillas más N-H si aplica. A reestimar tras las pruebas previas.

## Pruebas previas (Tier 0/1, CPU)

1. La reparametrización de P es exacta: con s ≡ 1 reproduce a B bit a bit; el gradiente respecto de θ′ es s · gradiente respecto de θ.
2. H congela de verdad: los coeficientes de las bases congeladas no cambian tras L-BFGS.
3. N conserva la distribución de s de P (mismos valores, permutados dentro de capa).
4. A\* de este experimento coincide con el A\* calculado en el EXP-0017 para los mismos estados (diferencia máxima 0).
5. Aislamiento: el mecanismo no importa `evals/`; la referencia entra solo por la entrada etiquetada `oracle_ceiling`.

## Superficie de edición declarada

Nuevo: `src/mechanisms/responsibility_focus.py`, `tests/test_responsibility_focus.py`, `experiments/mechanisms/EXP-0018-responsibility-ceiling/` (scripts y resultados). Sin cambios en `evals/`, `registry/claims.jsonl`, baselines ni en el EXP-0016/0017.

## Firma

- [ ] Preregistro: criterios, umbrales (0.85, 0.90, 4/5), β = 1, q = 0.5 y regla de decisión — Rodrigo Mansilla — fecha: ____
- [ ] Uso de la referencia como oráculo (brazos P, H, N, N-H con A\*), solo con etiqueta `oracle_ceiling` — Rodrigo Mansilla — fecha: ____
