---
experiment_id: EXP-0013
slug: run-diagnosis
status: decided         # draft | preregistered | running | scored | decided
suite: analysis_only (CPU)
problem_id: transversal (heat_1d, burgers_pinnacle_1d, allen_cahn_1d, ns2d_cg, dfg_2d, piezo_bimorph_indirect_2d)
model_id: model.rbf_pikan (y MLP pareados, como control)
mechanism_id: none (este estudio define candidatos; no prueba ninguno)
hypothesis_id: H-DIAG-1
preregistered_at: 2026-10-05
preregistered_by: Rodrigo Mansilla (investigador; firma en el chat: "firmado")
---

# EXP-0013 — Diagnóstico de qué hace cada corrida, para definir un mecanismo de *awareness*

Redactado el 2026-10-05 por un agente a partir de las instrucciones del investigador. **Firmado el 2026-10-05** por el investigador; desde ese momento el criterio de señal importante y la regla de falsación no se mueven.

## Pregunta central del proyecto (palabras del investigador)

**¿Y si el mecanismo supiera dónde poner su esfuerzo de manera automática?**

Los enfoques actuales (RAD, RAR-D, extensión de grilla dirigida) focalizan en puntos o zonas que se les indican con una regla fija. Aquí se busca *awareness*: que el modelo **decida solo** dónde, cuándo y en qué concentrar el esfuerzo (puntos, bases, términos de la pérdida, presupuesto entre Adam y L-BFGS). No es "atención" sobre una región dada. Puede que no se consiga en esta tesis; el objetivo es un **escalón (stepping stone)** hacia eso. Si el mecanismo resultante se parece a una arquitectura o técnica existente de otro dominio, eso es el mecanismo: la adaptación de un enfoque externo a una batería de problemas.

Este estudio **no elige el mecanismo por teoría**: diagnostica qué hizo cada corrida y qué señales, calculables sin la referencia, informan sobre dónde y cuándo el modelo falla. De ahí salen las familias candidatas.

## Alcance y restricciones

- **Solo CPU, sobre checkpoints y registros ya guardados. No se relanza ninguna campaña.**
- GPU únicamente para el bloque de **confirmación con semillas nuevas, si hace falta** (ver "Confirmación"), con tope de **1.5 h de sesión** y reserva mínima de **4 h** de cuota (hoy quedan ~5.7 h).
- No se modifica `evals/`, `registry/claims.jsonl`, `src/mechanisms/` ni los baselines v1/v2.
- Exploratorio: `scientific_accepted=false`. Nada de esto es un claim.

## Unidad de análisis y regla de separación

Unidad: **corrida** = problema × brazo (RBF/MLP) × semilla × etapa (fin de Adam / tras L-BFGS).

| Categoría | Qué es | Puede usarla un mecanismo |
|---|---|---|
| **Señal (rasgo)** | calculable durante el entrenamiento y **sin referencia** | sí |
| **Etiqueta (resultado)** | necesita la referencia (error de campo, error por región, QoI) | **no**, solo sirve para evaluar la señal |

Las etiquetas nunca entran a un modelo predictivo como característica para decidir un mecanismo; solo son la variable a explicar.

## Preguntas diagnósticas (por problema)

Q1. **Dónde está el error y dónde está el residuo.** ¿El residuo (o su concentración) predice la ubicación del error? Si no, un mecanismo guiado por residuo no puede funcionar en ese problema. Indicio previo (Burgers): el frente concentra el error (~29×) y solo 5–8 % de la masa del residuo.
Q2. **Capacidad u optimización.** ¿El error final es un límite de capacidad (el modelo no puede representar la solución) o de optimización (podría, pero no llega)? Se usa el ajuste directo a la referencia como prueba de capacidad donde exista (ya hecho para DFG: RBF de 4 032 parámetros llega a 1–5 %).
Q3. **Qué cambia de Adam a L-BFGS.** ¿Qué reparte L-BFGS (dispersión entre semillas, residuo local, términos de la pérdida)? ¿Cuándo llega a plateau y cuándo no aprende el fenómeno?
Q4. **Modo de falla.** Clasificar cada corrida: estado nulo (piezo), piso de discretización (Heat), choque sin resolver (Burgers), deriva de presión/caudal (DFG), plateau prematuro, otro.
Q5. **Awareness de bases.** ¿El ancho de la estructura difícil (frente, interfaz, capa límite) es menor que el espaciado/ancho de las bases RBF fijas? ¿Hay bases muertas o saturadas donde hace falta resolución? (señal sobre *dónde poner las bases*, no solo los puntos).
Q6. **Awareness de presupuesto.** ¿Hay señales de entrenamiento (tasa de mejora, plateau, conflicto de gradientes, rango del jacobiano) que indiquen *cuándo* cambiar de fase o de estrategia?

## Candidatas a señal (sin referencia)

Concentración espacial del residuo (Gini, masa en el top-k % de puntos; antecedente: RA-PIKAN); correlación residuo–|∇u|; razón (escala de la estructura)/(espaciado de bases); ocupación y número efectivo de bases, bases muertas; norma y razón de gradientes por término; coseno de conflicto entre términos; rango efectivo y número de condición del jacobiano; relación de actualización ‖Δθ‖/‖θ‖; ganancia y plateau de L-BFGS; simetrías y balances del problema (p. ej. antisimetría en Burgers, caudal en DFG); estabilidad entre semillas. La batería existente (`src/research_pipeline/sciml/signal_battery.py`, `dfg_signals.py`) cubre buena parte; las faltantes se implementan solo como cálculo sobre checkpoints.

## Análisis (fijado antes de calcular)

1. **Paso 0 — inventario** de checkpoints y registros por problema, brazo, semilla y etapa, con hashes y huecos declarados (se conoce de antemano que NS2d-CG y piezo no tienen etapa L-BFGS, y que el conflicto de gradientes solo existe en puntos de fase). Las celdas sin dato se marcan `null` con razón, nunca cero.
2. **Ficha de diagnóstico por corrida** (plantilla fija): contrato, receta, etapa, etiquetas (error global y por región), señales, modo de falla asignado y límites. Generada por script.
3. **Matriz problema × señal** con nivel de evidencia, dirección y estabilidad.
4. **Métodos estadísticos:** correlaciones de rango (Spearman) dentro de cada problema con intervalos por permutación; efectos pareados por semilla RBF−MLP; corrección de Holm por familia de pruebas; con n = 5 por celda se reportan tamaños de efecto con intervalos, no solo p. Métodos básicos de ML permitidos si hacen falta (p. ej. regresión regularizada, árboles/bosques con importancia por permutación) **solo con validación dejando una semilla fuera (leave-one-seed-out) y sin seleccionar características con los datos de evaluación**; los resultados de ML se reportan junto a su incertidumbre y nunca como único respaldo.
5. **Criterio fijo de "señal importante"** (vinculante): una señal es candidata si (a) se calcula sin referencia; (b) su relación con el error o con el modo de falla tiene la **misma dirección en al menos dos problemas** con |ρ| ≥ 0.5 e intervalo de permutación que excluye 0 tras Holm; (c) su ranking entre semillas es estable (τ de Kendall ≥ 0.5 entre semillas); (d) su costo de cálculo es < 10 % del tiempo de paso. Las señales que cumplen solo (b) en un problema se reportan como "específicas del problema".
6. **Del diagnóstico a familias de mecanismo:** cada señal candidata se asocia con la decisión de awareness que informaría (dónde: puntos/bases; cuándo: fase/presupuesto; en qué: términos de la pérdida). Se rankean familias por cuántas preguntas Q1–Q6 respalda, y se anotan antecedentes cercanos fuera del dominio (sin afirmar novedad).

## Contaminación declarada

Algunas señales ya se observaron en una primera pasada exploratoria (`signals-and-records.md`: caudal en DFG, bases muertas y masa de residuo del frente en Burgers, conflicto de gradientes). Este estudio es **exploratorio con respecto a ellas**; solo lo que se confirme con la regla de arriba sobre datos no usados para elegirlas cuenta como evidencia.

## Confirmación (solo si hace falta)

Se activa solo si una señal pasa (a)–(d) en descubrimiento pero su dirección no se sostiene en la validación leave-one-seed-out. Entonces: semillas nuevas, un máximo de 2 problemas entre Heat, Allen–Cahn y Burgers, con los mismos contratos y recetas de v1/v2, **tope de 1.5 h de sesión** y reserva ≥ 4 h de cuota. No se activa por defecto.

## Entregables

`experiments/diagnostics/EXP-0013-run-diagnosis/{results/,figures/}`: inventario, fichas por corrida (`cards/`), matriz problema × señal (CSV + figura), ranking de familias de mecanismo y `results.md`. Registro en `registry/runs.jsonl` (análisis) y, si corresponde, `failures.jsonl`.

## Condición de falsación

H-DIAG-1: existe al menos **una** señal sin referencia que cumple el criterio (a)–(d). Se rechaza si ninguna lo cumple; en ese caso el resultado es "las señales disponibles no informan dónde falla el modelo", y el siguiente paso es proponer señales nuevas, no elegir un mecanismo. Un resultado favorable no elige un mecanismo: da candidatas para un EXP-0014.

## Presupuesto

| Campo | Valor |
|---|---|
| Backend | CPU local (sin GPU por defecto) |
| Corridas nuevas | 0 (solo análisis de checkpoints) |
| GPU (solo confirmación opcional) | ≤ 1.5 h de sesión; reserva ≥ 4 h |

## Amenazas de validez

- **Reutilización de lo ya visto** (ver contaminación).
- **n pequeño** (5 semillas por celda): los intervalos serán anchos; no se sobreinterpretan p-valores.
- **Referencias de diferente calidad:** DFG y piezo usan FEM propios o de otro grupo (sin contraste de terceros); sus etiquetas pesan menos.
- **Etapas incompletas:** sin L-BFGS en NS2d-CG y piezo; se analizan solo con Adam.
- **Heterogeneidad entre problemas:** que una señal "funcione" en dos problemas puede deberse a un rasgo común no relacionado con el mecanismo; se reporta el contexto.
- **No se prueba causalidad:** las señales correlacionan; el efecto de actuar sobre ellas lo pondría a prueba un mecanismo posterior.

## Firma

- [x] Firmado por el investigador — fecha: 2026-10-05 — nombre: Rodrigo Mansilla (mensaje de chat "firmado", tras leer el spec completo)
