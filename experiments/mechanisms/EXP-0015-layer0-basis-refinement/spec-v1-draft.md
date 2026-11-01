---
experiment_id: EXP-0015
slug: layer0-basis-refinement
status: draft            # draft | preregistered | running | scored | decided — el investigador lo cambia a `preregistered`
suite: tier2_candidate
problem_id: problem.burgers_pinnacle_1d, problem.allen_cahn_1d (hipótesis), problem.heat_1d (control negativo)
model_id: model.rbf_pikan con bases de la capa 0 refinadas
mechanism_id: none (prueba de una señal de bases; no es todavía un mecanismo)
hypothesis_id: H-BASIS-1
preregistered_at:
preregistered_by:
---

# EXP-0015 — ¿Limita el espaciado de las bases de la capa 0 donde la estructura es más fina?

Redactado el 2026-10-05 por un agente a partir de la decisión del investigador (opción B de EXP-0013). **No está firmado**; hasta entonces no se ejecuta nada en GPU.

## Pregunta

EXP-0013 midió que la anchura de la estructura del propio modelo es mucho menor que el espaciado de las bases RBF de la capa 0 en Burgers (razón 0.066, 15×) y Allen–Cahn (0.24, 4×), y mayor en Heat (6.1), NS2d-CG (4.1) y piezo (9.4). Es un **indicio**, no una prueba (las capas ocultas pueden afinar la estructura y la razón es constante dentro de cada problema, así que no se pudo contrastar con los datos de EXP-0013). Este experimento lo contrasta variando solo el espaciado de la capa 0.

## Hipótesis (falsable)

H-BASIS-1: refinar las bases de la **capa 0** (11 → 41 bases por arista, con ancho = 1.75 × espaciado como en el control; capas ocultas sin cambios) reduce el L2 relativo tras L-BFGS en **Burgers** en ≥ 20 % (media geométrica de razones B41/C ≤ 0.80 y ≥ 4 de 5 semillas ≤ 0.80), y **no** produce una mejora ≥ 20 % en **Heat** (control negativo: allí la estructura ya es más gruesa que las bases, razón 6.1).

Allen–Cahn (razón 0.24) es la **réplica**: se espera efecto en la misma dirección; se reporta y no decide.

## Diseño

| Elemento | Valor |
|---|---|
| Brazos | **C** control (11 bases por arista en todas las capas, receta v1) y **B41** (41 bases solo en la capa 0) |
| Parámetros extra | solo la capa 0: +960 en el RBF16 (3 648 → 4 608; +26 %) y en el RBF48 (29 376 → 30 336; +3 %). Es una confusión de capacidad **declarada**, no aislable con este diseño |
| Burgers | RBF48 (receta de `burgers-close`: 8 192 puntos Sobol, 20 000 Adam float32, L-BFGS tope 600 s), semillas 301–305 |
| Allen–Cahn | RBF16 (receta de `ac-rerun`: 9 200 puntos, 40 000 Adam, L-BFGS tope 300 s), semillas 601–605 |
| Heat (control negativo) | RBF de S2 (3 648 par., float64, 4 096 puntos, 20 000 Adam, L-BFGS tope 700 s), semillas 901–905 |
| Comparador | **réplica del control en la misma sesión**, pareada por semilla (el control se comparte con EXP-0014 si se lanzan juntos) |
| Dosis | una sola (41); la respuesta a la dosis (21) queda para una iteración posterior |

## Métricas, guardias y criterio (vinculantes al firmar)

**Primaria (Burgers, B41 vs C):** L2 relativo contra Cole–Hopf **tras L-BFGS**, razón por semilla; se cumple con media geométrica ≤ 0.80 **y** ≥ 4/5 semillas ≤ 0.80.
**Control negativo (Heat):** la razón B41/C tras L-BFGS en media geométrica debe quedar **en [0.80, 1.25]**; si mejora ≥ 20 % (< 0.80), el espaciado de la capa 0 también ayuda donde la estructura ya es gruesa y la señal de EXP-0013 pierde interpretación.
**Guardias:** (g1) L2 al final de Adam, razón B41/C ≤ 1.25 en media geométrica; (g2) tiempo por corrida ≤ 1.5× el control; (g3) 5/5 corridas finitas.
**Se reportan:** ambas etapas (fin de Adam y tras L-BFGS) en los tres problemas; Allen–Cahn con el mismo criterio de 0.80.

## Condición de falsación

- **H-BASIS-1 se rechaza** si en Burgers la media geométrica B41/C tras L-BFGS es > 0.80, o < 4/5 semillas tienen razón ≤ 0.80, o se viola una guardia: el indicio sobre bases de EXP-0013 **no** se sostiene como efecto.
- **Se debilita** (aunque Burgers cumpla) si Heat mejora ≥ 20 %.
- **No concluyente:** media geométrica en (0.80, 1.00] o razones mezcladas.
- Un resultado favorable no es un claim (gate `scientific_claim`) y no identifica un mecanismo: solo informa si "dónde poner bases" merece una iteración.

## Límites declarados antes de correr

- **Confusión de capacidad** (parámetros de la capa 0) y de **espaciado/ancho** (al refinar 41 bases con ancho 1.75×espaciado, el ancho también baja; no se separan).
- **Potencia:** n = 5 y CV ≈ 0.6 entre semillas en Burgers tras L-BFGS.
- La razón estructura/espaciado de EXP-0013 se calculó sobre checkpoints de los controles, no sobre B41; en B41 cambia el denominador.
- Heat usa float64 y otro ejecutor (S2); su pareo es independiente del de Burgers/AC.

## Presupuesto

| Campo | Valor |
|---|---|
| Backend | Kaggle T4×2, privado; **se lanza después de la renovación de cuota (2026-10-10)** |
| Corridas | Burgers 10 (C, B41 × 5) + Allen–Cahn 10 + Heat 10 = 30; si se comparte el control con EXP-0014: 15 corridas nuevas (solo B41) |
| Costo estimado | Burgers ≈ 1.4 h + Allen–Cahn ≈ 1.3 h + Heat ≈ 1.6 h de sesión (≈ 4.3 h sin compartir control, ≈ 2.2 h con control compartido); **tope duro 5 h**; reserva ≥ 12 h |
| Costo conjunto con EXP-0014 | ≈ 6.8 h estimadas (cap 8 h), alrededor de 27 % de una cuota de 30 h |
| Pruebas previas (Tier 0/1, CPU) | humo del ejecutor con `bases0=41` en 100 pasos en CPU; comprobar que con `bases0=11` reproduce el modelo del control |

## Superficie de edición declarada

- [x] solo el ejecutor del experimento (parámetro `bases0` del modelo RBF; en Heat, `RBFLayer` con `basis_count` por capa) en `experiments/mechanisms/EXP-0015-layer0-basis-refinement/`.
- [ ] No se toca `src/mechanisms/`, `evals/`, la PDE ni las recetas de v1.

## Camino de iteraciones

| Resultado | Siguiente |
|---|---|
| H-BASIS-1 se sostiene | iteración con respuesta a la dosis (21, 41, 81), separación espaciado/ancho/parámetros y un mecanismo que **decida** dónde densificar bases (adaptativo, tipo Free-RBF-KAN guiado por señales propias) |
| Se rechaza | descartar bases de la capa 0 como cuello de botella; la estructura fina se resolvería en capas ocultas; se vuelve a la familia "dónde puntos" / "cuándo" |
| No concluyente | más semillas o la dosis 21 |

## Firma

- [ ] Firmado por el investigador — fecha: ____ — nombre: ____
