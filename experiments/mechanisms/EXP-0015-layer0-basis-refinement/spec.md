---
experiment_id: EXP-0015
slug: layer0-basis-refinement
version: v2-draft        # v1 (2026-10-05, sin firmar) archivado en spec-v1-draft.md
status: preregistered    # draft | preregistered | running | scored | decided
suite: tier2_candidate
problem_id: problem.burgers_pinnacle_1d (primario), problem.allen_cahn_1d (réplica), problem.heat_1d (control negativo)
model_id: model.rbf_pikan con capa 0 de centros por dimensión y ancho por base
mechanism_id: none (prueba de techo; ningún brazo es un mecanismo)
hypothesis_id: H-BASIS-1 (cantidad), H-BASIS-2 (ubicación, primaria), H-BASIS-3 (especificidad)
preregistered_at: 2026-10-18
preregistered_by: Rodrigo Mansilla (mensaje de chat "1. Firmo", 2026-10-18; registrado por el agente con su OK explícito)
---

# EXP-0015 v2 — ¿Cantidad o ubicación? Prueba de techo sobre las bases de la capa 0

Redactado el 2026-10-18 por un agente. Amplía la v1 (que solo probaba 11 → 41 bases uniformes) con dos brazos que separan **cuántas** bases de **dónde** están. **No está firmado**; hasta entonces no se ejecuta nada en GPU.

## Por qué cambia respecto a la v1

1. **La v1 confunde cantidad con resolución local.** Si U41 mejora no se sabe si fue por tener más bases o por tener bases cerca del frente. Si no mejora, tampoco descarta que *ubicarlas* ayude.
2. **El eje "dónde resuelve la red" necesita un techo antes de diseñar una señal.** Si ni siquiera las bases colocadas con la referencia (oráculo) mejoran el error, ninguna señal interna lo va a hacer, y la familia "bases" se descarta sin construir un mecanismo.
3. **El EXP-0014 se pospone** (decisión propuesta, pendiente del investigador): su gate λ = Spearman(|r|, ‖∇u‖) solo pesa cuando las dos señales coinciden, que es justo cuando la elección no importa, así que en la práctica coincide con la ablación de estructura (el propio spec espera G ≈ S). Su brazo S vuelve más adelante como la celda "puntos" de un factorial puntos × bases.

## Hipótesis (falsables)

- **H-BASIS-2 (primaria, ubicación):** con el mismo número de bases (11) y de parámetros, colocar los centros de la capa 0 en x según la estructura de la referencia (O11) reduce el L2 relativo tras L-BFGS en Burgers en ≥ 20 % frente al control C.
- **H-BASIS-1 (cantidad):** refinar la capa 0 de forma uniforme (11 → 41, U41) reduce ese error en ≥ 20 % (la hipótesis de la v1, sin cambios).
- **H-BASIS-3 (especificidad):** la misma no uniformidad colocada **lejos** del frente (X11) **no** mejora ≥ 5 %. Si mejora, la ganancia de O11 viene de la no uniformidad en sí, no de la ubicación.

**Expectativa declarada antes de correr:** efecto chico de U41. El control RBF48 llega a 7e-5 en Burgers con una estructura unas 15 veces más fina que el espaciado de la capa 0, lo que sugiere que las capas ocultas ya compensan. La pregunta informativa es H-BASIS-2.

## Brazos

| Brazo | Bases capa 0 (dim x) | Dim t | Ubicación en x | Parámetros |
|---|---|---|---|---|
| **C** | 11 | 11 uniformes | uniformes en [−1, 1] (control, receta v1) | = control |
| **U41** | 41 | 41 uniformes | uniformes | +3 % (RBF48), +26 % (RBF16) |
| **O11** | 11 | 11 uniformes | **oráculo**: equidistribución del monitor M(x) (ver abajo) | = control |
| **X11** | 11 | 11 uniformes | el mismo espaciado de O11, desplazado para que la zona densa quede lejos del frente (centro de densidad en x = +0.5 en Burgers; ver tabla por problema) | = control |

Capas ocultas idénticas al control en todos los brazos. O11 y X11 tienen **exactamente** el mismo multiconjunto de espaciados entre centros: solo cambia dónde están.

### Construcción del oráculo (fijada antes de correr)

- Monitor en x a partir de la referencia, promediado en t: m(x) = mean_t |∂u_ref/∂x|(x, t), en la malla de referencia.
- Monitor regularizado con un piso para evitar el colapso de centros: M(x) = f + (1 − f)·m(x)/mean_x m, con **f = 0.3**.
- Centros por equidistribución (de Boor): μ_g = F⁻¹((g − 1)/(G − 1)), con F la CDF normalizada de M en [−1, 1]; extremos fijos en ±1.
- Ancho por base: σ_g = 1.75 × (espaciado local medio), con espaciado local = (μ_{g+1} − μ_{g−1})/2 (en los extremos, la distancia al único vecino). Con espaciado uniforme reproduce σ = 0.35 del control.
- **Uso de la referencia:** el oráculo lee la referencia **una vez, antes de entrenar**, solo para fijar los centros. Esto viola a propósito la regla "el entrenamiento nunca lee la referencia" y por eso requiere aprobación explícita del investigador al firmar. O11 está etiquetado `oracle_ceiling`: **no es elegible como mecanismo ni como claim**; solo fija el techo.

### Por problema

| Problema | Modelo / receta | Ubicación de la zona densa | Brazos |
|---|---|---|---|
| Burgers (primario) | RBF48, receta `burgers-close` (8 192 Sobol, 20 000 Adam f32, L-BFGS tope 600 s), semillas 301–305 | O11: frente en x = 0. X11: zona densa en x = +0.5 | C, U41, O11, X11 |
| Allen–Cahn (réplica) | RBF16, receta `ac-rerun` (9 200 puntos, 40 000 Adam, L-BFGS tope 300 s), semillas 601–605 | O11: interfaces según m(x). X11: monitor reflejado (x → −x) si m es asimétrico; si es simétrico, desplazamiento de 0.5 con reflexión en los bordes | C, U41, O11 |
| Heat (control negativo) | RBF S2 (float64, 4 096 puntos, 20 000 Adam, L-BFGS tope 700 s), semillas 901–905 | O11: monitor de sin(πx) | C, U41, O11 |

## Cambio de implementación requerido (solo capa 0)

- En la capa 0, `centers` pasa de forma `[G]` (compartido entre dimensiones de entrada) a `[in, G]`, y `width` de escalar a `[in, G]`. Las capas ocultas no cambian.
- **Prueba de equivalencia (Tier 0, bloqueante):** con centros uniformes y ancho constante, el modelo nuevo reproduce **bit a bit** la salida y los gradientes del control con la misma semilla, en Burgers, Allen–Cahn y Heat (dos familias de código: `model.py` usa centros en [−1.05, 1.05] y σ = 0.21; `dfg.py`/Burgers usan [−1, 1] y σ = 0.35).
- Pruebas unitarias: monotonía de los centros, extremos en ±1, el multiconjunto de espaciados de O11 igual al de X11, σ_g > 0.

## Métricas

**Primaria (Burgers):** L2 relativo contra Cole–Hopf tras L-BFGS; razón por semilla brazo/C (misma sesión, misma semilla, mismo pool Sobol).

**Nueva (se reporta, no decide en esta iteración):** curva de error contra tiempo de reloj a lo largo de todo el pipeline. Se guardan checkpoints cada 1 000 pasos de Adam y cada bloque de L-BFGS, y se puntúan **después** del entrenamiento (el entrenamiento no lee la referencia). Se reporta el tiempo para llegar a un error objetivo ε (ε = 2× el L2 final de la mediana de C) y el área bajo log₁₀ L2 contra el tiempo. Motivo: en v2 todas las corridas de Burgers terminaron por tope de tiempo de L-BFGS, así que el valor final mide en parte el presupuesto.

**Brecha cerrada (definida aquí para las iteraciones siguientes):** para un futuro mecanismo M, GC = log(C/M) / log(C/O11), por semilla. Vale 0 si no gana nada y 1 si iguala al oráculo. Solo se define si H-BASIS-2 se sostiene.

**Diagnósticos:** distribución espacial del error (masa del error en la franja |x| < 0.05 de Burgers frente al resto), ocupación de las bases de la capa 0, razón estructura/espaciado recalculada con los centros de cada brazo.

## Criterios (vinculantes al firmar)

| Hipótesis | Se sostiene si (Burgers, tras L-BFGS) |
|---|---|
| H-BASIS-2 | media geométrica O11/C ≤ 0.80 **y** ≥ 4/5 semillas ≤ 0.80 |
| H-BASIS-1 | media geométrica U41/C ≤ 0.80 **y** ≥ 4/5 semillas ≤ 0.80 |
| H-BASIS-3 | media geométrica X11/C ≥ 0.95 |

**Guardias (todas):** (g1) fin de Adam, razón brazo/C ≤ 1.25 en media geométrica, para O11 y U41; (g2) tiempo por corrida ≤ 1.5× C; (g3) 5/5 corridas finitas.
**Control negativo (Heat):** O11/C y U41/C en [0.80, 1.25]. Si alguno mejora ≥ 20 %, la lectura "la capa 0 limita donde la estructura es fina" se debilita.
**Allen–Cahn:** mismo criterio de 0.80; se reporta como réplica, no decide.
**No concluyente:** media geométrica en (0.80, 1.00] o razones mezcladas.

## Qué se decide con el resultado

| O11 | U41 | X11 | Lectura | Siguiente |
|---|---|---|---|---|
| mejora | mejora o no | no mejora | la **ubicación** importa y hay techo medible | EXP-0016: mecanismo de r-adaptividad interna (centros movidos por un monitor calculado con señales propias de la red, sin referencia), con número de bases fijo; se mide con GC; brazo de ablación con centros entrenables por gradiente (delta frente a Free-RBF-KAN) |
| mejora | — | también mejora | importa la no uniformidad, no la ubicación | revisar el piso f y el ancho σ_g antes de cualquier mecanismo |
| no mejora | mejora | — | importa la **cantidad**, no la ubicación | la familia "bases" se reduce a capacidad; un mecanismo de ubicación no tiene margen |
| no mejora | no mejora | — | la capa 0 **no** es el cuello de botella en Burgers | descartar la capa 0; evaluar ocupación en capas ocultas o pasar al eje "cuándo" (traspaso Adam → L-BFGS) |

Cualquier resultado favorable es exploratorio (`scientific_accepted=false`) y no es un claim (gate `scientific_claim`).

## Límites declarados antes de correr

- **Potencia:** n = 5 y CV ≈ 0.6 entre semillas en Burgers tras L-BFGS; solo se detectan efectos grandes en la comparación pareada.
- **El oráculo es un techo de esta construcción** (monitor |∂u/∂x| promediado en t, f = 0.3, solo dim x, solo capa 0), no el techo absoluto de la familia.
- U41 conserva la confusión de capacidad de la v1 (+3 % en RBF48, +26 % en RBF16). O11 y X11 no la tienen.
- En U41, al refinar con ancho 1.75 × espaciado, también baja el ancho: espaciado y ancho no se separan.
- L-BFGS se detiene por tiempo: la razón final depende del presupuesto (por eso se añaden las curvas de error contra tiempo).

## Enmienda firmada el 2026-10-18 — Paso 0 bloqueante (CPU) y alcance a todos los problemas

Aprobada por el investigador en el mismo mensaje de la firma ("Prueba de ajuste OK", "EN TODOS LOS PROBLEMAS").

**Paso 0 — ajuste supervisado (Tier 1, CPU, bloqueante para el bloque GPU).** Antes de cualquier corrida en GPU, cada arquitectura de control se ajusta **directamente a la referencia** (pérdida MSE contra u_ref, sin término físico) en los seis problemas: Heat, Burgers, Allen–Cahn, NS2d-CG, DFG y piezo. Pregunta: ¿la arquitectura *puede representar* la solución con el error que reporta el baseline tras L-BFGS? Brazos: C (control) en todos; U41 y O11 en todos; X11 en Burgers y Allen–Cahn. Presupuesto fijo por corrida (Adam + L-BFGS, mismo número de pasos para todos los brazos de un problema), 3 semillas, float64. Se reporta el L2 relativo del ajuste frente al L2 del baseline PINN.

| Resultado del paso 0 en un problema | Lectura |
|---|---|
| L2 del ajuste de C ≪ L2 del baseline PINN (≥ 10×) | la representación no limita; el error del PINN es de optimización. El bloque GPU de ese problema se recorta |
| L2 del ajuste de C comparable al baseline (< 3×) y O11 o U41 mejoran ≥ 20 % | la capa 0 limita; el bloque GPU confirma |
| intermedio | se reporta; decide el investigador |

El paso 0 **no** usa la referencia en ningún mecanismo: es un diagnóstico, igual que las etiquetas del EXP-0013.

**Alcance de las pruebas de estrés.** Una vez definido el mecanismo (EXP-0016), las pruebas se extienden a **todos los problemas** (Heat, Burgers, Allen–Cahn, NS2d-CG, DFG y piezo), no solo a Burgers. Burgers sigue siendo el problema de desarrollo.

## Presupuesto

| Campo | Valor |
|---|---|
| Backend | Kaggle T4×2, privado |
| Corridas | Burgers 20 (4 brazos × 5) + Allen–Cahn 15 + Heat 15 = 50 |
| Costo estimado | Burgers ≈ 2.8 h + Allen–Cahn ≈ 2.0 h + Heat ≈ 2.4 h ≈ **7.2 h** de sesión; **tope duro 8 h**. Si hace falta recortar, Heat baja a C y O11 (≈ 1.6 h) |
| Orden de ejecución | Burgers primero (decide); Allen–Cahn y Heat solo si los brazos de Burgers terminan sin violar g3 |
| Pruebas previas (Tier 0/1, CPU) | equivalencia bit a bit; pruebas unitarias del constructor de centros; humo de 200 pasos por brazo en CPU |

## Superficie de edición declarada

- [x] la capa 0 del modelo RBF (centros `[in, G]`, ancho `[in, G]`) detrás de un parámetro de configuración; por defecto, comportamiento idéntico al control.
- [x] el ejecutor y el constructor de centros dentro de `experiments/mechanisms/EXP-0015-layer0-basis-refinement/`.
- [ ] No se toca `src/mechanisms/`, `evals/`, la PDE ni las recetas de v1/v2.

## Firma

- [x] Firmado por el investigador — fecha: 2026-10-18 — nombre: Rodrigo Mansilla (chat: "1. Firmo")
- [x] Aprobado el uso de la referencia por el brazo O11 (`oracle_ceiling`) — fecha: 2026-10-18 (incluido en la firma del punto 1)
- [ ] Decidido posponer EXP-0014 — fecha: ____
