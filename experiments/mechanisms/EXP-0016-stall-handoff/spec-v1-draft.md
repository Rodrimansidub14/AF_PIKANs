---
experiment_id: EXP-0016
slug: stall-handoff
version: v1-draft
status: draft            # draft | preregistered | running | scored | decided — el investigador lo cambia a `preregistered`
suite: tier2_candidate
problem_id: problem.burgers_pinnacle_1d (desarrollo), problem.allen_cahn_1d (réplica), problem.heat_1d (control negativo)
model_id: model.rbf_pikan (RBF48 en Burgers; RBF16 en Allen–Cahn y Heat), controles de baseline v1/v2
mechanism_id: mechanism.stall_handoff (src/mechanisms/stall_handoff.py; aún no escrito)
hypothesis_id: H-HANDOFF-1 (techo), H-HANDOFF-2 (señal en lazo abierto), H-HANDOFF-3 (lazo cerrado)
preregistered_at:
preregistered_by:
---

# EXP-0016 — ¿Puede la red decidir cuándo pasar de Adam a L-BFGS? Techo, señal en lazo abierto y lazo cerrado

> **Terminología (2026-10-19).** "PIKAN-PDE" = la RBF-PIKAN entrenada con la pérdida física (el sistema de la tesis). No es el MLP: el MLP (PINN clásico) solo aparece como comparador en los baselines.

Redactado el 2026-10-18 por un agente, a partir de la decisión del investigador de cancelar el bloque GPU del EXP-0015 y seguir la rama A (el error de la PIKAN-PDE es de optimización). **No está firmado.** Hasta entonces no se ejecuta nada en GPU. Los umbrales de este documento quedan fijos al firmar.

## Pregunta

Con el mismo presupuesto total de reloj, ¿pasar de Adam a L-BFGS en el momento en que la propia pérdida de la red se estanca da menos error que el paso fijo de la receta actual?

## Por qué esta rama y este mecanismo

**La rama A.** El paso 0 del EXP-0015 (ajuste supervisado de la misma arquitectura a la referencia, n = 3 en DFG, Burgers y Heat) mostró que en DFG la PIKAN-PDE queda 35× por encima de lo que la arquitectura puede representar; con n = 1, lo mismo en piezo (274×), NS2d-CG (22×) y Allen–Cahn (7×). Ahí el error es de optimización o de formulación, no de capacidad. En Burgers y Heat el paso 0 no concluye sobre el techo.

**Antecedentes propios que apuntan al traspaso, no a las bases ni a los puntos:**
- EXP-0011/0012: lo ganado en Adam se pierde con L-BFGS (RAD en Heat: 11× menos error al final de Adam, 2–5× más tras L-BFGS). La etapa y el estado con que se entrega a L-BFGS importan.
- EXP-0013 (Q3, modos de falla): `budget_limited` (L-BFGS terminó por tope) en Burgers 10 de 15–20 corridas, pero **solo en las campañas grandes** (24 000 Adam, tope de 1 800 s, y la de f64); en la receta `burgers-close` que usa este spec (20 000 Adam f32, tope de 600 s) las 15 corridas terminaron por **meseta**, L-BFGS de 59 a 196 s (RBF48: 179–196 s, 3 110–3 429 iteraciones). Allen–Cahn RBF16 también termina por meseta (86–127 s con tope de 300 s). Por tanto, con estas recetas el traspaso temprano **no** da a L-BFGS "más tiempo": lo que cambia es el estado con que L-BFGS arranca. En DFG con la receta de Daniel L-BFGS casi no mejora (razón 0.92–0.99). En piezo L-BFGS destruyó la solución en todas las variantes propias.
- Costo: en `burgers-close`, Adam ocupa ≈ 800 s y L-BFGS ≈ 190 s por corrida en T4. Un traspaso a 0.5 de Adam ahorra ≈ 400 s (≈ 40 % del reloj de la corrida); por eso el tiempo para llegar a un error objetivo se reporta junto con el error final.
- EXP-0013 (Q6): el residuo y su concentración al final de Adam anticipan cuánto ganará L-BFGS solo en DFG (ρ ≈ −0.74); en Burgers y Allen–Cahn la relación es débil (|ρ| ≤ 0.44). Es un indicio de que el estado de Adam informa a L-BFGS, y es débil. Se declara como límite (ver abajo).

**Por qué un solo mecanismo y no dos.** Se evaluaron dos candidatos: (1) traspaso decidido por la red, y (2) condicionamiento por capa de los coeficientes RBF. Se elige (1) porque:
1. su señal es un escalar que ya se registra (la pérdida de entrenamiento), de costo de cálculo nulo, y se valida en lazo abierto sin escribir un mecanismo de entrenamiento;
2. el techo se mide barato con un barrido de traspaso (la lección del EXP-0015: el oráculo se mide **antes** de construir una señal);
3. solo puede traspasar **antes o igual** que el control, así que no puede gastar más reloj que él y la comparación a reloj total igual es directa;
4. (2) no tiene un techo barato definido, sus señales candidatas (ocupación, presión de gradiente) no mostraron consistencia entre problemas en EXP-0013, y toca la parametrización del modelo (pruebas de equivalencia, como en EXP-0015).

(2) queda como candidato de una iteración posterior si (1) se cierra en el techo.

## Mecanismo (fijado antes de correr)

`src/mechanisms/stall_handoff.py`: función pura de la historia de pérdida; **no importa `evals`** ni lee la referencia.

- Entrada: la pérdida de entrenamiento total registrada cada 100 pasos de Adam.
- Tasa de mejora por bloque: se agrupa la historia en bloques de W = 500 pasos; m_j = media de log L en el bloque j; r_j = (m_{j−1} − m_j) / W (mejora de log-pérdida por paso).
- Tasa de referencia: r_ref = mediana de r_j en los bloques entre los pasos 2 000 y 4 000 (pasado el transitorio inicial).
- Disparo: primer bloque j con paso ≥ k_min = 4 000 tal que ρ_j = r_j / r_ref < ρ* = 0.10 durante 3 bloques consecutivos. El paso de traspaso k_t es el fin del tercer bloque.
- Tope: si no dispara, k_t = k_max = pasos de Adam de la receta (nunca traspasa después que el control).
- Un único valor fijo de ρ* (0.10); 0.05 y 0.20 se reportan como sensibilidad y **no deciden**. Sin calibración sobre los datos de evaluación.
- L-BFGS arranca desde los pesos en k_t (no hay estado del optimizador que traspasar). Puntos de colocación: el pool uniforme fijo de la semilla, igual que el control (regla heredada de EXP-0011/0012).

Señal secundaria que solo se registra, sin decidir: relación de actualización ‖Δθ‖/‖θ‖ por 500 pasos.

### Vista previa de la señal sobre historias ya registradas (Tier 1, solo pérdida)

`preview_trigger.py` aplica la señal, con los parámetros de arriba sin cambios, a las historias de pérdida de Adam de las campañas de baseline. **Lee solo pérdida y reloj; no lee ningún error contra la referencia ni puntúa nada.** Sirve para comprobar que la señal no queda inerte (k_t = k_max siempre) ni dispara en el borde de la ventana siempre. Los parámetros ρ*, W, k_min y la persistencia **no se ajustaron** después de verla.

| Problema | Semilla → k_t (disparó en `stall`; si no, k_max) | Disparó | Reloj de Adam ahorrado (T4) |
|---|---|---|---|
| Burgers RBF48 (k_max 20 000) | 301: 17 500 · 302: 12 500 · 303: 20 000 (no dispara) · 304: 20 000 (dispara justo en k_max) · 305: 10 500 | 3 de 5 antes de k_max | 100–380 s en las 3 que disparan |
| Allen–Cahn RBF16 (k_max 40 000) | 601: 18 500 · 602: 10 500 · 603: 13 000 · 604: 13 000 · 605: 16 500 | 5 de 5 | 430–610 s |
| Heat RBF16 f64 (k_max 20 000) | 901: 7 500 · 902–905: 20 000 (no disparan) | 1 de 5 | 350 s en 901 |

Lectura: la señal no es inerte y no dispara siempre en el mismo punto (Burgers 10 500–20 000; Allen–Cahn 10 500–18 500). Hay dos consecuencias para el diseño: (1) en Burgers dos de cinco semillas quedan con k_t = k_max, es decir, idénticas al control por construcción, lo que diluye el efecto medible de la señal y es una razón más para que la etapa A (techo) decida antes de la B; (2) en Allen–Cahn todos los k_t caen entre los puntos 10 000 y 24 000 de la malla, por lo que la interpolación entre dos puntos de la malla pesa mucho en E(k_t). Se declara para la lectura de la etapa B; no se cambia la malla.

## Etapas

| Etapa | Qué prueba | Costo | Se lanza con esta firma |
|---|---|---|---|
| **A — techo** | ¿Existe un traspaso mejor que el fijo con un reloj total no mayor? (oráculo; usa la referencia solo para puntuar) | GPU, ≈ 3.1 h esperadas, tope 5 h | sí |
| **B — señal en lazo abierto** | ¿El k_t que calcula la señal cae donde el barrido dice que conviene? Se calcula sobre los registros de la etapa A | CPU, sin corridas nuevas | sí (depende de A) |
| **C — lazo cerrado** | ¿El mecanismo, corriendo solo, gana al control a igual reloj? | GPU, por definir tras A y B | **no**: requiere una segunda firma (go/no-go) |
| **D — prueba de estrés, seis problemas** | Heat, Burgers, Allen–Cahn, NS2d-CG, DFG y piezo con el mecanismo ya definido; en piezo la opción "no traspasar" es válida | por definir | **no** |

Burgers sigue siendo el problema de desarrollo; la prueba de estrés en los seis problemas (etapa D) es posterior a definir el mecanismo.

### Etapa A — barrido de traspaso (techo)

Por semilla y problema se entrena Adam **una vez** (receta del control), se guardan los pesos en los pasos de la malla de traspaso y desde cada uno corre L-BFGS (strong-Wolfe, pool uniforme fijo) con el presupuesto restante para igualar el reloj total del control.

- **Malla de traspaso:** k ∈ {0.25, 0.40, 0.60, 0.80, 1.00} × pasos de Adam de la receta (Burgers: 5 000, 8 000, 12 000, 16 000, 20 000). k = 1.00 es el control. Heat usa {0.25, 0.60, 1.00}.
- **Reloj igualado (cota):** el tope de L-BFGS del traspaso k es T_total − t_Adam(k), con T_total = tiempo de Adam del control + su tope de L-BFGS, medido por semilla en la misma sesión. Es una cota: la receta termina por meseta (regla de la receta: 3 bloques con mejora relativa < 1e-9), así que el tope casi nunca actúa y el traspaso temprano termina antes que el control. Se guarda la curva de error contra reloj de cada L-BFGS (checkpoints cada 30 s), puntuada **después** del entrenamiento (el entrenamiento no lee la referencia). Se reporta también el tiempo total por brazo.
- **Recetas:** idénticas a v1/v2. Burgers: RBF48, 8 192 Sobol, 20 000 Adam f32, L-BFGS tope 600 s con parada por meseta, semillas 301–305. Allen–Cahn: RBF16, 9 200 puntos, 40 000 Adam, L-BFGS tope 300 s con parada por meseta, semillas 601–603. Heat: RBF16 S2, f64, 4 096 puntos, 20 000 Adam, semillas 901–902.
- **Desviación declarada en Heat:** L-BFGS con tope de **200 s** para todos los traspasos y para el control de esta etapa (la receta archivada llegaba a 15 000 iteraciones, ≈ 540–700 s). Motivo medido: en Heat el error ya está en su piso a las 50 iteraciones (≈ 3 s) y las 15 000 siguientes mejoran < 10 % (`heat-lbfgs-redo-20260930`). Heat es el control negativo; el criterio [0.80, 1.25] no depende de ese 10 %.
- **Lectura de E(k_t) fuera de la malla:** entre dos puntos de la malla se interpola en log; por debajo del primer punto se toma el primero y por encima del último, el último.
- **Definiciones por semilla:** E(k) = L2 relativo contra la referencia a reloj total igual. k* = argmin_k E(k) (el oráculo). E_fijo = E(1.00).

### Etapa B — índice de efectividad en lazo abierto

Con la historia de pérdida de la corrida de control de cada semilla se calcula k_t **sin referencia**. Luego se lee E(k_t) del barrido (interpolación descrita arriba).

- **Índice de efectividad:** EI = [log E_fijo − log E(k_t)] / [log E_fijo − log E(k*)]. Vale 1 si la señal iguala al oráculo, 0 si no mejora al paso fijo y < 0 si empeora. Solo se define donde log E_fijo − log E(k*) > log(1.05) (hay techo medible).
- **Error de ubicación:** log(k_t / k*).

## Hipótesis y criterios (vinculantes al firmar)

| Hipótesis | Se sostiene si |
|---|---|
| **H-HANDOFF-1 (techo, Burgers)** | media geométrica de E(k*)/E_fijo ≤ 0.80 **y** ≥ 4/5 semillas ≤ 0.80 |
| **H-HANDOFF-2 (señal, Burgers y Allen–Cahn)** | sobre las semillas con techo medible: mediana de EI ≥ 0.50 **y** E(k_t) ≤ 1.25·E(k*) en ≥ 2/3 de las semillas, en cada uno de los dos problemas |
| **H-HANDOFF-3 (lazo cerrado, etapa C)** | media geométrica T/C ≤ 0.80 y ≥ 4/5 semillas ≤ 0.80 en Burgers, con las guardias de abajo. Solo se evalúa con la segunda firma |

**Reglas de decisión entre etapas:**
- Si H-HANDOFF-1 **no** se sostiene: se cierra la familia "traspaso" en esta construcción (el techo del oráculo no da 20 %); no se construye la señal ni se pasa a C. Se reporta el techo por problema.
- Si H-HANDOFF-1 se sostiene y H-HANDOFF-2 **no**: la señal por estancamiento no localiza el momento; se proponen otras señales (p. ej. ‖Δθ‖/‖θ‖ registrada) en una iteración nueva, no se cierra el lazo.
- Si ambas se sostienen: el investigador decide la segunda firma para la etapa C.

**Control negativo (Heat):** E(k*)/E_fijo en [0.80, 1.25] (Heat está en su piso de discretización; si el barrido encuentra ≥ 20 % de mejora allí, la lectura "el traspaso fijo es el cuello de botella en problemas difíciles" se debilita). Allen–Cahn: mismo criterio de 0.80 en H-HANDOFF-1; se reporta como réplica y no decide la etapa A.

**Guardias de la etapa C (fijadas ahora, vinculantes si se firma C):**
- g1: en reloj total igual, T no puede quedar peor que C por más de 1.25× en media geométrica (el final de Adam no se evalúa: el mecanismo lo acorta por diseño).
- g2: reloj total de T ≤ T_total del control (por construcción, k_t ≤ k_max).
- g3: 5/5 corridas con pérdida finita.
- g4: réplica en Allen–Cahn: razón T/C en media geométrica ≤ 1.25.
- **Brazo de atribución E:** traspaso fijo temprano en la mediana de los k_t de las cinco semillas. Si T no mejora a E (razón T/E > 0.95), la ganancia es de traspasar antes, no de la señal de la red; se reporta.

## Condición de falsación

- **A:** si la media geométrica de E(k*)/E_fijo en Burgers es > 0.80 o menos de 4/5 semillas llegan a ≤ 0.80, H-HANDOFF-1 se rechaza y el traspaso por estancamiento no tiene techo en Burgers.
- **B:** si la mediana de EI < 0.50 o E(k_t) > 1.25·E(k*) en menos de 2/3 de las semillas en alguno de los dos problemas, H-HANDOFF-2 se rechaza.
- **Banda no concluyente:** media geométrica entre 0.80 y 1.00, o razones mezcladas. Un resultado favorable **no** es un claim (gate `scientific_claim`); `scientific_accepted=false`.

## Límites declarados antes de correr

- **Potencia:** n = 5 y CV ≈ 0.6 entre semillas en Burgers tras L-BFGS; solo se detectan efectos grandes. Allen–Cahn con n = 3 y Heat con n = 2 son descriptivos.
- **La señal puede no localizar el momento.** En EXP-0013 la relación entre el estado de Adam y la ganancia de L-BFGS fue fuerte solo en DFG y débil en Burgers y Allen–Cahn. Una etapa B negativa es un resultado válido.
- **El reloj varía con el hardware:** control y barrido corren en la misma sesión y tipo de GPU, y se guardan las curvas de error contra reloj, no solo el valor final. En estas recetas L-BFGS termina por meseta y no por tope, así que el efecto esperado es el del estado de entrega, no el del tiempo extra.
- **El oráculo es un techo de esta construcción** (malla de cinco traspasos, un tope de L-BFGS, ρ* y W únicos), no el techo absoluto de la familia.
- **Burgers, Allen–Cahn y Heat no cubren los modos de falla de piezo y DFG** (L-BFGS destruye la solución; L-BFGS sin presupuesto). Eso es la etapa D, no esta firma.
- **El paso 0 no mide el techo de la arquitectura en Burgers** (el ajuste supervisado quedó por debajo de la PIKAN-PDE con menos presupuesto): que el error de Burgers sea de optimización se sostiene por EXP-0011/0012 y EXP-0013, no por el paso 0.

## Presupuesto

| Campo | Valor |
|---|---|
| Backend | Kaggle T4×2, privado |
| Corridas de la malla | Burgers 5 semillas × 5 traspasos + Allen–Cahn 3 × 5 + Heat 2 × 3, con un Adam compartido por semilla; un trabajo por semilla, dos GPU en paralelo |
| Costo esperado | Burgers ≈ 1.9 h + Allen–Cahn ≈ 0.9 h + Heat ≈ 0.3 h ≈ **3.1 h** de sesión. **Tope duro 5 h.** Si hace falta recortar, Heat sale primero y Allen–Cahn baja a 2 semillas |
| Cota superior | si L-BFGS agotara su tope en todos los traspasos tempranos: ≈ 7 h. No se espera (las recetas terminan por meseta), pero no está medido para estados de Adam tempranos; por eso el tope duro de 5 h y el orden Burgers → Allen–Cahn → Heat |
| Origen de las cifras | tiempos **medidos** en T4 en las campañas de baseline: Burgers RBF48 f32, Adam 791–807 s y L-BFGS 179–196 s por corrida; Allen–Cahn RBF16 f32, Adam 772–822 s y L-BFGS 86–127 s; Heat RBF16 f64, Adam ≈ 560 s. Se supone ≈ 300 s por L-BFGS desde un estado temprano en Burgers, ≈ 150 s en Allen–Cahn y ≈ 200 s (tope) en Heat. Esos tres son los supuestos **no medidos** |
| Pruebas previas (Tier 0/1, CPU) | ver el cuadro de abajo |

**Estado de las pruebas previas (2026-10-19, CPU):**

| Prueba | Estado |
|---|---|
| (i) unitarias de la señal con curvas sintéticas | **hecha**: `tests/test_stall_handoff.py`, 17 pruebas pasan (estancamiento, sin disparo, tope k_max, k_min, picos aislados, persistencia, ruido del 3 %, invariancia de escala, causalidad sin mirar el futuro, monotonía en ρ*, historias degeneradas y entradas inválidas) |
| (iii) aislamiento: el mecanismo no importa `evals` | **hecha**: `tests/test_mechanism_isolation.py` recorre todo `src/mechanisms/` y pasa; además la prueba propia comprueba que el módulo solo importa `numpy` y `math` |
| (iv) humo de la receta en CPU | **hecha** para Burgers RBF48 y Allen–Cahn RBF16 (200 pasos de Adam + L-BFGS de 20 s, pérdida finita y decreciente). Heat no se corrió: usa el código de la campaña S2 |
| (v) medición de tiempos | **hecha**: ver abajo |
| (ii) el traspaso en k = 1.00 reproduce el control bit a bit | **hecha en CPU** (2026-10-19): `sweep.py` + `tests/test_stall_handoff_sweep.py`, 21 pruebas pasan (Burgers y Allen–Cahn, red chica, L-BFGS limitado por iteraciones). El estado de Adam, el estado final de L-BFGS, el estado/iteraciones, las puntuaciones y la historia de pérdida de k = 1.00 son idénticos al control original (`run_worker` de la campaña, sin tocar). También: el orden del barrido no cambia ningún resultado; los 5 traspasos terminan en estados distintos; el entrenamiento corre con un objeto que solo tiene `loss` (sin `score` ni referencia) |
| (ii-GPU) misma comprobación en GPU | **pendiente, antes de lanzar**: en GPU las reducciones atómicas pueden romper la igualdad bit a bit. Se corre un control y un barrido corto en T4 y se mide el ruido de GPU (dos controles entre sí); k = 1.00 debe quedar dentro de ese ruido. Si no, la etapa A compara contra la réplica del control **dentro del barrido** (que ya es el comparador pareado), no contra una corrida aparte |
| Ejecutor de Heat | **pendiente**: `sweep.py` cubre Burgers y Allen–Cahn (clases de la campaña `pikan_campaign.py`); Heat usa el código de la campaña S2 y necesita un adaptador. Heat es el control negativo y el primero en recortarse (ver Presupuesto); las hipótesis H-HANDOFF-1 y 2 no dependen de él |

**Tiempos medidos en CPU local** (4 hilos, `timing_smoke.json`): Burgers 0.97 s por paso de Adam y 1.05 s por iteración de L-BFGS; Allen–Cahn 0.35 s y 0.43 s. En T4 la campaña registra ≈ 0.040 s y ≈ 0.020 s por paso de Adam: la CPU es ≈ 24× y ≈ 17× más lenta. **Consecuencia:** un solo Adam de Burgers (20 000 pasos) tardaría ≈ 5.4 h en esta CPU. La etapa A no se puede correr en CPU; las pruebas de CPU quedan en humo y en la señal.

## Superficie de edición declarada

- [x] `src/mechanisms/stall_handoff.py` (función del disparo) y sus pruebas.
- [x] el ejecutor del barrido y los scripts de análisis dentro de `experiments/mechanisms/EXP-0016-stall-handoff/`.
- [ ] No se toca `evals/`, el modelo base, la PDE ni las recetas de v1/v2.

## Amenazas de validez

- **Fuga de la referencia:** la señal usa solo la pérdida de entrenamiento; la referencia puntúa las salidas después. El oráculo k* usa la referencia a propósito y está etiquetado `oracle_ceiling`: no es elegible como mecanismo ni como claim.
- **Selección posterior:** ρ*, W, k_min y la persistencia están fijos antes de ver el barrido; las sensibilidades no deciden.
- **Efecto de "traspasar antes" frente a efecto de la señal:** se separa con el brazo E en la etapa C; en las etapas A–B se ve en k* frente a k_t.
- **Reloj y hardware:** control y barrido en la misma sesión.
- **Estado de Adam y L-BFGS no independientes:** los E(k) de una misma semilla comparten la trayectoria de Adam, así que los cinco puntos de una semilla no son muestras independientes; la inferencia usa las semillas.

## Firma

- [ ] Firmado por el investigador (etapas A y B) — fecha: ____ — nombre: ____
- [ ] Aprobado el uso de la referencia en el barrido (oráculo `oracle_ceiling`) — fecha: ____
- [ ] Segunda firma para la etapa C (solo tras leer A y B) — fecha: ____
