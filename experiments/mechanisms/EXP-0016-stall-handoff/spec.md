---
experiment_id: EXP-0016
slug: stall-handoff
version: v2             # firmada el 2026-10-19; v1 (2026-10-18, sin firmar) archivado en spec-v1-draft.md
status: decided       # draft | preregistered | running | scored | decided — el investigador lo cambia a `preregistered`
suite: tier2_candidate
problem_id: problem.burgers_pinnacle_1d (desarrollo), problem.allen_cahn_1d (réplica), problem.heat_1d (control negativo)
model_id: model.rbf_pikan (RBF48 en Burgers; RBF16 en Allen–Cahn y Heat), controles de baseline v1/v2
mechanism_id: mechanism.settling_focus (src/mechanisms/settling_focus.py, escrito y probado el 2026-10-19). Línea base: mechanism.stall_handoff (src/mechanisms/stall_handoff.py, escrito y probado)
hypothesis_id: H-HANDOFF-1 (techo de error), H-HANDOFF-T (techo de costo), H-AWARE-1 (la representación sabe dónde falla), H-HANDOFF-2 (cuándo), H-FOCUS-1 (lazo cerrado)
preregistered_at: 2026-10-19
preregistered_by: Rodrigo Mansilla (investigador; firma en el chat: "firmo")
---

# EXP-0016 v2 — Focalización guiada por el asentamiento de la representación interna

> **Terminología.** "PIKAN-PDE" = la RBF-PIKAN entrenada con la pérdida física (el sistema de la tesis). El MLP no participa en este experimento.

Redactado el 2026-10-19 por un agente sobre la v1 (Claude Code, 2026-10-18), a pedido del investigador: alinear el mecanismo con el título y los objetivos del protocolo. **No está firmado.**

## Qué cambia respecto a la v1 y por qué

La v1 proponía pasar de Adam a L-BFGS cuando la **pérdida total** se estanca. Es un buen diseño experimental, pero no es el artefacto del protocolo:

| Exigencia del protocolo | v1 | v2 |
|---|---|---|
| Título y objetivo general: focalización guiada por **señales internas del modelo** | señal = pérdida total (escalar, externa a la representación) | señal = asentamiento de los coeficientes de cada base RBF de la capa 0 (interna) |
| §4.3 y §4.4: el mecanismo debe **modificar la distribución espacial del aprendizaje** hacia las regiones de mayor variación | decide *cuándo*, no *dónde* | decide *cuándo* (traspaso) **y** *dónde* (peso de la pérdida por región, según qué bases no se asentaron) |
| §4.2: unidad de análisis = comportamiento espacial de la representación | error global | mapa espacial de asentamiento frente a mapa de error; error en regiones críticas |
| §4.4: misma arquitectura con y sin focalización, sin cambios de capacidad | sí | sí: el mecanismo no agrega parámetros ni bases (lección del EXP-0015: U41 mejoró también Heat por capacidad) |
| §4.5: no respaldada si mejora el error global empeorando el local en regiones críticas | no se medía | guardia explícita |

Se conserva de la v1: el techo con oráculo (etapa A), el reloj igualado, el pool uniforme fijo para L-BFGS, el control negativo y el ejecutor `sweep.py` con sus 21 pruebas. La señal de estancamiento de la pérdida pasa a ser la **línea base** que la señal interna tiene que superar.

## Pregunta

¿Puede la PIKAN-PDE, leyendo cuánto se asentó cada una de sus bases, decidir **cuándo** pasar a la fase de refinamiento y **dónde** concentrar esa fase, y con eso mejorar el error en las regiones críticas sin empeorar el global, a igual reloj y sin agregar capacidad?

## Por qué esta rama

El paso 0 del EXP-0015 (n = 3 en DFG, Burgers y Heat; n = 1 en el resto) mostró que la arquitectura representa la referencia entre 7× y 274× mejor de lo que la PIKAN-PDE la aprende en DFG, NS2d-CG, Allen–Cahn y piezo: el error es de optimización, no de capacidad. Mover o agregar bases no es la palanca (H-BASIS-2 no se sostiene en Burgers; U41 mejora también Heat). EXP-0011/0012 mostró que el estado con que se entrega a L-BFGS importa más que la calidad al final de Adam. EXP-0013 mostró que el residuo no localiza el error de forma consistente (solape del 5 % superior: 0.47–0.59 en DFG, 0–0.19 en Heat, Allen–Cahn, NS2d-CG y piezo): una señal externa no alcanza.

Por eso la focalización se mueve del *dónde poner puntos* (RAR, RAD) y del *dónde poner bases* (EXP-0015) al **dónde y cuándo concentrar el esfuerzo de optimización**, leído desde la representación.

**Delta frente a RAR (frase falsable para la tesis):** RAR y RAD redistribuyen dónde se evalúa la pérdida según el residuo, que es externo a la red; este mecanismo pondera la pérdida según qué partes de la representación todavía no se asentaron, una señal interna, sin mover puntos ni cambiar la capacidad. Lo observable que RAR no produce: un mapa de asentamiento por base que localiza el error mejor que el residuo (H-AWARE-1).

## Señales (fijadas antes de correr)

Ninguna lee la referencia. Ninguna importa `evals`.

**Señal interna (principal), asentamiento por base.** Cada 500 pasos de Adam se guardan los coeficientes de la capa 0, c[o, d, g] (salida o, dimensión de entrada d, base g). Para cada base (d, g):

S_dg(t) = ‖c[:, d, g](t) − c[:, d, g](t − 500)‖ / ‖c[:, d, g](t)‖

y su tasa relativa ρ_dg(t) = S_dg(t) / mediana de S_dg en los pasos 2 000–4 000. Una base está **asentada** cuando ρ_dg < 0.10 durante 3 ventanas seguidas (mismos umbrales que la señal de pérdida de la v1, sin recalibrar).

- **Mapa espacial de no asentamiento:** m(x) = Σ_d Σ_g φ_dg(x_d) · ρ_dg / Σ_d Σ_g φ_dg(x_d), con φ_dg la base gaussiana en la coordenada normalizada x_d. Como la capa 0 opera sobre las coordenadas, cada base corresponde a una franja del dominio.
- **Disparo (cuándo):** primer paso ≥ 4 000 en que la fracción de bases asentadas llega a q* = 0.80. Si no llega, k_t = k_max (nunca después que el control).

**Señal de línea base (cuándo, sin dónde):** estancamiento de la pérdida total, tal como estaba en la v1 (`src/mechanisms/stall_handoff.py`, 17 pruebas).

**Señal externa de comparación (dónde):** residuo |r(x)| al mismo paso (la de RAR).

Sensibilidades que se reportan y **no deciden:** q* ∈ {0.6, 0.9}; umbral de ρ ∈ {0.05, 0.20}; capa 1 en lugar de capa 0 para el mapa.

### Vista previa de la señal de línea base (estancamiento de la pérdida) sobre historias ya registradas

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
| **A — techo** | ¿Existe un traspaso mejor que el fijo, en error (H-HANDOFF-1) o en reloj (H-HANDOFF-T)? Oráculo; la referencia solo puntúa | GPU, ≈ 3.1 h esperadas, tope 5 h | sí |
| **B — señales en lazo abierto** | ¿El mapa interno localiza el error mejor que el residuo (H-AWARE-1)? ¿El disparo interno cae donde conviene, mejor que el de pérdida (H-HANDOFF-2)? Sobre los registros de A | CPU, sin corridas nuevas | sí (depende de A) |
| **C — lazo cerrado** | ¿La fase de refinamiento focalizada por el mapa interno gana al control en las regiones críticas sin perder en el global (H-FOCUS-1)? | GPU, por definir tras A y B | **no**: segunda firma |
| **D — prueba de estrés, seis problemas** | Heat, Burgers, Allen–Cahn, NS2d-CG, DFG y piezo (objetivo específico 3) | por definir | **no** |

### Etapa A — barrido de traspaso (techo)

Por semilla y problema se entrena Adam **una vez** (receta del control), se guardan los pesos en los pasos de la malla de traspaso y desde cada uno corre L-BFGS (strong-Wolfe, pool uniforme fijo) con el presupuesto restante para igualar el reloj total del control.

- **Malla de traspaso:** k ∈ {0.25, 0.40, 0.60, 0.80, 1.00} × pasos de Adam de la receta (Burgers: 5 000, 8 000, 12 000, 16 000, 20 000). k = 1.00 es el control. Heat usa {0.25, 0.60, 1.00}.
- **Reloj igualado (cota):** el tope de L-BFGS del traspaso k es T_total − t_Adam(k), con T_total = tiempo de Adam del control + su tope de L-BFGS, medido por semilla en la misma sesión. Es una cota: la receta termina por meseta (regla de la receta: 3 bloques con mejora relativa < 1e-9), así que el tope casi nunca actúa y el traspaso temprano termina antes que el control. Se guarda la curva de error contra reloj de cada L-BFGS (checkpoints cada 30 s), puntuada **después** del entrenamiento (el entrenamiento no lee la referencia). Se reporta también el tiempo total por brazo.
- **Recetas:** idénticas a v1/v2. Burgers: RBF48, 8 192 Sobol, 20 000 Adam f32, L-BFGS tope 600 s con parada por meseta, semillas 301–305. Allen–Cahn: RBF16, 9 200 puntos, 40 000 Adam, L-BFGS tope 300 s con parada por meseta, semillas 601–603. Heat: RBF16 S2, f64, 4 096 puntos, 20 000 Adam, semillas 901–902.
- **Desviación declarada en Heat:** L-BFGS con tope de **200 s** para todos los traspasos y para el control de esta etapa (la receta archivada llegaba a 15 000 iteraciones, ≈ 540–700 s). Motivo medido: en Heat el error ya está en su piso a las 50 iteraciones (≈ 3 s) y las 15 000 siguientes mejoran < 10 % (`heat-lbfgs-redo-20260930`). Heat es el control negativo; el criterio [0.80, 1.25] no depende de ese 10 %.
- **Lectura de E(k_t) fuera de la malla:** entre dos puntos de la malla se interpola en log; por debajo del primer punto se toma el primero y por encima del último, el último.
- **Definiciones por semilla:** E(k) = L2 relativo contra la referencia a reloj total igual. k* = argmin_k E(k) (el oráculo). E_fijo = E(1.00).

**Registro adicional exigido en la etapa A (nuevo en la v2):** en la corrida de Adam compartida por semilla, los coeficientes y la base lineal de la capa 0 cada 500 pasos (Burgers RBF48: 2 × 48 × 11 coeficientes por instante, < 1 MB por semilla), y la pérdida por punto del pool fijo en los mismos pasos. Sin esto la etapa B no se puede calcular. Se agrega a `sweep.py` con una prueba de que el registro no cambia el resultado (k = 1.00 sigue reproduciendo el control).

### Etapa B — señales en lazo abierto (CPU, sobre los registros de A)

**B1 — ¿la representación sabe dónde falla? (H-AWARE-1).** En los checkpoints de Adam de la etapa A se calculan, sin referencia, el mapa interno m(x) y el residuo |r(x)|, y aparte, con referencia (solo como etiqueta, igual que en EXP-0013), el error |u − u_ref|(x). Métricas por checkpoint: Spearman entre mapa y error, y solape entre el 5 % superior del mapa y el 5 % superior del error (azar = 0.05). Se comparan las dos señales en los mismos checkpoints.

**B2 — ¿el disparo cae donde conviene? (H-HANDOFF-2).** Con el disparo interno y el de pérdida se calcula k_t por semilla, sin referencia. Se lee E(k_t) del barrido (interpolación en log entre puntos de la malla). Índice de efectividad: EI = [log E_fijo − log E(k_t)] / [log E_fijo − log E(k*)], definido donde hay techo medible (log E_fijo − log E(k*) > log 1.05). Se reporta también el error de ubicación log(k_t / k*).

### Etapa C — lazo cerrado (requiere segunda firma; se fija aquí para que no se ajuste después)

Brazos, misma arquitectura, mismos parámetros, mismo reloj total:
- **C** control (receta fija).
- **E** traspaso fijo temprano, en la mediana de los k_t de la etapa B (atribuye el efecto a *traspasar antes*).
- **L** traspaso por estancamiento de la pérdida, sin focalizar (línea base de la v1).
- **T** mecanismo completo: traspaso por asentamiento **y** fase de refinamiento con la pérdida ponderada por el mapa interno congelado en k_t: w(x) = 1 + α · m̂(x) / media(m̂), α = 1, sobre el pool uniforme fijo (no se mueven puntos; regla heredada de EXP-0011/0012).
- **R** traspaso por asentamiento con la pérdida ponderada por el residuo en k_t (misma fórmula, señal externa): separa *qué señal* focaliza.

## Hipótesis y criterios (vinculantes al firmar)

| Hipótesis | Se sostiene si |
|---|---|
| **H-HANDOFF-1 (techo de error, Burgers)** | media geométrica de E(k*)/E_fijo ≤ 0.80 **y** ≥ 4/5 semillas ≤ 0.80 |
| **H-HANDOFF-T (techo de costo, Burgers), nueva** | existe un traspaso k < 1.00 con E(k) ≤ 1.05·E_fijo y reloj total ≤ 0.60·T_control en ≥ 4/5 semillas |
| **H-AWARE-1 (Burgers y Allen–Cahn)** | en cada problema, mediana sobre checkpoints de Spearman(m, error) ≥ 0.30, solape del 5 % superior ≥ 0.25 (5× el azar), y ambos mayores que los del residuo en ≥ 2/3 de las semillas |
| **H-HANDOFF-2 (Burgers y Allen–Cahn)** | sobre las semillas con techo medible: mediana de EI del disparo interno ≥ 0.50, y EI interno ≥ EI de la pérdida en ≥ 2/3 de las semillas |
| **H-FOCUS-1 (etapa C, Burgers)** | T/C del error **en la región crítica** (frente: |x| ≤ 5ν, t ≥ 1/π) con media geométrica ≤ 0.80 y ≥ 4/5 semillas ≤ 0.80; guardias abajo. Solo con segunda firma |

**Reglas de decisión entre etapas:**
- H-HANDOFF-1 y H-HANDOFF-T fallan: el traspaso no tiene techo ni en error ni en costo; la etapa C se rediseña sin el eje *cuándo* (solo focalización del *dónde* con traspaso fijo).
- H-HANDOFF-T se sostiene y H-HANDOFF-1 no: el resultado de A es de costo; C se evalúa a igual error con menos reloj.
- **H-AWARE-1 falla:** la representación no localiza su propio error; el *dónde* interno se cierra en esta construcción y se reporta como resultado negativo de la tesis (es el resultado más directamente ligado al título). No se pasa a C con el brazo T.
- H-AWARE-1 se sostiene: el investigador decide la segunda firma para C.

**Control negativo (Heat):** E(k*)/E_fijo en [0.80, 1.25] y H-AWARE-1 no se exige (Heat no tiene región crítica).

**Guardias de la etapa C (fijadas ahora):**
- g1 (§4.5 del protocolo): el error **global** de T no puede empeorar más de 1.10× el de C en media geométrica. Mejorar la región crítica empeorando el global, o al revés, no respalda la hipótesis.
- g2: reloj total de T ≤ el de C.
- g3: 5/5 corridas finitas.
- g4: réplica en Allen–Cahn (región crítica = interfaz, |u| < 0.9): T/C ≤ 1.25 en media geométrica.
- Atribución: T frente a E (¿es la señal o es traspasar antes?), T frente a L (¿suma el *dónde*?), T frente a R (¿la señal interna focaliza mejor que el residuo?). Se reportan las tres razones.

## Condición de falsación

- **A:** H-HANDOFF-1 y H-HANDOFF-T se rechazan con los criterios de la tabla.
- **B:** H-AWARE-1 se rechaza si no se cumple alguno de sus tres criterios en alguno de los dos problemas; H-HANDOFF-2 igual.
- **Banda no concluyente:** media geométrica entre 0.80 y 1.00, o razones mezcladas. Un resultado favorable no es un claim (gate `scientific_claim`); `scientific_accepted=false`.

## Límites declarados antes de correr

- **Potencia:** n = 5 y CV ≈ 0.6 entre semillas en Burgers tras L-BFGS; Allen–Cahn (n = 3) y Heat (n = 2) son descriptivos.
- **La capa 0 es separable por dimensión:** el mapa m(x) es una suma de franjas en x y en t, no una región arbitraria. Puede no resolver un error localizado en un punto (x, t). La sensibilidad con la capa 1 lo explora pero no decide.
- **Asentarse no es converger:** una base puede estancarse lejos de su valor bueno. Por eso H-AWARE-1 se mide contra el error real y no se asume.
- **El oráculo es un techo de esta construcción** (malla de cinco traspasos, un tope de L-BFGS), no el techo absoluto de la familia.
- **Burgers, Allen–Cahn y Heat no cubren los modos de falla de piezo y DFG** (L-BFGS destruye la solución; L-BFGS sin presupuesto): etapa D.
- **Etapa 2 del protocolo (oscilatorio, alta frecuencia):** Burgers y Allen–Cahn cubren frentes e interfaces multiescala, no oscilaciones. Si el asesor exige alta frecuencia, se agrega un Helmholtz con k alto como réplica en la etapa D.

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
| (ii-GPU) misma comprobación en GPU | **medida el 2026-10-19**: resultado en la tabla de pruebas de la v2, más abajo (Allen–Cahn determinista; Burgers reproducible después del calentamiento) |
| Ejecutor de Heat | **pendiente**: `sweep.py` cubre Burgers y Allen–Cahn (clases de la campaña `pikan_campaign.py`); Heat usa el código de la campaña S2 y necesita un adaptador. Heat es el control negativo y el primero en recortarse (ver Presupuesto); las hipótesis H-HANDOFF-1 y 2 no dependen de él |

**Tiempos medidos en CPU local** (4 hilos, `timing_smoke.json`): Burgers 0.97 s por paso de Adam y 1.05 s por iteración de L-BFGS; Allen–Cahn 0.35 s y 0.43 s. En T4 la campaña registra ≈ 0.040 s y ≈ 0.020 s por paso de Adam: la CPU es ≈ 24× y ≈ 17× más lenta. **Consecuencia:** un solo Adam de Burgers (20 000 pasos) tardaría ≈ 5.4 h en esta CPU. La etapa A no se puede correr en CPU; las pruebas de CPU quedan en humo y en la señal.

**Pruebas previas que agrega la v2 (CPU, 2026-10-19):**

| Prueba | Estado |
|---|---|
| unitarias de la señal de asentamiento, del mapa m(x) y de los pesos | **hecha**: `tests/test_settling_focus.py`, 33 pruebas pasan (base que se asienta antes/después, que oscila, muerta, invariancia de escala, persistencia, causalidad sin mirar el futuro, monotonía en q*, mapa constante si ρ constante, mapa como combinación convexa de ρ, pesos de media 1) |
| aislamiento: `settling_focus.py` no importa `evals` | **hecha**: lo recorre `tests/test_mechanism_isolation.py` y lo comprueba su propia prueba (solo `numpy` y `math`) |
| el registro de coeficientes y de r² por punto en `sweep.py` no cambia el resultado | **hecha**: `tests/test_stall_handoff_sweep.py`, 31 pruebas pasan. Las pruebas de igualdad bit a bit de k = 1.00 con el control original corren **con el registro activado**; además, una corrida con el registro apagado da estados idénticos; el registro tiene la forma preregistrada; los coeficientes registrados coinciden con los del modelo; la media de r² por punto es la pérdida de entrenamiento |
| vista previa del mapa interno sobre checkpoints de Adam archivados | **hecha en lo posible**, ver abajo |
| (ii-GPU) igualdad en GPU | **medida el 2026-10-19** (kernel `rodrimandub/af-pikans-exp-0016-precheck-gpu`, un T4, 20 min; `results/precheck-gpu/`). **Allen–Cahn: determinista**, los 3 controles y k = 1.00 del barrido son idénticos bit a bit. **Burgers: los controles 2 y 3 y k = 1.00 del barrido son idénticos bit a bit; el control 1, la primera corrida del proceso, difiere** (estado a 35 % de distancia relativa; error final 1.7e-3 frente a 3.8e-3). Lectura: k = 1.00 reproduce el control en GPU **después del calentamiento**; la primera corrida de un proceso no es reproducible. El criterio escrito en el script (desvío ≤ 2 × ruido) da "no" en un indicador porque el ruido lo fija ese único control atípico, no porque el barrido se desvíe de los controles 2 y 3. **Consecuencia para la etapa A:** calentar antes de la primera semilla de cada proceso (corrida desechable con las mismas formas), y comprobar el calentamiento en GPU antes del bloque grande. La comparación sigue siendo contra la réplica del control dentro del barrido |
| calentamiento antes de la primera semilla de cada proceso | **hecho y comprobado en GPU (2026-10-19)**: `sweep.warmup` (corrida desechable de 50 pasos, mismas formas, 13 s en T4; prueba en CPU: no cambia ningún resultado). Kernel `rodrimandub/af-pikans-exp-0016-warmup-check` (`results/warmup-check-gpu/`): en un proceso nuevo, tras calentar, el control A, el control B y k = 1.00 del barrido son **idénticos bit a bit** (hash del estado de Adam y de L-BFGS) y coinciden con los controles 2 y 3 de la prueba previa, de otra sesión de Kaggle; no coinciden con el control 1 en frío. Una primera corrida calentada reproduce a las siguientes |
| ejecutor de Heat | **sigue pendiente** (ver la fila de arriba; es lo primero que se recorta) |

**Decisiones de implementación fijadas al escribir `settling_focus.py`** (no cambian ningún umbral; se declaran porque el spec no las precisaba):
- La tasa de referencia es la mediana de S_dg en las ventanas que **terminan** en (2 000, 4 000] (4 ventanas), la misma convención que la señal de pérdida de la v1. Se exige k_min ≥ 4 000 para que ninguna decisión use una referencia a medias.
- Una base cuyos coeficientes no se mueven (muerta) tiene ρ = 0 y cuenta como asentada; igual una base con tasa de referencia 0. Las bases asentadas se cuentan sobre todas las D × G bases de la capa 0 (Burgers RBF48: 2 × 11 = 22; 80 % = 18 bases).
- Los pesos w = 1 + α·m/media(m) se **reescalan a media 1** para que la pérdida ponderada conserve la escala de la no ponderada (no cambia los pesos relativos; evita alterar L-BFGS por escala).
- `sweep.py` registra, cada 500 pasos de Adam y en el paso 0, los coeficientes y la base lineal de la capa 0 y r² por punto del pool fijo (`<problema>-seed<semilla>-adam-trace.npz`), y calcula, sin referencia, el disparo interno para (q*, ρ*) = (0.8, 0.10) y las cuatro sensibilidades. El tiempo del registro no cuenta en el reloj de Adam (`record_seconds`).

**Vista previa del mapa interno (Heat S2, 5 semillas, checkpoints de Adam archivados).** Solo Heat guarda 4 instantes de Adam por semilla (pasos ≈ 4 300, 10 700, 17 800 y 20 000); Burgers y Allen–Cahn guardan solo el final y no se pueden ver hasta la etapa A. No es la señal preregistrada (los instantes no están separados por 500 pasos y no hay ventana 2 000–4 000): se usa la tasa de cambio por 500 pasos entre instantes, normalizada por la mediana entre bases del mismo intervalo. `preview_settling.py`; no lee la referencia.
- **El mapa no es constante:** rango/media del mapa entre 0.50 y 0.90 y coeficiente de variación entre 0.11 y 0.20 en las 5 semillas; la tasa por base varía entre 0.40× y 2.06× la mediana.
- **Su forma se repite solo en parte entre semillas:** Spearman del mapa entre pares de semillas, mediana 0.45 (rango 0.17–0.83, 10 pares). Las bases más inestables en el último tramo se repiten (x-dimensión, bases 5, 7, 8 en 901–904), no en la 905.
- Lectura: la señal no es una constante ni puro ruido, pero tampoco una huella estable: no sostiene por sí misma H-AWARE-1. Heat es suave y no tiene región crítica; esto solo descarta el fallo trivial (mapa plano).

## Superficie de edición declarada

- [x] `src/mechanisms/settling_focus.py` (señal de asentamiento, mapa y disparo) y sus pruebas; `src/mechanisms/stall_handoff.py` queda como línea base.
- [x] el ejecutor del barrido (`sweep.py`, con el registro de coeficientes) y los scripts de análisis dentro de `experiments/mechanisms/EXP-0016-stall-handoff/`.
- [ ] No se toca `evals/`, el modelo base, la PDE ni las recetas de v1/v2.

## Amenazas de validez

- **Fuga de la referencia:** la señal usa solo la pérdida de entrenamiento; la referencia puntúa las salidas después. El oráculo k* usa la referencia a propósito y está etiquetado `oracle_ceiling`: no es elegible como mecanismo ni como claim.
- **Selección posterior:** ρ*, W, k_min y la persistencia están fijos antes de ver el barrido; las sensibilidades no deciden.
- **Efecto de "traspasar antes" frente a efecto de la señal:** se separa con el brazo E en la etapa C; en las etapas A–B se ve en k* frente a k_t.
- **Reloj y hardware:** control y barrido en la misma sesión.
- **Estado de Adam y L-BFGS no independientes:** los E(k) de una misma semilla comparten la trayectoria de Adam, así que los cinco puntos de una semilla no son muestras independientes; la inferencia usa las semillas.

- **La región crítica se define con la referencia** (frente de Burgers e interfaz de Allen–Cahn, con las definiciones de `evals/problems/*/reference.py`), solo para puntuar. El mecanismo nunca la ve.

## Firma

- [x] Firmado por el investigador (etapas A y B) — fecha: 2026-10-19 — nombre: Rodrigo Mansilla (chat: "firmo", tras revisar el spec v2 completo)
- [x] Aprobado el uso de la referencia en el barrido (oráculo `oracle_ceiling`) y en las etiquetas de B1 — fecha: 2026-10-19 — Rodrigo Mansilla (chat: "Apruebo el uso de la referencia en el barrido (oráculo oracle_ceiling) y en las etiquetas de B1")
- [ ] Segunda firma para la etapa C (solo tras leer A y B) — fecha: ____
