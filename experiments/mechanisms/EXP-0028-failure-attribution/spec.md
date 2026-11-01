---
experiment_id: EXP-0028
slug: failure-attribution
version: v1.1
status: preregistered   # firmado por el investigador en el chat 2026-10-23 («Ya firmo todo»), incluida la enmienda v1.1
suite: tier2_candidate
problem_id: banco = Allen–Cahn, Burgers, Fisher–KPP, NS2d-CG, DFG (se miden); piezo bimorfo (se predice y se corre al final)
model_id: model.rbf_pikan (solo PIKAN; no se corren MLP ni se repite la línea base)
mechanism_id: mechanism.gram_preconditioning (brazos J, F, E, I; B = ID)
hypothesis_id: H-ATTR-A, H-ATTR-B, H-ATTR-C, H-LEV-1, H-PRED-1
drafted_at: 2026-10-23
requested_by: Rodrigo Mansilla (chat 2026-10-23, tras la reunión)
---

# EXP-0028 — ¿Qué componente del mecanismo, qué propiedad del problema y qué parte del entrenamiento hacen que falle en un problema y no en otro?

## Alcance fijado por el investigador
- Solo PIKAN (RBF-KAN). **No** se corren MLP/PINN ni soluciones estándar, **no** se repite la línea base: B se usa solo como control pareado dentro de cada experimento.
- **Estructura (decisión del investigador, 2026-10-23):** el banco de problemas es Allen–Cahn, Burgers, Fisher–KPP, **NS2d-CG y DFG**; en todos se *mide* qué parte del mecanismo sirve y por qué. Con ese historial se **predice, antes de correr piezo**, qué partes de un mecanismo servirán acopladas al bimorfo piezoeléctrico del paper (mecánica + electricidad). **Piezo es el único problema fuera de muestra**, el que más cómputo exige, y se corre al final (sí o sí). NS y DFG no se predicen: se miden y entran al historial.
- Se sigue la línea de EXP-0023 a 0027; no se re-proponen las familias cerradas (0014–0018, observador, cadencia/recálculo, predictor por pérdida inicial).

## Antecedentes que fijan el punto de partida
| Hecho | Fuente |
|---|---|
| E reduce el error en Allen–Cahn (0.306 agrupado, 10/10), lo empeora en Fisher–KPP (mediana 2.7–5.3, 1/5) | EXP-0023 1-C |
| En Burgers depende de la semilla; recálculo y cadencia no son el factor (EXP-0026). La pérdida de B en el paso de intervención **no** predice qué semillas dañan (ρ = +0.18, p = 0.44; EXP-0027) | EXP-0026/0027 |
| F (Gram fijo, sin residuo, sin activaciones entrenadas) retiene parte de la ganancia en Allen–Cahn (0.514) | EXP-0023 1-C |
| E no focaliza la interfaz: reduce más fuera de ella | EXP-0023 1-C |
| Lectura de R: ~45 % puntos únicos, 36–46 % en la banda del choque | EXP-0026 |

## Pregunta
Tres ejes, que se contestan por separado y luego se unen:
- **Eje A, mecanismo:** de qué parte de E sale la ganancia o el daño.
- **Eje B, problema:** qué propiedades medibles antes de entrenar acompañan a un efecto bueno o malo.
- **Eje C, entrenamiento:** si la falla ocurre en la fase de Adam, al pasar a L-BFGS, o en ambas.

## Fase 1 — descomposición del mecanismo (GPU, solo PIKAN)

**Brazos** (todos del módulo `gram_preconditioning`, sin tocar el mecanismo):

| Brazo | Qué conserva | Qué aísla frente al anterior |
|---|---|---|
| B (= ID) | nada | control |
| F | Gram fijo de la base, uniforme en [−1, 1]; sin residuo; sin recálculo | corrige el solape/condicionamiento **de la base RBF** |
| J | diagonal del bloque de E (escala por coeficiente) | solo **escala** adaptada al residuo |
| E | bloque completo por arista, recalculado con el residuo | solape + **acoplamiento** + dependencia del residuo |
| I | un bloque por entrada (acopla aristas) | acoplamiento entre aristas |

Contrastes pareados por semilla, sobre log(e_X/e_B):
- **D5 (E frente a F):** ¿la ventaja de E es dependencia del residuo o solo corrección del solape de la base? Es el contraste que responde a la decisión pendiente D5.
- **J frente a B:** ¿basta con reescalar?
- **E frente a J:** ¿importan los términos fuera de la diagonal?
- **I frente a E:** ¿importa acoplar aristas?

**Fase 1a (ya implementado):** Allen–Cahn d = 1e‑3 y d = 2e‑4, Burgers RBF48, Fisher–KPP. **Fase 1c (tras generalizar el mecanismo al residuo vectorial, ver Fase 4):** NS2d‑CG y DFG con los mismos brazos y 10 semillas, para que el historial incluya acoplamiento de varias ecuaciones. Detalle de 1a: con el arnés de 1‑B/1‑C sin cambios y **semillas nuevas 4001–4010** (verificado por búsqueda: no usadas en `experiments/` ni `registry/`). 5 brazos × 4 problemas × 10 semillas = 200 corridas; se reutilizan los B ya guardados **solo** donde recetas y semillas coincidan exactamente (no es el caso en 4001–4010, así que B se corre). Estimación ~1.5–2 min por corrida en T4 → ~2 h de pared en T4×2; tope de despacho y de kill se fijan al firmar.

**Reglas (fijadas antes de ver datos, por problema, sin agrupar):**
- H-ATTR-A (la ganancia viene de la base): en un problema donde E mejora a B (mediana ≤ 0.90, ≥ 7/10), F alcanza ≥ 70 % de la ganancia de E en escala log (log(F/B) ≤ 0.7·log(E/B)) → la ventaja es de corrección de la base, no del residuo.
- H-ATTR-B (el daño viene de la fuerza del precondicionador): en un problema donde E empeora (mediana > 1.1), J también empeora y F no → el daño está en la parte adaptada al residuo; si F también empeora → está en la reparametrización misma.
- Un contraste «respaldado» exige mediana pareada y conteo ≥ 7/10; «rechazado» si no; no se bajan umbrales.

## Fase 1b — dónde se produce el efecto (eje C)
Sobre las mismas corridas se guardan, con la rutina de `sweep`, error y pérdida al final de Adam y al final de L‑BFGS. Se reporta por problema y brazo la razón e_X/e_B en **ambos** puntos. Antecedente: lo ganado en Adam puede perderse con L‑BFGS (EXP‑0011/0012). Descriptivo, sin umbral propio; alimenta la lectura de A.

## Fase 2 — propiedades por PDE (eje B, CPU, antes de entrenar)

Para cada problema se mide **a la inicialización**, con el mismo modelo y puntos que el entrenamiento, una batería fija (se fija aquí, no después de ver resultados):
1. cond(B) de los bloques por arista (min/mediana/máx), y rango de valores singulares de P (módulo ya existente);
2. solape de la base: número de condición del Gram uniforme de las 11 bases (depende del ancho y no del problema; sirve como covariable de F);
3. concentración espacial del residuo inicial (fracción de masa del |r| en el 10 % de puntos más grande);
4. rigidez/escala del término de reacción-difusión-advección (número tipo Péclet o Damköhler calculado del PDE con parámetros dados);
5. ratio de términos de borde frente a PDE en la pérdida inicial;
6. nº de salidas y de ecuaciones acopladas.

Variable respuesta por problema: mediana de log(e_E/e_B) y de log(e_F/e_B) de la fase 1. Con 4–6 problemas el ajuste es **descriptivo** (tabla problema × propiedad × efecto, sin regresión con p). La fuerza del eje B viene de la predicción de la fase 3.

## Fase 3 — predicción para piezo, antes de correrlo (H-PRED-1)
Con el banco medido (Allen–Cahn, Burgers, KPP, NS2d‑CG, DFG: fases 1, 1b, 1c y 2) y **antes de correr cualquier cosa de piezo**, se escribe y se firma `predictions.md`. Contenido:
1. **Inventario:** tabla banco × propiedad × efecto de cada brazo (F, J, E, I) con el historial por problema, incluyendo qué componente explicó el efecto (H-ATTR‑A/B).
2. **Propiedades de piezo medidas a la inicialización** con la misma batería de la fase 2 (CPU, sin entrenar): acoplamiento mecánico–eléctrico (8 salidas, formulación mixta), bordes duros por levantamiento, cond(B) por arista, concentración del residuo, escalas dispares entre desplazamiento (~1e‑6 m) y potencial (100 V), ratio borde/PDE.
3. **Predicción por brazo** (F, J, E, I): dirección (mejora / empeora / indiferente) y magnitud aproximada de log(e_X/e_B), con la regla de inferencia usada (qué problema(s) del banco se parecen a piezo y por qué propiedad). Debe decir además **qué componente del mecanismo se espera que sirva** (base, escala, acoplamiento, residuo) y cuáles no.
- **H-PRED-1 respaldada** si la dirección prevista coincide con la observada en ≥ 3 de los 4 brazos y la dirección del mejor brazo prevista es la observada (piezo es **un** problema: una coincidencia es un indicio, no una prueba; así se declara y no se generaliza a «el mecanismo funciona en acoplamientos»).
- Si falla, se reporta con la tabla de errores de predicción y la propiedad que engañó; `predictions.md` no se reescribe.
- Mientras el banco no esté completo (falta 1c) se puede escribir una **predicción preliminar** solo con AC/Burgers/KPP; no sustituye a la final y se firma aparte.

## Fase 4 — generalización a NS/DFG (1c) y experimento final en piezo (requiere trabajo previo)
NS2d‑CG y DFG (fase 1c) entran al banco con brazos B, F, J, E, I y 10 semillas; piezo se corre **después** de firmar `predictions.md`, con brazos B, F, J, E, I y 10 semillas (el cómputo mayor). **No se lanza** hasta tener:
- *Mecanismo en residuo vectorial:* `gn_blocks`/`make_residual` solo soportan hoy Burgers y Allen–Cahn (escalar). Hace falta la interfaz `prob.residual_terms` con todos los términos (PDE y borde) y los pesos de la pérdida, y `functional_call` sobre el modelo real. Inventario completo en `docs/plans/2026-10-20-faltantes-ns-dfg-piezo.md` §A. Primera prueba en CPU sobre NS2d‑CG (más barato). Hallazgos del inventario (2026-10-23, por verificar al implementar): `make_residual` exige `prob.transform` y ramas `burgers_pinnacle_1d`/`allen_cahn` (`gram_preconditioning.py:88-104`); `gn_blocks` solo usa el residuo PDE sin términos de borde ni pesos (`:129-149`); `_forward` replica `RBFKAN` sin `output_scales` (DFG). NS2d‑CG: pérdida de 8 términos sin pesos, full‑batch, Adam 20 000, float32, y el arnés reutilizable es el worker de `pikan_campaign.py` con `lbfgs_stage` de `sweep.py` (L‑BFGS por iteraciones). DFG usa minibatches y, según el código leído, **no contiene L‑BFGS** (contradice el plan del 2026-10-20: verificar). Por eso **NS2d‑CG se implementa primero y DFG después, ambos dentro del banco** (decisión del investigador: se miden los dos antes de predecir piezo); DFG necesita además un adaptador y su receta de L‑BFGS por verificar.
- *Piezo, benchmark del paper (bimorfo PVDF, código de González; EXP‑0009):* receta Adam lr 1e‑3, 20 000 épocas, float64, sin L‑BFGS, 2 048 interiores y 256 de frontera/interfaz, remuestreo cada 200 épocas, 8 salidas con bordes duros, `poling_sign = −1`. Línea base RBF16 ya congelada (0.092, n = 5): **no se repite**. Se corre el mecanismo en la receta del paper aunque difiera de una decisión previa nuestra y se reporta la diferencia. **Debe resolverse antes de firmar la fase 4:** (i) pesos adaptativos del código de Daniel frente a «pesos fijos» de `metodologia/baseline/piezo.tex:32`; (ii) signos de e21/e22 (tabla actual con signos opuestos, archivada con el mismo signo); (iii) licencia (MIT declarada sin archivo LICENSE); (iv) la referencia es del mismo grupo (sin contraste de terceros: se declara). Sin etapa L‑BFGS en piezo, el mecanismo actúa solo en Adam (cadencia 1 000 como en el resto).
- *Presupuesto medido* del Jacobiano con 8 salidas, en CPU, antes de pedir GPU.

## Qué NO hace este experimento
No repite línea base ni MLP; no introduce mecanismos nuevos; no ajusta umbrales después de ver resultados; no afirma causalidad por propiedad con 4–6 problemas.

## Superficie de edición y firma
Solo archivos nuevos en `experiments/mechanisms/EXP-0028-failure-attribution/` (ejecutor, análisis, `predictions.md`). No se modifica `gram_preconditioning.py` salvo la generalización a residuo vectorial, que se hará con pruebas y su propia nota. No se toca `evals/` ni `registry/claims.jsonl`. **Fases 1a, 1b y 2 pueden firmarse ya; 1c y 4 cuando exista el mecanismo en residuo vectorial; la 3 (predicciones para piezo) cuando el banco esté medido.**

## Enmienda v1.1 (2026-10-23, firmada en el chat: «tomar todas las palancas en cuenta… ya firmo todo»)

**Cambio de criterio.** Las palancas cerradas en negativo (EXP‑0014 a 0018, 0024 idea A, palanca R de 0025) **se reviven** y se evalúan con **un mismo criterio en todos los problemas del banco**. No se exige que una palanca sirva en dos problemas: se reporta **qué sirve y dónde**, y el cierre de cada EXP anterior se conserva como lo que fue (decisión bajo su criterio), pero ya no limita lo que se reporta aquí.

**Criterio uniforme (por palanca × problema, pareado por semilla con B, sin agrupar problemas):**
- *sirve*: mediana de e_X/e_B ≤ 0.90 y mejora en ≥ 70 % de las semillas, sin semilla > 2×;
- *daña*: mediana ≥ 1.10 y empeora en ≥ 70 %;
- *dependiente de semilla*: ni una ni otra y alguna semilla fuera de [0.5, 2];
- *indiferente*: el resto.
Se reportan siempre mediana, k/n, razón máxima, y la fase (Adam o L‑BFGS) donde aparece el efecto. Umbrales fijados aquí; no se cambian tras ver datos.

**Mapa de palancas** (entregable `levers.md` + figuras): palanca, señal que la gobierna, EXP de origen, código en `src/mechanisms/`, problemas con datos. Palancas: (1) generación/remuestreo de puntos [0014 S, 0025 R]; (2) centros y anchos [0015, 0024‑A]; (3) parametrización de coeficientes [0023: F, J, E, I]; (4) términos y pesos de la pérdida / foco por responsabilidad [0016–0018]; (5) fase del optimizador [0016 traspaso]; (6) forma del residuo / autodiferenciación [medida, sin mecanismo propio: se inventaría]; (7) base/activación [inventario]. Las palancas 6 y 7 no se corren en esta campaña; se inventarían y se discuten en las predicciones.

**Fase 0 (nueva, CPU, ya).** Reanálisis uniforme con el criterio de arriba de **todos los datos guardados** de cada palanca en cada problema donde existan, más figuras: mapa de calor palanca × problema (log mediana e_X/e_B), correlaciones entre efectos de palancas (Spearman por semilla donde haya semillas comunes) y contra las propiedades de la fase 2. Salida: `fase0/` con CSV, figuras y `levers.md`.

**Fase 1 ampliada.** Además de B, F, J, E, I (10 semillas, 4001–4010), brazos de las palancas reviven con el código existente y 5 semillas (4001–4005) por problema: S (0014), R (0025), C (regrid idea A, 0024), W (foco por responsabilidad, 0018), H (traspaso, 0016). Solo se corre un brazo en un problema si su código ya lo soporta; el resto queda en el mapa como «sin datos». El presupuesto total se mide en CPU antes de lanzar y se respeta un tope de 4 h de pared por lanzamiento.

**Análisis estadístico (fase 2 y mapa):** heatmaps, correlaciones de Spearman con intervalo bootstrap, y, donde n lo permita, comparación pareada con permutación; con ≤ 6 problemas, descriptivo declarado.

**Autorización.** El investigador firmó todo el spec y ordenó ejecutar sin más preguntas. Se entiende como autorización de ejecutar CPU y de lanzar los bloques GPU de las fases 1a/1b/1ampliada que cumplan: pruebas previas pasadas, presupuesto medido ≤ 4 h por lanzamiento, y sin cambio de umbrales. Los bloques de las fases 1c y 4 requieren que su mecanismo (residuo vectorial) pase sus pruebas; la fase 3 exige firmar `predictions.md` (acto del investigador sobre el contenido final) antes de cualquier corrida de piezo.

## Enmienda v1.2 (2026-10-23): receta corta para el banco, receta completa para confirmar (Fase 1d)

**Motivo (pregunta del investigador):** la línea base comparable con la literatura es la de la receta completa (v1/v2; piezo con la receta del paper: Adam 20 000, float64, sin L-BFGS). Las fases 1 y 1c usan recetas cortas (Adam 2 000–4 000, L-BFGS ≤ 500 it.) para poder cruzar muchas palancas con semillas pareadas, y por eso sus errores absolutos son mayores que los de la línea base. Los efectos medidos a presupuesto corto **pueden no sostenerse a presupuesto completo** (antecedente: lo ganado en Adam se pierde con L-BFGS, EXP-0011/0012), y piezo se corre con receta completa. Una predicción para piezo basada solo en efectos de receta corta quedaría expuesta a ese salto.
**Fase 1d (confirmación):** tras la fase 1 y la 1c, con receta completa de cada problema (la de su línea base), brazos B y los dos o tres brazos que hayan salido «sirve» o «daña» más claramente en cada problema, 5 semillas, en Allen–Cahn, Burgers, KPP y NS2d-CG. Etiquetas con el mismo criterio uniforme. Es lo que en el brief se llamaba D3, ahora con el papel de ser **condición de la predicción para piezo**: `predictions.md` debe declarar, por problema, si el efecto se sostuvo a receta completa, y pesa más lo confirmado que lo corto.
**Piezo:** su corrida final usa la receta del paper. Si en 1d un efecto se pierde a receta completa en los problemas parecidos a piezo, la predicción para ese brazo baja a «indiferente».
**Lo que no cambia:** no se repite la línea base; los números de línea base ya congelados son el B de referencia absoluta y el B pareado de cada bloque es el control del contraste.

## Enmienda v1.3 (2026-10-24, antes de firmar predictions.md y antes de correr piezo): criterio de H-PRED-1 que no se gane por tasa base

**Motivo:** el cruce (`cruce/cruce.md`) muestra que, con 5 problemas de banco y un objetivo, el predictor ingenuo «F, E e I mejoran, J es indiferente» ya acierta ≥ 3 de 4 brazos por tasa base, así que la regla «≥ 3 de 4 direcciones» (Fase 3) no distingue conocimiento de propiedades de una apuesta. Además las propiedades del banco (n_outputs, n_loss_terms, borde/PDE, rms del residuo) forman una sola partición colineal; el stand-in de piezo cae fuera del rango del banco (dispersión de traza entre aristas de 6.6 décadas frente a 1.7–3.5), de modo que J, E e I se extrapolan.
**Criterio (sustituye al de la Fase 3, fijado antes de ver piezo):** la métrica de piezo es el error al **final de Adam** (la receta del paper no tiene L-BFGS). H-PRED-1 se respalda solo si se cumplen **las dos**: (i) el brazo mejor y el peor previstos coinciden con los observados (entre F, E, I, J); (ii) la magnitud de log10(e_X/e_B) prevista para F y E queda a menos de 0.3 décadas de la observada (mediana de 5 semillas). Además se reporta la tasa base ingenua («todos mejoran salvo J indiferente») y el acierto de H-PRED-1 se declara **solo si supera** a esa tasa en direcciones. Con un objetivo, un acierto es un indicio y no se generaliza; un fallo se reporta con la propiedad que engañó.
**Predicciones que NO se hacen:** si `predictions.md` declara «sin predicción» para un brazo (p. ej. I en piezo: la causa de su efecto, acoplamiento de salidas o ausencia de borde, no se puede separar con el banco), ese brazo queda fuera de (i) y (ii) y se corre igual como exploratorio. Es preferible a una predicción sin base.
**Condición de receta completa (v1.2):** de los efectos que `predictions.md` use como evidencia, solo cuentan los confirmados en 1d para ese problema; en Allen–Cahn el efecto se mantiene al final de Adam (E 0.21, F 0.28) y se debilita tras L-BFGS (E 0.39, F 0.81); como piezo no usa L-BFGS, la comparación relevante es la de fin de Adam.
**Fase 1e (pendiente, tras 1d):** confirmación con receta completa de NS2d-CG y DFG para I, F y E (5 semillas), porque J e I no están cubiertos a receta completa en ningún problema y I es el brazo que más separa al banco.

## Enmienda v1.4 (2026-10-24, instrucción del investigador en el chat)
1. **J e I entran a la confirmación con receta completa (Fase 1e).** Antes de firmar `predictions.md`, J e I se corren con la receta completa de cada problema (5 semillas, 4101–4105, pareadas con el B de 1d) en Allen–Cahn, KPP y Burgers (si la cuota lo permite) y, junto con B, F y E, en NS2d-CG y DFG a receta completa. Orden de prioridad por costo/valor: NS y DFG (B, F, E, J, I), Allen–Cahn y KPP (J, I), Burgers (J, I) al final.
2. **Piezo: se porta el código del paper** (bimorfo PVDF, repo de González, formulación indirecta con la configuración del paper: Adam 20 000, float64, 2 048 interiores + 256 de frontera/interfaz, remuestreo cada 200, bordes duros por levantamiento) como `PiezoProblem` con `residual_terms` vectorial. Validación previa: reproducir la línea base congelada (RBF16 0.092 ± 0.021 en u, n = 5) y la pérdida del código original paso a paso. Las diferencias entre decisiones previas del proyecto y el paper se resuelven a favor del paper y se reportan (pesos, signo de polarización, e31/e33). La licencia (MIT declarada sin archivo LICENSE) queda como nota para el registro de procedencia.

## Enmienda v1.5 (2026-10-24): el benchmark del paper son dos problemas (converso y directo)
Tras leer el paper (González, Higueros, Shaw, Barrientos, *Math. Comput. Appl.* 31, 214, 2026) el benchmark de piezo es el bimorfo PVDF con **dos** problemas: **converso** (100 V arriba, 0 V abajo) y **directo** (fuerza de 0.1 N en el extremo libre, electrodo superior flotante, carga neta cero). La receta del paper: MLP 3×50 SiLU con 4 entradas (ξ, η, η_I, fase p) y 8 campos condicionados por fase, **pesos fijos** (converso 1/1/1; directo λ_BC = 10), bordes duros por transformación, Adam 20 000 lr 1e-3 float64 **sin L-BFGS**, remuestreo cada 200, 2 048 puntos por capa + 256 de frontera/interfaz, y en el converso *mechanical gradient routing* (stop-gradient en las tensiones independientes dentro de los residuos constitutivos mecánicos). Resuelve la contradicción pesos adaptativos (repo) frente a pesos fijos (tesis): **gana el paper, pesos fijos**. Valores de referencia del MLP del paper (validación del porte): converso L2 u 0.107, v 0.150, φ 0.045; directo u 0.117, v 0.153, φ 0.067.
**Consecuencias:** (1) `predictions.md` tiene **dos objetivos** (converso y directo); H-PRED-1 (v1.3) se evalúa en cada uno y se reporta por separado, sin promediarlos. (2) El *routing* hace que el Jacobiano del Gauss-Newton no coincida con el gradiente exacto del converso: se declara cómo entra al mecanismo. (3) El «indirecto» de EXP-0009 corresponde al converso y su línea base congelada se compara solo con él; el directo no tiene línea base propia y su B pareado es el control.

## Enmienda v1.6 (2026-10-24): registro de artefactos internos por corrida
**Hallazgo de auditoría (pregunta del investigador):** las corridas de las fases 1, 1c, 1d y 1e guardan por corrida un JSON con: historia de pérdida (20–400 puntos), error al final de Adam y al final de L-BFGS, pérdida por término (solo NS/DFG), diagnóstico de cada recálculo (`pre_diag`: segundos, puntos, cond de P), trayectoria de L-BFGS (1d), reloj y semilla. **No se guardan:** pesos/coeficientes/centros del modelo, campos de solución, residuos por punto, espectros de P, ocupación de bases ni normas de gradiente. Solo se conserva un checkpoint de fin de Adam para B en Allen–Cahn (20 corridas). Por tanto el análisis de interpretabilidad (mapas de error, residuo, asentamiento) no es posible con lo ya corrido en NS/DFG/KPP/Burgers.
**Regla desde ahora:** el bloque de piezo (converso y directo) guarda artefactos completos (`piezo/artifacts.py`: state_dict a intervalos, campos en la malla 201×21, residuos por término y por punto, historia de pérdida por término, normas de gradiente, espectros de P y cond por bloque, ocupación de bases, consistencia constitutiva, hash de código y datos). Para el banco, si el presupuesto lo permite se **repite una rebanada** con artefactos (receta corta, 2 semillas por brazo en Allen–Cahn, Burgers, KPP, NS y DFG; ~1–3 min por corrida en T4) para producir los mapas comparativos sin gastar los bloques grandes. Los resultados ya puntuados no cambian.
