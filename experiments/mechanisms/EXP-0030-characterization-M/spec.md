---
experiment_id: EXP-0030
slug: characterization-M
status: preregistered     # draft | preregistered | running | scored | decided
suite: exploratory_post_hoc_cpu   # sin suite del verificador: solo lectura de EXP-0029
problem_id: piezo_bimorph_pvdf (converso y directo)
model_id: RBF-PIKAN (receta del paper, float64, Adam 20 000, sin L-BFGS)
mechanism_id: gram_preconditioning, brazo M (comparado con B y E)
hypothesis_id: ninguna (exploratorio; no se confirma nada)
preregistered_at: 2026-10-26
preregistered_by: Rodrigo Mansilla (por chat; exploratorio: la firma congela la lista de calculos y las decisiones declaradas)
---

# EXP-0030 — Caracterización de los resultados del mecanismo M en piezo

**Estado: borrador sin firmar (2026-10-26).** Redactado por el agente a pedido del investigador. Nada de lo que sigue es un pre-registro confirmatorio: el investigador ya vio los resultados de EXP-0029 (`results/EXP0029_vs_predicciones.md`, `LECCIONES_EXP0029.md`). Esta spec fija *qué se calcula y cómo se reporta* antes de calcular las cantidades nuevas, y declara de entrada que todo es **exploratorio y post hoc**.

## Pregunta

¿Cómo se comportan B, E y M en piezo más allá del error del último paso: dónde está la diferencia (efecto, cola, tiempo de salida del llano), de qué parte del mecanismo viene (cuerpo E o patas), qué ocurre con cada ecuación y campo, y cuánto de lo observado resiste un cambio en las semillas consideradas?

## Hipótesis

No hay. Es un estudio de caracterización: no se declara una dirección esperada ni un delta que cuente. Las predicciones firmadas de EXP-0029 (`DISENO_FINAL.md` §3) **no se reevalúan ni se modifican aquí**; su veredicto vive en `results/EXP0029_vs_predicciones.md`.

## Datos de entrada (solo lectura, sin cómputo nuevo)

Fuente: `experiments/mechanisms/EXP-0028-failure-attribution/results/`.

| Dato | Dónde | Cobertura |
|---|---|---|
| M, converso | `piezo-exp0029-final/piezo_exp0029/` | semillas 20260728–30 (n = 3); 31–32 fallaron por memoria, **no existen** |
| M, directo | ídem | semillas 28–30 (n = 3) |
| B y E, converso | semilla 28 en `piezo-s1a/exp0028_piezo/`; 29–30 en `piezo-modal-run/piezo/`; 31–32 en `piezo-exp0029-final` | n = 5 |
| B y E, directo | semilla 28 en `piezo-s1b/exp0028_piezo/`; 29–30 en `piezo-modal-run/piezo/` | n = 3 |
| S y E con `ws4000` (T1/T2) | `piezo-exp0029/`, `piezo-exp0029-t2/` | n = 3, horizonte corto |
| Referencia FEM | `fem_converse_reference.npz`, `fem_direct_reference.npz`, `fem_flux_*.npz` | malla `xy`, campos `u`, `v`, `phi` |

Contenido verificado de un artefacto de M (converso, semilla 28, 1 344 arreglos, ≈ 21 MB): estados del modelo (`s000000` … `s020000`, coeficientes, `pre_P`), `refreshNN` con traza, condición y cuotas por grupo (`gtrace_mec`, `gtrace_elec`, `gtrace_equil`, `gtrace_borde`), `settle/stepNNNNNN`, historial `hist/*` con 16 términos de pérdida; el JSON trae `hist` (cada 500 pasos: `l2re_u`, `l2re_v`, `l2re_phi`), `loss_trace` por paso, `pre_events`, `terms_end`, `score_final`, `score_flux`, `interface_gap`, `run_card`.

**Por verificar antes de firmar** (no se asume): (i) si B y E de las semillas 28–30 (ya ubicados) traen estados y términos, no solo el JSON; `LECCIONES_EXP0029.md` indica que las corridas viejas de E registran la pérdida cada 100 pasos, no por paso; (ii) que los estados reproducen `terms_end` al recalcular sobre el pool de entrenamiento (prueba de cordura de D); (iii) dónde queda registrado el `b` elegido por P2 en cada recálculo.

## Métricas y salidas (todas descriptivas)

No hay métrica primaria de decisión. Las salidas son tablas y figuras por **(problema, brazo, semilla)**, con la columna de hardware (T4, H100, A100) siempre visible.

### Bloque A — Efecto
- `l2re_u` y `l2re_phi` al paso 20 000 (el reportado en EXP0029).
- **Métrica robusta (post hoc):** mediana de las evaluaciones de los pasos 18 000–20 000 (`eval_every = 500`: cinco puntos). Se declara aquí como secundaria y se rotula *post hoc*.
- Tiempo de salida del llano: primer paso evaluado con `l2re_u < 0.5`.
- Cola: mínimo, mediana y máximo por brazo.
- Razones pareadas por semilla M/E, M/B y E/B: valores individuales, mediana y rango. **Sin intervalos de confianza ni p-valores** (n ≤ 5).

### Bloque B — Estabilidad de la conclusión frente a las semillas
Para E frente a B en el converso (n = 5): victorias y mediana de la razón E/B en cada uno de los 5 subconjuntos de 4 semillas y en los 10 subconjuntos de 3. Mismo cálculo para M frente a E y B sobre las 3 semillas disponibles (dejar una fuera). Se reporta cuántos subconjuntos dan cada signo. Es un conteo combinatorio, no una prueba.

### Bloque C — Atribución (B → E → M)
- Efecto del cuerpo: razones E/B. Efecto de las patas: razones M/E.
- Señales registradas en cada recálculo: cuotas por grupo (mecánica, eléctrica, equilibrio, borde), condición efectiva (mediana y p90 por capa), `b` elegido, eventos de P4 en sombra (`pre_events`). Se presentan como trayectorias por paso, junto al error de cada semilla.
- T1/T2 (S y E, `ws4000`) se reutilizan tal como están, sin recalcular.
- Sin correlaciones entre señales y resultado a nivel de semilla (n demasiado pequeño); solo se muestra la relación a nivel de arista dentro de una corrida, como descripción.

### Bloque D — Física
1. **Términos de pérdida** (`hist/term/*`: constitutivas, equilibrio, Gauss, bordes): razón M/E por término y mediana de las evaluaciones finales. Son términos sobre el *pool de entrenamiento*: se rotulan así.
2. **Residual por ecuación sobre malla reservada**, recalculado en CPU float64 desde los estados (`s020000` y algunos pasos intermedios) con `piezo_problem.py` y `residual_terms.py`. Prueba de cordura previa: reproducir `terms_end` sobre el pool de entrenamiento.
3. Error por campo ($u$, $v$, $\phi$), error de flujo (`score_flux`) y `interface_gap`.
4. **Chequeos de admisibilidad** que se miran *antes* de comparar residuales, a la manera de [Musah 2026] (lectura de resumen): carga neta del electrodo flotante (directo), trazas de borde, consistencia constitutiva y las simetrías de geometría y polarización que `piezo/ALINEACION_PAPER.md` permita enunciar (**lista a cerrar y verificar antes de firmar**). Se reporta el valor de cada chequeo; no se fija umbral de aprobación.

### Bloque E — Sensibilidad retrospectiva (sustituye a Morris)
Efecto descriptivo de cada factor presente en los datos (cuerpo, patas, problema, hardware) sobre la métrica robusta, como diferencias de medianas del logaritmo de la razón. No es una descomposición de varianza. **Morris queda diferido** hasta que haya cómputo; su diseño (factores: tope de P2, cadencia de recálculo, piso de P4, calendario de P5; trayectorias; medida robusta con pocas muestras) se deja como nota, no como compromiso.

### Bloque F — Incertidumbre
Campos predichos sobre la malla `xy` de la referencia a partir de `s020000`; mediana y rango intercuartílico entre semillas por brazo; correlación de Spearman entre el mapa de dispersión y el mapa de error, **solo como descripción** (n = 3 o 5).

### Bloque G — Responsible AI (documentación, no cálculo)
- *Experiment card* por brazo: hardware, versiones de bibliotecas, semillas, firmas (`run_card`, `meta_json`), costo (s/paso, horas, USD).
- Proveniencia: qué escribió el agente (scripts de este análisis) y qué decidió una persona (firma, métrica robusta, lectura).
- Dominio de validez: un bimorfo PVDF, dos problemas (converso, directo), receta del paper, float64, Adam 20 000 sin L-BFGS, n ≤ 5.
- Catálogo de fallos: OOM de M converso 31–32; saltos de pérdida; E semilla 32 (0.14 → 0.09 → 0.52).
- Limitaciones de comparabilidad: hardware mixto (el pareo por semilla no es exacto en float64 entre dispositivos), n distinto por brazo.

### Bloque H — Alineación de gradientes (añadido 2026-10-26, post hoc; ver `CARACTERIZACION_PLAN.md`)
Gradiente de la pérdida ponderada de cada grupo físico (mec, elec, equil, borde) respecto a los coeficientes efectivos y, para E y M, respecto a c′ (lo que ve Adam), en `s000000, s002000, s004000, s010000, s020000` de B, E y M (semillas 28–30, ambos problemas, 128 puntos por clave, CPU float64). Salidas descriptivas: coseno por pares, razón de normas (conflicto tipo I) y dispersión de la norma del gradiente entre aristas. Marco: Wang et al. (arXiv:2502.00604, resumen). Sin umbrales ni pruebas.

## Decisiones declaradas

1. Sin p-valores ni intervalos inferenciales con n ≤ 5 (preferencia del investigador).
2. Sin cotas numéricas de decisión para la física: las razones por ecuación se reportan tal cual. Se conservan solo las cotas ya firmadas en EXP-0029 (φ ≤ 1.20, costo ≤ 1.10). La literatura consultada no ofrece un margen estándar; un margen de equivalencia, de usarse, debe salir del ruido entre semillas, no de una constante.
3. La métrica robusta (A) es post hoc y así se rotula.
4. Cualquier cantidad que se añada **después** de ver salidas se registra en la tabla «Desviaciones» al final, con fecha y motivo.

## Condición de falsación

No aplica: no hay hipótesis. Se declara en cambio qué **no se afirmará** con esta spec: que M, E o las patas «mejoran» o «empeoran» el error en general; que un patrón entre semillas sea estadísticamente distinguible del azar; que la sensibilidad a hiperparámetros esté caracterizada.

## Presupuesto

| Campo | Valor |
|---|---|
| Suite / tier | exploratorio, Tier 0–1 |
| Backend | `local_cpu` (Python 3.12 con numpy y torch) |
| GPU | 0 horas |
| Corridas de entrenamiento | 0 |
| Datos | ≈ 651 MB, solo lectura |
| Riesgo de tiempo | el recálculo de D(2) y F sobre mallas densas; medirlo en una semilla antes del resto |

Pendiente fuera de esta spec: M converso 31–32 (≈ 3.5–4 USD en A100). El investigador decidió no gastar más cómputo; queda como limitación.

## Superficie de edición declarada

El agente puede escribir únicamente en:
- [x] `experiments/mechanisms/EXP-0030-characterization-M/` (`analysis/`, `results/`, `figures/`)

No puede tocar `evals/`, `src/mechanisms/`, el modelo, la definición de la PDE, `results/` de EXP-0028/0029 ni `registry/claims.jsonl`. Los scripts leen los artefactos sin modificarlos.

## Amenazas de validez

- **Post hoc / garden of forking paths:** el investigador ya conoce los resultados; los bloques reflejan lo que se vio. Mitigación: declararlo, registrar desviaciones y no derivar claims.
- **Leakage entrenamiento/evaluación:** los términos `hist/term/*` se calculan sobre el pool de entrenamiento; la comparación física (D2) debe hacerse sobre la malla reservada.
- **Hardware mixto y n pequeño:** B y E 28–30 en T4 y H100, el resto en A100; M converso solo n = 3.
- **Un estado cada 1 000 pasos** para la reconstrucción de campos; el error evaluado cada 500.
- **Conclusión inestable a las semillas:** ya ocurrió (E frente a B pasó de 0.198 → 0.053 con n = 3 a medias iguales con n = 5). El Bloque B existe para cuantificar ese riesgo.
- **Referencias de literatura:** leídas como resúmenes o fragmentos, no completas; no citar sin verificar el texto.

## Literatura de apoyo (estado de verificación)

| Fuente | Se usa para | Estado |
|---|---|---|
| Musah (2026), arXiv:2608.21683 | un residuo agregado puede elegir la peor solución; puerta de admisibilidad antes de comparar | resumen leído |
| Benavoli et al. (2017), JMLR 18(77) | región de equivalencia práctica (ROPE) | fragmentos |
| Bouthillier et al. (2021), MLSys | varianza entre semillas en comparaciones | resumen |
| Morris (1991); Campolongo et al. (2007); SALib | Morris/μ\* (diferido) | resumen y documentación |
| Guo et al. (2020), arXiv:2010.12060 | precedente PINN con Morris (30 trayectorias, 4 niveles) | fragmento |
| fANOVA / Sobol | alternativas de sensibilidad (diferido) | resumen |

## Salida y destino

Los resultados alimentan `ProyectoGrad/j2-resultados.tex` y `j3-discusion.tex` (el flujo es: correr aquí, escribir allá). Ninguna afirmación pasa al texto sin que el investigador la acepte. Se esperan a lo sumo un `decision.md` con `decision: deferred` o `inconclusive` y ningún claim.

## Desviaciones

| Fecha | Qué se añadió o cambió | Motivo |
|---|---|---|
| 2026-10-26 | Verificación de los puntos «Por verificar»: (i) B y E de las semillas 28-30 traen estados cada 2 000 pasos, campos y residuales en malla en 4 instantes y `hist/term` cada 100 pasos, **sin `loss_trace` por paso**; los saltos de B/E 28-30 y de M no son comparables con la misma definición. (iii) `b_mix` queda en `pre_diag` del JSON (en E es solo el valor por defecto, no se aplica). | Lectura de los artefactos antes de correr. |
| 2026-10-26 | **Confusión total hardware × brazo en M frente a E.** Hardware por corrida: B y E semilla 28 en T4; semillas 29-30 en H100; B y E semillas 31-32 en A100-80GB; **M converso y directo 28 en A100-80GB, M directo 29-30 en A100-40GB**. No existe ningún par M/E con el mismo hardware y la misma semilla. Además, las únicas semillas de E donde no se ve la ventaja sobre B (31, 32) son las de la A100. Efecto de las patas (Bloque C) y factor hardware (Bloque E) **no son identificables** con estos datos; se reportan rotulados como confundidos. | El pareo por semilla ya se había declarado inexacto entre dispositivos; falta decir que para M frente a E es total. |
| 2026-10-26 | La ejecución se pidió por chat («si está bien, correla») sin cambiar `status` a `preregistered`: se corre como análisis exploratorio con la spec en borrador; la firma queda pendiente del investigador. | Instrucción del investigador; solo lectura, sin costo. |

| 2026-10-26 | Bloque H (alineación de gradientes) añadido y corrido; `CARACTERIZACION_PLAN.md` fija nueve preguntas (Q1–Q9) y el veredicto por método de xAI; `CIERRE_OBJETIVO3.md`, citado en el registro, no existe en el repositorio y se redactará de nuevo. | Petición del investigador: caracterización más profunda, con literatura y preguntas explícitas. |

| 2026-10-26 | Bloques I (invariantes integrales de equilibrio; sustituyen a las simetrias, porque `ALINEACION_PAPER.md` no enuncia ninguna), J (dinamica efectiva y paso de separacion), K (uso de aristas y similitud de funciones de la capa 0), L (probabilidad de mejora y perfiles) corridos en CPU; `CIERRE_OBJETIVO3.md` reescrito. El investigador respondio «dale» a seguir con esos cinco puntos; la spec sigue en `draft` hasta que lo firme. | Cierre del objetivo 3 con las preguntas Q1-Q9. |

## Firma

~~Este borrador no está firmado.~~ **Firmada el 2026-10-26 por el investigador en el chat** («Firmo la spec de EXP-0030 (status preregistered, exploratorio)»); las cantidades de los bloques A–L se congelan como exploratorias y post hoc. Entra en vigor cuando el investigador cambia `status` a `preregistered` y añade fecha y nombre; para un estudio exploratorio, esa firma solo congela la lista de cálculos y los puntos «Decisiones declaradas».
