# Brief de decisión para el investigador — 2026-09-28

Autor: agente de investigación (solo lectura sobre evals y registry).
Este documento **recomienda**; no acepta nada. Las decisiones son tuyas (gates `scientific_claim` y `eval_integrity`). Todo sigue en `scientific_accepted=false`.

Los datos de DFG-FEM y del rerun de Heat B-spline están **pendientes (Codex en curso)**; no se leyeron ni se usan aquí. Los archivos que Codex edita se leyeron tal como estaban al inicio; si cambian, revisa la sección "qué desbloquea" de cada decisión.

---

## 1. Qué significan los números

| Término | Qué es, en cristiano | Ejemplo concreto |
|---|---|---|
| **L2 relativo** | Distancia entre la solución de la red y la solución "verdadera", medida sobre todo el dominio y dividida por el tamaño de la solución verdadera. Es un "porcentaje de error medio" del campo entero, no de un punto. Penaliza más los errores grandes; un frente abrupto mal colocado pesa mucho. | 0.0165 = ~1.65 % de error relativo medio. 0.0001 = 0.01 %. |
| **media ± desviación estándar, n=5** | La misma receta se entrenó 5 veces con semillas distintas. "Media" = promedio; "±" = cuánto varían las 5 corridas entre sí. Con n=5 la SD es poco fiable. | RBF Allen–Cahn 0.0165 ± 0.0024: casi todas las corridas entre 0.014 y 0.019. MLP 0.0788 ± 0.0177: entre 0.06 y 0.10. Los rangos no se tocan. |
| **Pareado** | Cada corrida de A se compara con la de B que usó **la misma semilla y los mismos puntos**; se analiza la diferencia por semilla. Elimina la suerte compartida y es mucho más potente con n=5. | AC semillas 401–405, RBF−MLP: −0.039, −0.077, −0.053, −0.068, −0.074. Las 5 son negativas: el RBF gana 5 de 5. |
| **No pareado** | Las corridas de A y B no comparten semilla ni puntos (o vienen de otro presupuesto/versión). Solo se comparan medias y dispersiones, y las diferencias de protocolo se mezclan con las del método. | Heat: DeepXDE usó semillas 11,22,33,44,55; nuestros modelos 901–905. |
| **Referencia candidata vs aceptada** | La "referencia" es la solución que se toma como verdad para medir errores. *Candidata*: la calculó un agente y tiene buena pinta. *Aceptada*: tú decidiste que sirve para claims. Hoy ninguna está aceptada. | AC: la referencia FD/RK4 (8192 intervalos) es candidata. |
| **Por qué importa 1e-7 entre dos solvers** | Una referencia sirve si su error es **mucho menor** que el error que quieres medir (regla práctica: >=10x, mejor >=100x). Si dos métodos independientes (diferencias finitas+RK4 y pseudo-espectral+ETDRK4) coinciden a 3.8e-7, es muy improbable que ambos compartan un error grande. | Error del modelo 1.65e-2 ÷ 3.84e-7 = **~43,000 veces**. Tras L-BFGS (9.16e-3): ~24,000 veces. La referencia es ~4 órdenes de magnitud más exacta que lo que mide. |
| **Cd, Cl, Δp (DFG)** | Cd = arrastre (fuerza en la dirección del flujo sobre el cilindro); Cl = sustentación (transversal; casi cero por simetría, por eso su error relativo engaña); Δp = caída de presión delante/detrás del cilindro. Son 3 números resumen publicados por FEATFLOW. | Referencia: Cd=5.5795, Cl=0.0106, Δp=0.1175. RBF (301–305, tras L-BFGS): Cd=5.12 (8.2 % de error; corregido 2026-09-30, antes 1.46 y 73.8 % por fórmula errónea), Δp=0.106 (9.9 %). |
| **screening vs comparable_to_literature** | *Screening*: prueba exploratoria interna; sirve para elegir qué probar, no para afirmar nada externo. *comparable_to_literature*: reprodujimos el **mismo problema, evaluador y métrica** de un artículo y el número coincide. Es un chequeo de "nuestro baseline no está roto", no un "somos SOTA". | Burgers: 0.01326 (nuestro) vs 0.0133 (PINNacle) con el mismo archivo FEM = comparable. AC vs AEH-PINN = screening. |

Regla de lectura que aplico a todo: **una diferencia solo cuenta si es mayor que la dispersión entre semillas Y viene de un protocolo comparable.**

---

## 2. Las decisiones, reducidas

Quedan **4 decisiones** más un aviso (DFG, no decidible aún). D1 desbloquea Allen–Cahn en D4.

### D1. ¿Usar la referencia Allen–Cahn (ETDRK4/FD8192) para puntuar modelos en claims?

**Recomendación: ACEPTAR con redacción acotada** ("referencia numérica de trabajo verificada por dos solvers independientes; no por la tabla `.mat`").

**Justificación.** Dos solvers de familias distintas (diferencias finitas+RK4 y pseudo-espectral Fourier-seno+ETDRK4 en float64) coinciden a 3.8e-7 (relativo) y 5.5e-6 (máximo absoluto) en 512x256. ETDRK4 converge por sí solo: pasar de N=4096 a 8192 mueve el resultado 9.3e-13; reducir dt a la mitad, 1.3e-11. Además hay un estudio BDF previo (AC-L4) que pasó sus umbrales pre-declarados (último par espacial 4.6e-6; control temporal 2.4e-10). Es más validación de la que documentan los artículos comparables (sección 3: AEH-PINN no describe cómo validó su referencia; PINNacle usa COMSOL/Chebfun sin publicar convergencia).

**Evidencia, en lenguaje llano.**
- FD vs ETDRK4: 3.84e-7 → coinciden en ~7 cifras. El error del modelo (1.65e-2) es ~43,000x mayor.
- Convergencia ETDRK4: 1e-11 a 1e-13 → la solución ya no se mueve al refinar.
- Contra el `.mat` oficial de DeepXDE (interpolado a 201x101): 1.57e-4 (máx 4.9e-4). Es ~400x peor que el acuerdo entre nuestros solvers; el `.mat` (malla 0.01 + interpolación) es la pieza menos exacta. Aun así es ~100x menor que el error del modelo (1.65e-2) y ~130x menor que el de DeepXDE oficial (2.0e-2), así que **no cambia ningún ranking**.

**Riesgo si me equivoco.** Que ambos solvers compartan un error sistemático (p. ej., IC/BC mal implementadas igual en los dos). Mitigación parcial: el contrato coincide con el `.mat` a 1.6e-4, lo que descarta un error de física mayor que ~1e-3.

**Qué cambiaría mi recomendación.** (a) Una revisión de código que encuentre IC/BC compartidas entre solvers; (b) que la discrepancia con el `.mat` no provenga de la interpolación (no está demostrado; el protocolo decía investigarlo antes de concluir); (c) un número BDF vs ETDRK4 en malla común: no lo vi en el expediente y sería la tercera pata más barata.

**Lo que NO autoriza D1:** decir que superamos a AEH-PINN, ni un claim por sí solo.

### D2. ¿Qué frase sobre Burgers vs PINNacle es defendible?

**Recomendación: ACEPTAR con redacción más estrecha.**

Frase sugerida: *"Con la receta oficial de PINNacle (MLP 5x100, Adam→L-BFGS, 20k épocas), sobre el archivo FEM oficial `burgers1d.dat` y su métrica L2RE, obtuvimos 0.01326 ± 0.00003 (n=5), frente a 0.0133 ± 0.00005 publicado (n=3)."* Es una **verificación de reproducción del baseline**, no un resultado de rendimiento.

**Evidencia, en lenguaje llano.**
- Nuestro 0.013264 vs publicado 0.0133: diferencia 3.6e-5, menor que la SD publicada (5.3e-5). Coinciden a 3 cifras.
- Misma instancia (ν=0.01/π, IC −sin(πx)), mismo archivo con hash Git fijado, misma métrica.

**Advertencia que no está en el expediente y cambia la lectura.** Contra la solución analítica (Cole–Hopf) el mismo modelo tiene error 5.7e-4, **23 veces menor** que contra el FEM (1.3e-2). O sea: el "1.33e-2" de PINNacle está dominado por lo grueso del FEM de referencia (101x11 puntos), no por el modelo (inferencia mía a partir de tus dos scores). Coincidir con PINNacle demuestra que *nuestro pipeline es igual al suyo*, no que el modelo sea competitivo. Además vRBA reporta ~1e-8 en el mismo PDE (malla/evaluador no declarados), lo que muestra que 1e-2 está lejos del techo de la literatura. "Comparable a PINNacle" es correcto; "competitivo con SOTA" **no**.

**Riesgo si me equivoco.** Que se lea "comparable a PINNacle" como "buen resultado". Se evita con la redacción.

**Qué cambiaría mi recomendación.** Que el `burgers1d.dat` difiera del de PINNacle (el hash Git lo cubre); o querer un claim de precisión, que exigiría replicar vRBA (5k Adam + 60k SSBroyden, fuera de presupuesto).

### D3. ¿Qué frase sobre Heat 1D es defendible (RBF vs DeepXDE; B-spline pendiente)?

**Recomendación: DIFERIR el claim comparativo; permitir solo la frase interna descriptiva.**

Frase permitida hoy: *"En Heat 1D (α=0.1, analítica, 512x256), con 20,000 pasos Adam, RBF-PIKAN alcanzó 2.06e-6 ± 2.3e-7 (n=5) y la reproducción DeepXDE 1.12.2 α=0.1 5.8e-4 ± 4.0e-4 (n=5); comparación descriptiva, semillas distintas, arquitectura y optimizador distintos."*

**Evidencia, en lenguaje llano.**
- RBF S2 2.06e-6 vs DeepXDE 5.8e-4 → ~280x menor. Las dispersiones no se tocan, así que es real *como descripción*, pero no atribuible al método: cambian arquitectura (RBF vs MLP tanh), puntos (Sobol vs Hammersley), precisión y optimizador (nuestro L-BFGS se detuvo temprano en 4 de 5).
- **El resultado depende del presupuesto**: el mismo RBF dio 2.2e-4 (archivado), 1.7e-4 (v4, 500 s) y 2.1e-6 (S2, 20k pasos): ~100x solo por pasos. Un número que se mueve 100x con el presupuesto no es propiedad del método.
- **Inversión de ranking**: en v4 (tiempo fijo) B-spline < RBF (3.4e-5 vs 1.7e-4); en S2 (20k pasos) RBF < B-spline (2.1e-6 vs 7.6e-5), pero el B-spline fue **truncado** a ~7,800 pasos por el tope de 680 s. Se compara un RBF completo contra un B-spline incompleto: no dice cuál es mejor.
- DeepXDE con α=0.4 (1.5e-3) no es el mismo problema.
- No hay cifra publicada de Heat 1D α=0.1 en L2 relativo (sección 3).

**Riesgo si me equivoco.** Aceptar hoy un "RBF 280x mejor que DeepXDE" que un revisor desmontaría con presupuesto por pasos igualado.

**Qué desbloquea (pendiente Codex).** El rerun de B-spline con pasos completos. Si llega a 20k pasos, mi recomendación pasa a ACEPTAR una frase descriptiva del tipo "ambos PIKAN con pasos igualados quedan >100x por debajo de la receta DeepXDE, no pareado", declarando que DeepXDE no se re-entrenó con las semillas 901–905. Si sigue truncado: `screening_only`.

### D4. ¿Congelar algún baseline (gate `eval_integrity`)?

**Recomendación: DIFERIR. No congelar nada hoy.** Orden futuro: (1) Burgers-PINNacle, (2) Allen–Cahn tras D1 (RBF con L-BFGS), (3) Heat tras el rerun.

**Justificación.** Congelar fija la vara: los umbrales de aceptación de mecanismos se derivan del baseline congelado y "un baseline congelado no se edita, se supersede" (specs/proceso-de-ingenieria.md, reglas 3 y 4; `freeze_baseline` requiere tu gate). Dos hechos:
1. El proceso indica que el bloqueante es firmar EXP-0001 y congelar `heat_1d@model.pikan.baseline`; Heat es el problema más inestable (D3).
2. Los baselines de este expediente son scorecards de comparación con literatura, no necesariamente el baseline de mecanismo (`model.pikan.baseline`). Congelar "PINNacle Vanilla" no fija el umbral de un mecanismo sobre RBF-Sobol.

**Riesgo si me equivoco.** Congelar pronto: fijas una vara que habrá que superseder (coste bajo). No congelar: el loop de mecanismos sigue en `threshold_not_defined`. Esperar 1–2 días cuesta menos.

**Qué cambiaría mi recomendación.** Necesitar arrancar hoy la campaña de mecanismos: congela primero Burgers (única con reproducción bajo evaluador del paper, dispersión mínima, referencia ya cerrada).

### DFG Re=20 (no decidible aún)

**No hay decisión hoy.** Falta la referencia FEM convergida (dos intentos fallaron por infraestructura: `base64` y `gmsh`); Codex la rehace (pendiente). Sin campo FEM no hay error L2 de u/v/p, y sin comparador neural publicado del mismo contrato (no lo encontré) no hay comparabilidad.

Cómo leer lo que hay: Cd está 8.2 % bajo la referencia en 301–305 tras L-BFGS (5.12 vs 5.58; corregido 2026-09-30: el 73.8 % / 1.46 usaba la fórmula tangencial errónea), Δp a 9.9 %. La cohorte 501–505 (Cd a 5.3 %) es **independiente**; con la medición corregida sí se observa que L-BFGS mejoró Cd en 301–305 (28.9 % → 8.2 %). **Inconsistencia interna (RESUELTA 2026-09-30):** 30 % = σ·n antes de L-BFGS; 74 % = fórmula tangencial errónea; 5–6 % = 501–505 con σ·n tras L-BFGS. Texto original: `reference-readiness.md` reporta un re-score con tracción σn (Cd 3.89, error 30 %) que difiere de las otras lecturas; antes de citar cualquier Cd hay que aclarar qué evaluador de fuerzas está vigente.

**Qué desbloquea.** Un FEM P2/P1 con Cd/Cl/Δp cerca de FEATFLOW (criterio del plan: 0.5 % / 5e-4 abs / 0.5 %) y estable entre los dos niveles finos habilitaría una futura "D5: aceptar referencia de campo DFG" y el error L2 de campo de las cohortes. Nada de DFG debe entrar a un claim antes.

---

## 3. Evidencia comparable de la literatura

Lecturas externas de esta sesión (solo lectura, URL anunciada antes de leer): `https://arxiv.org/html/2512.17607` (AEH-PINN) y `https://arxiv.org/html/2306.08827` (PINNacle). El resto viene de `literature-heat-burgers.md` y `literature-ac-dfg.md`, sin re-verificar por mí.

### 3.1 Allen–Cahn 1D (d=0.001, IC x²cos(πx), Dirichlet −1)

| Fuente | Resultado | ¿Mismo PDE? | ¿Misma métrica/referencia? | ¿Mismo presupuesto? | Lectura |
|---|---|---|---|---|---|
| Gao & Yang, AEH-PINN, arXiv 2512.17607 (2025). https://arxiv.org/html/2512.17607 | ReL2 = 8.12e-5 (media de 10 repeticiones); SAPINN 2.11e-4; RAD 1.32e-3 | Sí | **No verificable**: §4.5 no describe cómo se calculó la referencia ni la malla de test | No: float32, 4x50 tanh, Adam hasta 100k épocas; nosotros float64 y otros pools | Solo orienta: 8.12e-5 vs nuestro 9.16e-3 (RBF+L-BFGS) ≈ 113x; vs 1.65e-2 ≈ 203x |
| DeepXDE, demo oficial. https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/allen.cahn.html | Sin error tabulado; nuestra reproducción 2.0e-2 ± 6e-3 | Sí | Referencia `.mat` sin procedencia documentada | Sí (receta oficial) | Baseline de receta |
| Wang et al., gPINN, arXiv 2111.02801. https://arxiv.org/abs/2111.02801 | PINN estándar ~1 % con ~4,000 puntos; gPINN+RAR ~0.1 % con ~1,500 | Sí | No (curvas por número de puntos) | No | Orden de magnitud: 1e-2 con PINN estándar es esperable; nuestros 1.65e-2 y 9e-3 están en ese rango |
| PINNs with Learnable Quadrature, NeurIPS 2025. https://papers.neurips.cc/paper_files/paper/2025/file/382b95a6580cfb0b5ca33c74b4e0e770-Paper-Conference.pdf | Sobol 0.20 ± 0.10; LearnQuad 0.03 ± 0.008 (n=5, 1,000 puntos) | Sí | Referencia FD, test 10x más denso | No (1,000 puntos) | Otro régimen de puntos; no comparable en número |
| PirateNets, JMLR 2024 | — | **No** (d=1e-4, periódico) | — | — | Solo contexto |

**Práctica de referencia (pregunta pedida).** En lo que pude leer, los artículos con AC (AEH-PINN §4.5; PINNacle usa COMSOL/Chebfun) **no publican verificación con un segundo solver independiente**, y DeepXDE distribuye un `.mat` sin procedencia. Un acuerdo de 3.8e-7 entre dos solvers más convergencia interna es por tanto **más de lo habitual**, no menos. Límite: no revisé el código/apéndice de AEH-PINN ni RBA/vRBA para AC; podría haber detalles allí.

**Hallazgo que cambia la comparación con AEH-PINN.** Nuestro `.mat` interpolado difiere de nuestra referencia en 1.57e-4, **mayor** que el ReL2 de AEH-PINN (8.12e-5). Si AEH-PINN puntuó contra una referencia con error del orden del `.mat` (plausible, no verificado), su número estaría bajo el ruido de su propia referencia y la "brecha de 203x" no sería medible. Conclusión honesta: **no se puede afirmar una brecha de 113x–203x**, solo que estamos por encima de su número declarado por >=1 orden de magnitud. Falta reproducirlo (100k épocas x 10; fuera de presupuesto).

### 3.2 Burgers 1D canónico (ν=0.01/π)

| Fuente | Resultado | ¿Mismo PDE? | ¿Misma métrica/referencia? | ¿Mismo presupuesto? | Lectura |
|---|---|---|---|---|---|
| Hao et al., PINNacle, NeurIPS 2024, arXiv 2306.08827. https://arxiv.org/html/2306.08827 · https://github.com/i207M/PINNacle | Vanilla 1.45e-2; +L-BFGS 1.33e-2 ± 5.3e-5 (3 repeticiones); referencias COMSOL (FEM) o Chebfun | Sí | **Sí** (mismo `burgers1d.dat`, L2RE) | Sí (20k épocas, Adam→L-BFGS) | Nuestro 0.013264 ± 2.9e-5 (n=5) coincide. Único punto realmente apples-to-apples del expediente |
| Toscano et al., vRBA, npj AI 2026. https://www.nature.com/articles/s44387-026-00084-4 | ~1e-8 (baseline 2.78e-8; vRBA 8.4e-9) | Sí | **No**: malla y referencia no declaradas | No: 5k Adam + 60k SSBroyden, red 3x30 | Techo de precisión conocido; muestra que 1e-2 no es SOTA. No reproducido |

### 3.3 Heat 1D (α=0.1, sin(πx), Dirichlet 0)

**Hallazgo negativo importante:** PINNacle **no incluye Heat 1D** (sus casos de calor son Heat2d-VC/MS/CG/LT; Heat2d-VC vanilla 1.01e0, lectura de hoy) ni Allen–Cahn en sus 22 casos. No encontré un L2 relativo publicado para este PDE exacto.

| Fuente | Resultado | Comparabilidad |
|---|---|---|
| DeepXDE, tutorial Heat. https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/heat.html | Sin error tabulado; usa α=0.4 | Receta reproducible. El 5.8e-4 es **nuestra** reproducción con α=0.1, no un dato publicado |
| HyPINO, NeurIPS 2025. https://arxiv.org/html/2509.05117 | MSE 2.3e-2 (HyPINO), 8.0e-5 (HyPINO^10); afinado + L-BFGS 1.6e-9 | Mismo PDE, métrica distinta (MSE) y método con preentrenamiento supervisado (77 M parámetros, 4 GPU): no comparable |

Consecuencia: para Heat, "comparable a la literatura" solo puede significar "comparable a la receta DeepXDE reproducida por nosotros"; no hay número externo. Es otra razón de la recomendación de D3.

### 3.4 DFG 2D-1 Re=20

Referencia oficial de QoIs: FEATFLOW (https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html): Cd=5.57953523384, Cl=0.010618948146, Δp=0.11752016697. Un FEM básico Taylor–Hood P2/P1 (jNO, ~10k DOFs) reporta errores de ~0.1 % / 1.9 % / 0.7 % en Cd/Cl/Δp, así que el criterio de convergencia de Codex es realista. **No encontré ningún PINN publicado en este benchmark exacto (estacionario, Re=20, QoIs FEATFLOW)**; los PINN de cilindro conocidos (p. ej. DeepXDE inverso, Re=100) no aplican. No existe comparación "vs literatura" para DFG, solo vs FEATFLOW (y vs FEM cuando llegue).

---

## 4. Tabla de decisión

| Decisión | Recomendación | Confianza | Qué falta |
|---|---|---|---|
| D1 Referencia Allen–Cahn (ETDRK4/FD8192) | Aceptar con redacción acotada (2 solvers independientes, no la tabla `.mat`) | Alta | Revisar que IC/BC no se compartan entre solvers; opcional BDF vs ETDRK4 en malla común |
| D2 Burgers "comparable a PINNacle" | Aceptar con redacción estrecha (reproducción del baseline bajo su evaluador; no SOTA) | Alta | Nada obligatorio; para SOTA, replicar vRBA (fuera de presupuesto) |
| D3 Heat: RBF vs DeepXDE | Diferir el claim; solo frase descriptiva interna | Media | Rerun B-spline a 20k pasos (Codex); DeepXDE con mismas semillas si se quiere parear |
| D4 Congelar baseline (`eval_integrity`) | Diferir; orden futuro: Burgers, AC tras D1, Heat | Media | Firma de EXP-0001; decidir qué baseline es de mecanismo y cuál de literatura |
| DFG | No decidible | — | FEM convergido de 3 niveles; evaluador de fuerzas aclarado 2026-09-30: σ·n vigente; el 74 % era un bug |

**Lo mínimo que tienes que firmar hoy:** D1 (referencia Allen–Cahn con redacción acotada) y D2 (frase estrecha de Burgers); lo demás puede esperar a Codex.

---

## 5. Debilidades y límites

- **Ninguna comparación con literatura es completa.** Solo Burgers/PINNacle es apples-to-apples; AC vs AEH-PINN tiene referencia no verificable y presupuesto distinto; Heat no tiene número publicado.
- **n=5 (y n=3 en PINNacle).** Las SD son orientativas. En Burgers ambas SD son minúsculas (3e-5): esperable con L-BFGS que converge al mismo mínimo; confirma reproducibilidad, no distingue métodos.
- **Cohortes no independientes.** AC: los rescoring Adam y L-BFGS son sobre los mismos checkpoints. DFG: las cohortes 301–305 y 501–505 no son comparables entre sí.
- **Heat depende del presupuesto** (~100x); el ledger de v4 sigue `incomplete_validation`.
- **Inconsistencia interna DFG** (Cd 30 % / 74 % / 5 % según documento y cohorte); resuelta el 2026-09-30, ver local-20260930/README.md.
- **Pendientes de Codex** (FEM DFG, B-spline rerun): no usados; si cambian, rehacer D3 y DFG.
- **Alcance de verificación:** leí dos fuentes externas hoy; el resto viene de los `literature-*.md`. No corrí código ni validadores. Todo "aceptar" aquí es una **recomendación**; la aceptación es tuya y se registra por ti en `registry/claims.jsonl`.
