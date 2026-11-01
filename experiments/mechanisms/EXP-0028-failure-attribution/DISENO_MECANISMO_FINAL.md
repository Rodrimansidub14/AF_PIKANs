> **ACTUALIZACIÓN (2026-10-25, tarde): este borrador se superó.** El análisis de propiedades de piezo (`ANALISIS_PIEZO_PROPIEDADES.md`) retira la hipótesis de amplificación uniforme de F y la variante «Fn» (F con P reescalada): F falla por desajuste de dirección (empeora el condicionamiento en 82-88 % de las aristas) y no por escala. La regla de activación por dispersión a la inicialización también se matiza: en piezo la dispersión crece durante el entrenamiento (3.3 → 3.95 y 3.75 → 4.40). La propuesta vigente está en el análisis de piezo: E como base, J como control, F y Fn fuera. El resto de este documento queda como registro del razonamiento previo.

# Diseño del mecanismo final (borrador, sin cómputo nuevo) — 2026-10-25

Estado: **propuesta para revisión del investigador.** Solo usa resultados ya registrados. No hay nada firmado ni lanzado.

## 1. Lo que la evidencia sostiene (y lo que no)
| Pieza | Evidencia | Estado |
|---|---|---|
| **B** (control, sin precondicionador) | referencia de todos los contrastes | es el control, no un mecanismo |
| **E** (bloque por arista, P desde el residuo, recalculada cada 1 000 pasos) | Allen–Cahn 0.21 (Adam completo), KPP 0.08, piezo converso 0.18 y directo 0.82 (mediana de 3), NS completo 0.96 (neutro), Burgers 0.44 (n = 2, con cola en receta corta) | **sirve o es neutro en todos los problemas medidos a receta completa**; cola de semillas malas en Burgers sin explicar |
| **F** (Gram fijo, sin residuo) | robusto en el banco corto (4 de 5); con receta completa se diluye (AC 0.81, NS 0.97); en piezo **empeora** (1.34 converso, 3.93 directo) | **depende del problema**; hipótesis de falla en §2 |
| **J** (diagonal de E) | casi siempre daña o es indiferente (Burgers, DFG, KPP corto); mejora en el converso de piezo (0.67) y KPP completo (0.2) | sin patrón estable; no es candidata |
| **Fn** (F con P reescalada a máx. valor singular 1) | **ninguna: es una predicción del diagnóstico, no un resultado** | por probar |
| S, R, C, W (muestreo, centros, foco) | ayudan en Allen–Cahn corto, dañan en Burgers y KPP | no entran al mecanismo final |

## 2. Una señal candidata ya medida y sin costo: dispersión de curvatura entre aristas
`edge_trace_log10_p95_p5` (log10 de p95/p5 de la traza del Gauss-Newton por arista, a la inicialización; `fase2/` y `piezo/properties_piezo.csv`), contra el efecto de F al final de Adam a receta completa donde existe:

| Problema | dispersión | efecto de F (razón e_F/e_B) |
|---|---|---|
| Burgers | 1.68 | sirve (corto 0.29 final; n = 2 completo, 0.26 en Adam) |
| NS2d-CG | 2.23 | 0.97 (indiferente, completo) |
| DFG | 2.54 | sirve (0.57, corto) |
| Allen–Cahn | 3.20 | sirve (0.28 en Adam, completo) |
| KPP | 3.46 | dependiente de semilla (0.64 tras L-BFGS, completo) |
| Piezo converso | 3.59 | **1.34** (empeora; 0.64, 4.80, 1.34) |
| Piezo directo | 3.82 | **3.93** (empeora en 3 de 3) |

Lectura: F es segura mientras la dispersión es baja (≲ 3.3) y falla por encima. **Son 7 puntos de 6 familias de problemas, con recetas mezcladas (corta y completa); es una hipótesis coherente con el diagnóstico (el paso efectivo de F se amplifica hasta ~31× de forma uniforme, ciego a la curvatura de cada arista), no un resultado.** La señal se mide a la inicialización con un recálculo de bloques (≈ 10-15 s en H100), antes de entrenar.

## 3. Diseño propuesto (a debatir)
Mecanismo en dos niveles, sin selección por tablas externas:
1. **Medir** a la inicialización la dispersión d = log10(p95/p5) de las trazas por arista (un recálculo de bloques; es el costo de un paso de E).
2. **Activar:** por defecto **E** (es la única pieza que sirvió o fue neutra en todos los problemas completos). **F/Fn** solo como variante barata si d ≤ d₀ (sin recálculos; ahorra ≈ 3-5 % de tiempo en piezo), con d₀ ≈ 3.3 a calibrar.
3. **Guarda de estabilidad** (nueva, sin probar): vigilar la razón entre la pérdida actual y su mínimo previo (los «eventos > 5×» del diagnóstico de F) y, si E empeora a B de forma sostenida, volver a P = I desde el último punto estable. Atiende la cola de semillas malas de Burgers (EXP-0026/0027: dependencia de semilla sin predictor).
Lo que **no** se afirma: que la regla d₀ generalice (7 puntos, uno solo con F fallando), que Fn funcione, ni que la guarda sea necesaria.

## 4. Qué falta para decidir (en orden de costo)
| Pregunta | Cómo se responde | Costo |
|---|---|---|
| ¿d separa donde F sirve de donde F falla? | analizar dispersión por semilla contra el efecto de F por semilla con los artefactos existentes (sin cómputo); sumar la dispersión de Burgers y KPP por semilla de la fase 2 | 0 |
| ¿Fn se estabiliza en piezo? | 6 corridas (3 semillas × 2 problemas), pareadas con B y E ya existentes | ≈ 6 USD (Modal H100, 4 procesos) |
| ¿La guarda evita las semillas malas de E? | reanalizar EXP-0026/0027 (Burgers) con los «eventos > 5×» sobre `hist`; luego 5 semillas malas conocidas con y sin guarda | análisis 0; prueba ≈ 1-2 USD en T4 |
| ¿E mejora con un bloque más frecuente o P normalizada en piezo? | no hay evidencia que lo justifique hoy | no se corre |

## 5. Entregable siguiente
`EXP-0029-mecanismo-final` (spec con hipótesis, criterios y brazos: B, E, Fn, y E con guarda), con su propia firma. Predicciones por escrito antes de correr.
