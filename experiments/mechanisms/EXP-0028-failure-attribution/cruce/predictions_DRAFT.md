# BORRADOR NO FIRMADO: predicciones para el bimorfo piezoelectrico (EXP-0028, H-PRED-1)

**Estado: DRAFT. No es `predictions.md`. Lo firma el investigador (spec fase 3); hasta entonces no tiene valor de preregistro.** Escrito el 2026-10-24 por un agente a partir de `cruce.md`. No se ha corrido nada de piezo con el mecanismo. Los numeros salen de `cruce/` y `fase2/properties_terms.md`.

## 0. Potencia, dicho antes que nada

- Evidencia: 6 problemas (≈5 independientes: AC 1e-3 y 2e-4 comparten red y pool), receta corta, 4-10 semillas pareadas. Objetivo: **un** problema. Una coincidencia de direccion es un indicio, no una prueba (spec fase 3).
- Una prediccion ingenua sin contenido de propiedades («F, E, I mejoran; J indiferente») acierta ya 3-4 de 4 con la tasa base del banco (F sirve en 5/6, E en 4/6, I en 2/6, J nunca sirve en Adam). **Cumplir «≥ 3 de 4 brazos» no distingue el conocimiento de las propiedades de ese predictor base**; si el investigador quiere que H-PRED-1 pese, conviene fijar tambien la magnitud (abajo) y que la prediccion del mejor brazo no sea la tasa base.
- Piezo corre **solo Adam (20 000 pasos, float64), sin L-BFGS**. Por eso uso como referencia el **efecto al final de Adam** del banco (cuyo error es el que importa) y trato como no aplicable lo que en el banco aparece solo en L-BFGS (daño de J en Burgers y DFG, R en AC).
- Receta corta (Adam 2 000-4 000 float32) frente a Adam 20 000 float64: el efecto puede no sostenerse (v1.2).

## 1. Propiedades de piezo medibles sin entrenar

**Lo que existe**: `src/mechanisms/residual_terms.py` solo trae adaptadores para NS2d-CG, DFG (`DFGTerms`) y Burgers/AC; `piezo_residual_terms` es un contrato que lanza `NotImplementedError`. **No existe `PiezoProblem`** ni el codigo de Daniel en el arbol (`pinn_piezo`, solo se importa desde un clon externo en `daniel_arm.py`).

**Que se midio igualmente**: un **stand-in**, la reimplementacion propia `piezo_pinn.py` (EXP-0009; «diagnostico negativo»), con un adaptador de terminos escrito en `fase2/measure_properties_terms.py` (reproduce su `loss_fn`, error relativo 1e-16). RBF16 de 11 bases, 8 salidas (u, v, phi, sx, sz, tau, Dx, Dy), bordes duros en u, v, phi, escalas SC_*, 16 terminos de peso 1, 2048 interiores + 256 por borde, float64, semillas 4001-4010. **Lo que el stand-in NO es**: el trabajo de Daniel tiene 4 entradas (xi, eta, |eta-0.5|, fase), formulacion por capas con interfaz, escalas por campo propias, enrutado stop-gradient de la constitutiva mecanica, remuestreo cada 200 epocas y **pesos adaptativos**; el stand-in tiene 2 entradas, pesos 1 y escalas distintas. Las propiedades que dependen de escalas y pesos (disparidad de terminos, borde/PDE, traza GN por termino) **cambian con la formulacion** y deben remedirse con el codigo real.

| propiedad (piezo stand-in; mediana de 10 semillas) | piezo | rango del banco (6 problemas) |
|---|---|---|
| acoplamiento: salidas / terminos de perdida | 8 / 16 | 1-3 / 1-13 |
| concentracion r^2, top 10 % | 0.588 [0.51, 0.71] | 0.21-0.62 (KPP 0.62, NS 0.31) |
| borde/PDE en la perdida inicial | 2.8e-7 | 0 (duro) ; 87 NS ; 593 DFG |
| traza GN borde/PDE | 1.3e-7 | 0 ; 7.7 NS ; 37.7 DFG |
| max/min de la perdida por termino | 1.0e9 (sz = 92 % de la perdida) | 1.5e3 NS ; 1.7e7 DFG |
| razon de escalas de salida (u 3e-7 m, phi 100 V, tensiones 6e3 Pa...) | 2e10 (phi/u: 3e8) | 1-5 |
| cond(B) mediana / minima | 8.7e3 / 2.0e3 | 8.5e3-1.0e4 / 2.0e3-4.1e3 |
| log10 p95/p5 de traza(B) entre aristas (3 semillas) | **6.6** | 1.7-3.5 |
| log10 max/min de traza entre capas (3 semillas) | **4.2** | 1.9-3.0 |
| Peclet / Damkohler | no aplica (sin adveccion ni reaccion) | |
| gn_trace_bc/pde y borde duro | bordes duros en u, v, phi, pero las tracciones libres y Dx = 0 son terminos | |

Lectura: (i) piezo es el problema de mayor **acoplamiento** (8 salidas) y el de mayor **disparidad de escalas** del conjunto, y esta disparidad esta **fuera del rango observado** (4-8 ordenes de magnitud mas que el banco): lo que el banco dice de J, E, I en este eje es extrapolacion pura. (ii) Respecto al **borde**, el stand-in se parece a AC/Burgers/KPP (el borde no pesa en la perdida ni en el GN, 1e-7) y no a NS/DFG; si en el codigo de Daniel los pesos adaptativos suben el borde, esa lectura cambia. (iii) La concentracion del residuo (0.59) es la mas parecida a KPP (0.62), el problema donde F y E no sirven. (iv) cond(B), el unico numero que el spec pedia como covariable de F, es igual que en el resto (no separa).
Faltan para medirlo con el problema real: un `PiezoProblem` con `residual_terms` (terminos PDE, constitutivos, interfaz y borde con **los pesos de la formulacion del paper**), `model_fn` que envuelva el modelo completo (red + escalas + levantamiento de voltaje), una decision sobre `mechanical_constitutive_stopgrad` en el GN, y los pools del paper; ademas, de `piezo.tex`, resolver pesos adaptativos frente a fijos, los signos de e21/e22 y la licencia (spec fase 4).

## 2. Predicciones por brazo (efecto = razon mediana e_X/e_B al final de Adam, pareada con B)

Escala usada: «mejora» = razon < 0.9; «indiferente» = 0.9-1.1; «empeora» = > 1.1. Los problemas parecidos se eligen **por propiedad**, no por tema fisico.

| brazo | direccion | magnitud aproximada (razon, intervalo plausible) | problemas del banco que se le parecen y por que | confianza |
|---|---|---|---|---|
| **F** (Gram fijo) | **mejora** | 0.7 (0.4-1.1) | NS y DFG (Adam 0.68 y 0.80; 10/10): multi-salida, mixto, sin Peclet. Contra: KPP (1.09), el de residuo mas concentrado (0.62; piezo 0.59), donde F no sirve. F corrige la base RBF, que es la misma (11 bases, ancho 0.35) y no depende del problema; el solape de la base de piezo es el de todos. | baja a media (≈ 0.6 de que la direccion sea «mejora») |
| **E** (bloque por arista con residuo) | **mejora modesta** | 0.8 (0.4-1.2) | NS y DFG (0.64, 0.83; 10/10) por acoplamiento; AC (0.28-0.40) si el borde no pesa (bc/PDE ~ 0 como piezo). Contra: Burgers y KPP (1.14 y 0.76, dependiente de semilla), por concentracion (KPP) y por ser problemas de transporte. En el banco el beneficio de E decrece donde hay terminos de borde (rho +0.85, 6 puntos): el stand-in tiene borde ≈ 0, que apunta al lado AC, pero 7 ordenes de magnitud de disparidad entre terminos hacen que el residuo (que pondera B) lo domine un solo termino (sz): E ≈ un bloque del termino dominante. | baja (≈ 0.55) |
| **J** (diagonal del bloque de E) | **indiferente** | 1.0 (0.7-1.5) | AC, NS, DFG en Adam (0.96-1.02, 5-6/10 mejoras); daña en KPP (2.7) y es dependiente en Burgers (1.9). El daño de J en Burgers y DFG aparece solo en L-BFGS, que en piezo no existe. Nota: toda la familia normaliza por arista (m = traza/n), de modo que la disparidad de 6.6 decadas entre aristas queda igualada tambien por J: es la propiedad de piezo mas alejada del banco y puede dar un efecto grande en cualquier direccion; no hay dato que la ancle. | baja (indiferente ≈ 0.4, mejora ≈ 0.3, empeora ≈ 0.3) |
| **I** (bloque por entrada, acopla aristas) | **mejora** si el acoplamiento de salidas es la causa; **empeora** si lo es la ausencia de terminos de borde | si mejora: 0.65 (0.4-0.9); si empeora: > 1.3 (hasta 6 como Burgers) | NS y DFG (0.56, 0.76; 10/10) frente a todos los escalares (0.18-6.2, dependiente o catastrofico). **El banco no puede decir cual**: `n_outputs`, `n_loss_terms`, `bc_over_pde` e `resid_rms0` dan la misma particion {NS, DFG} frente a {escalares}. Piezo cae del lado NS/DFG por acoplamiento (8 salidas) y del lado escalar por borde (≈ 0). Es el brazo con el efecto mas grande en el banco y con la prediccion mas incierta. El bloque por entrada de 8 salidas es 88 x 88 por entrada: tambien es el mas caro. | muy baja (≈ 0.45 de «mejora») |

**Mejor brazo previsto:** I si el acoplamiento es la causa, si no F. Declaro **I (confianza muy baja), segundo F**. Si el investigador prefiere una apuesta que no dependa de un factor sin desempatar, F.
**Componente del mecanismo que se espera que sirva:** la **base** (F ya captura casi todo lo que captura E en NS/DFG: D5, 0.94 y 1.19-1.32 en final; H-ATTR-A respaldada en NS-flat y DFG, no en AC/KPP). No se espera que sirva **solo la escala (J)**, y la dependencia del residuo (E frente a F) solo aporta algo en AC/Burgers/KPP, no en NS/DFG. El **acoplamiento entre aristas (I)** es el candidato de mayor efecto y de mayor incertidumbre.

## 3. Otras palancas (no son brazos del spec de piezo; para el inventario)

| palanca | prediccion en piezo | base |
|---|---|---|
| S (0014, muestreo auto-compuerta) | empeora o indiferente (≥ 1.0) | daña en AC 1e-3, Burgers y KPP (1.4-12.9), indiferente en AC 2e-4; sin datos NS/DFG. Confianza baja. |
| R (0025, RAD) | indiferente o empeora | en el banco solo sirve en AC y solo en L-BFGS; daña en Burgers y KPP; el residuo de piezo es concentrado (como KPP), y la receta del paper ya remuestrea cada 200 epocas. Sin datos NS/DFG. Baja. |
| C (regrid, 0024-A) | indiferente o empeora | sirve en AC (0.67-0.79), daña en Burgers y KPP; sin datos NS/DFG; piezo tiene 4 entradas en la formulacion de Daniel. Muy baja. |
| W (foco por responsabilidad, oraculo), H (traspaso) | no candidatas / no aplican | W indiferente en todo; H requiere L-BFGS (piezo no lo tiene). |
| 6 (forma del residuo), 7 (base/activacion) | sin dato | solo inventario. |

## 4. Evidencia de receta completa que falta (fase 1d) y su efecto sobre estas predicciones

- **Corriendo**: AC 1e-3 (B + E, F, C), Burgers (B + F, E), KPP (B + E, F), 5 semillas (4101-4105), receta de la linea base. Hasta que terminen, **ninguna** de las cifras anteriores esta confirmada a receta completa en ningun problema.
- **No hay (ni esta presupuestado en 1d)**: NS a receta completa (`run_phase1c.py --recipe full`), DFG a receta completa, **J e I en cualquier problema a receta completa**. Es decir: de los dos problemas mas parecidos a piezo por estructura (NS y DFG) y de los dos brazos mas inciertos (J, I) no habra confirmacion. Por la regla v1.2, para los brazos F, E, J, I la prediccion que se firme debe declarar que se apoya en receta corta; la regla «si un efecto se pierde a receta completa en los problemas parecidos, baja a indiferente» solo se puede aplicar a E y F y solo en AC/Burgers/KPP (que son los menos parecidos).
- Sugerencia para la firma: (a) esperar 1d y bajar F y E a «indiferente» si en AC (el problema donde mas mejoran) el efecto de receta completa es > 0.9; (b) pedir NS a receta completa (el banco mas parecido a piezo en acoplamiento) para al menos B, F, E, I; (c) remedir las propiedades de piezo con el codigo real antes de firmar, sobre todo borde/PDE, disparidad de terminos y escala entre aristas, que son las que dependen de pesos y escalas.

## 5. Lo que este borrador no dice

- No afirma que el mecanismo funcione en acoplamientos ni que alguna propiedad **cause** el efecto: son 6 puntos descriptivos con factores colineales.
- No usa p-values. Las confianzas son subjetivas, calibradas contra la tasa base del banco, no estimaciones estadisticas.
- Todas las propiedades de piezo son del stand-in, no del problema del paper.
