# EXP-0028, predictions.md: predicciones para el bimorfo piezoeléctrico (H-PRED-1)

**Estado: FIRMADA por el investigador en el chat el 2026-10-25** («Firmo por mejora modesta» y «firmo» predictions.md). Decisión de la sección 2 sobre E y F: **mejora modesta** (la propuesta por defecto del borrador era «indiferente»). Preparada el 2026-10-24 (agente) con spec v1.1 a v1.6, `cruce/`, `results/` y `piezo/properties_piezo.md`; semillas definitivas 20260728 a 20260732. Desde esta firma no se reescribe (spec fase 3) y el criterio queda fijo antes de correr piezo. Con un objetivo por vez, un acierto es un indicio y no se generaliza.

## 1. Evidencia del banco (efecto = razón mediana e_X/e_B pareada, fin de Adam)

Bandas de dirección (fijadas aquí): mejora < 0.90; indiferente 0.90 a 1.10; empeora > 1.10, sobre la mediana. Se reporta además la etiqueta uniforme del spec v1.1.

**Receta corta (Adam 2 000 a 4 000, float32; n = 10).** Pesa menos.

| problema | F | J | E | I |
|---|---|---|---|---|
| AC 1e-3 / 2e-4 | 0.29 / 0.47 | 0.99 / 0.96 | 0.28 / 0.40 | 0.22 / 0.18 (dep. semilla) |
| Burgers | 0.63 (dep.) | 1.93 (dep.) | 1.14 (dep.) | 6.24 (daña) |
| KPP | 1.09 (dep.) | 2.72 (daña) | 0.76 (dep.) | 1.32 (dep.) |
| NS2d-CG | 0.68 | 1.00 | 0.64 | 0.56 |
| DFG | 0.80 | 1.02 | 0.83 | 0.76 |

**Receta completa (la de cada problema; n = 5 salvo indicación).** Pesa más. Solo fin de Adam, que es la métrica de piezo (Adam 20 000, float64, sin L-BFGS).

| problema | F | J | E | I | lectura |
|---|---|---|---|---|---|
| AC 1e-3 (Adam 40 000) | 0.281 (5/5) sirve | sin dato | 0.212 (5/5) sirve | sin dato | sostenido en Adam; tras L-BFGS E 0.39 y F 0.81 (dep. semilla) |
| KPP (receta AC) | 0.429 (4/5, dep.) | sin dato | 0.077 (5/5) sirve | sin dato | E aparece (en corto no servía) |
| NS2d-CG (Adam 20 000, **n = 3**) | 0.970 | 1.150 (0/3, daña) | 0.960 | 0.966 | todo ~0.98: el efecto de la receta corta desaparece; provisional |
| Burgers (**n = 2**) | 0.258 | sin dato | 0.435 | sin dato | sin veredicto; tras L-BFGS F daña (2.05) |
| DFG | sin confirmar | sin confirmar | sin confirmar | sin confirmar | solo receta corta |

Lectura honesta: a receta completa, F y E mejoran en tres de cuatro problemas (AC, KPP, Burgers con n = 2) y se diluyen en el único acoplado medido (NS, n = 3). **La hipótesis de trabajo (efecto de eficiencia con poco presupuesto, que se diluye si B tiene presupuesto suficiente) está respaldada por un solo problema y contradicha por AC, donde con 40 000 pasos E sigue en 0.21.** Por eso la dilución no se asume universal: se asume para piezo solo en la medida en que piezo se parezca a NS (sección 2).

**Atribución (H-ATTR):** D5 (E frente a F) a receta completa en Adam: AC 0.72, KPP 0.24, Burgers 0.62 (n = 2), NS 0.99. H-ATTR-A (la ganancia es de la base) respaldada en corto en NS-flat, DFG y Burgers; no respaldada en AC ni KPP a receta completa. El residuo (E sobre F) aporta donde hay efecto; en NS no hay nada que atribuir.

**Propiedades del problema real** (`piezo/properties_piezo.md`, 3 semillas; reemplazan al stand-in, cuyo borde/PDE 3e-7, concentración 0.59 y disparidad 1e9 eran artefactos de escalas y pesos):

| propiedad | converso | directo | banco |
|---|---|---|---|
| salidas / términos de pérdida | 8 / 16 | 8 / 17 (9 no nulos) | 1 a 3 / 1 a 13 |
| borde/PDE (pérdida; traza GN) | 8e-4 (2.8e-4) | 1e-3 (2.2e-3) | 0 en AC, Burgers, KPP; 87 a 593 en NS, DFG |
| concentración del residuo (top 10 %, r²) | 0.32 | 0.32 | 0.21 a 0.62 (NS 0.31, DFG 0.33) |
| término dominante de la pérdida inicial | equilibrio_y, 94 % | equilibrio_y, 100 % | |
| dispersión de trazas entre aristas (log10 p95/p5) | 3.7 | 3.9 | 1.7 a 3.5 (fuera) |
| razón de escalas de salida | 2e10 | 4e10 | 1 a 5 (fuera) |
| cond(B) mediana | 8.0e3 | 8.4e3 | 8.5e3 a 1.0e4 (no separa) |

Piezo se parece a NS/DFG en acoplamiento y en concentración, y a AC/Burgers/KPP en peso del borde (la partición que en el banco era colineal queda **dividida**: multi-salida sin borde dominante, algo que el banco no contiene). Las propiedades del banco no separan converso de directo.

## 2. Predicciones (cada objetivo por separado; métrica `l2re_u` al paso 20 000, mediana de 5 semillas pareadas con su B)

Regla de inferencia: (a) lo confirmado a receta completa pesa más que lo corto; (b) el único problema acoplado con receta completa, NS, da indiferente y es el más parecido en acoplamiento, concentración y receta (Adam 20 000, sin dependencia de L-BFGS); (c) AC y KPP, que sí mejoran, difieren de piezo en 1 salida y en no ser mixtos, pero comparten el borde sin peso; (d) con la pérdida inicial dominada por un solo término (94 a 100 %), el residuo que pondera a E casi no aporta información sobre los demás términos. Resultado: **decisión del investigador al firmar (2026-10-25): «mejora modesta» para F y E.** El borrador proponía «indiferente» por NS; el investigador eligió el otro lado con estos argumentos: (1) a receta completa y al final de Adam F y E mejoran en tres de cuatro problemas; (2) piezo usa bordes duros y su borde/PDE medido es 2e-4 a 1.5e-3, como los problemas escalares donde el mecanismo ayudó, y no como NS/DFG (87 y 593), donde no ayuda con receta completa; (3) «indiferente» casi no se puede falsar con el criterio de 0.3 décadas, mientras «mejora modesta» exige acertar el rango. NS queda como la evidencia en contra.

| brazo | dirección | log10(e_X/e_B) previsto [intervalo 80 %] | problemas parecidos y propiedad | P(mejora / indif. / empeora) |
|---|---|---|---|---|
| F | **mejora modesta** | -0.22 [-0.50, 0.00] (razón ≈ 0.6) | AC (0.28 en Adam), KPP, Burgers (n = 2) a receta completa; borde/PDE 2e-4 a 1.5e-3 como los escalares con borde duro; en contra: NS completa (0.97), 8 salidas y escalas 2e10 a 4e10 sin ancla | 0.55 / 0.35 / 0.10 |
| E | **mejora modesta** | -0.30 [-0.65, -0.05] (razón ≈ 0.5) | AC (0.21 en Adam), KPP (0.077), Burgers (0.44, n = 2); E mejor que F en D5 de AC; en contra: NS completa (0.96) | 0.60 / 0.30 / 0.10 |
| J | empeora leve | +0.08 [-0.05, +0.45] | J nunca sirve en Adam (0.96 a 1.00 en AC, NS, DFG; 1.93 Burgers; 2.72 KPP; 1.15 NS completa). La dispersión de trazas fuera de rango no ancla un efecto | 0.15 / 0.40 / 0.45 |
| I | **sin predicción** | sin predicción | el banco no separa acoplamiento de salidas y ausencia de borde (colineales); piezo es el primer punto multi-salida sin borde dominante. I en NS completa: 0.97; en Burgers corto: 6.2 | no puntúa |

Idénticas para **converso** y **directo** (las propiedades del banco no los separan). Diferencias conocidas que no se usan para separarlas: el converso enruta el gradiente y su Jacobiano GN lleva el mismo `detach` (B no es el GN exacto del escalar); el directo usa un sustituto en la carga neta cero y λ_BC = 10, y 8 de sus términos son 0 exacto.

**Brazo mejor previsto: E. Brazo peor previsto: J** (entre F, E, J). Con F y E previstos cercanos (0.6 y 0.5) el orden F/E es poco más que una apuesta; se declara porque v1.3 lo exige.3 lo exige. **Componente que se espera que sirva: la corrección de la base (F capta la mayor parte, como en Burgers, NS y DFG en D5) con aporte posible del residuo (E algo mejor que F)** (ni base, ni escala, ni residuo); si algo sirve, la base (F) antes que el residuo. No se espera que sirva la escala sola (J).

**Otras palancas (S, R, C, W, H):** no se corren en piezo; sin predicción, no aplican.

## 3. Criterio de H-PRED-1 (v1.3, textual) y tasa base ingenua

H-PRED-1 se respalda **solo si se cumplen las dos**: (i) el brazo mejor y el peor previstos coinciden con los observados (entre F, E, I, J); (ii) la magnitud de log10(e_X/e_B) prevista para F y E queda a menos de 0.3 décadas de la observada (mediana de 5 semillas). Además se declara el acierto **solo si supera** la tasa base ingenua en direcciones. Los brazos «sin predicción» (I) quedan fuera de (i) y (ii) y se corren como exploratorios. Se aplica **por separado a converso y directo** y no se promedian.

Operacionalización fijada aquí: «supera» = más direcciones acertadas que el predictor ingenuo («F, E, I mejoran; J indiferente») sobre los brazos con predicción (F, E, J), por objetivo; el empate no supera. Mis direcciones coinciden con las ingenuas en F y E y difieren solo en J (empeora leve frente a indiferente): si F y E mejoran y J > 1.10 supero al ingenuo; si F y E mejoran y J queda en 0.90 a 1.10 empato o pierdo; el resultado decide en J.

| fuente | F mejora | E mejora | I mejora | J indif. | aciertos ingenuo |
|---|---|---|---|---|---|
| corto, 6 problemas (bandas de mediana) | 5/6 | 5/6 | 4/6 | 4/6 | 18/24 = 75 %; ≥ 3/4 en 4 de 6 problemas |
| completa, fin de Adam (AC, KPP, Burgers n = 2, NS n = 3) | 3/4 | 3/4 | 0/1 (NS) | 0/1 (NS) | NS: 0/4; resto: F y E |
| acierto del ingenuo en NS completa | 0 | 0 | 0 | 0 | 0/4 |
| predicción de este documento (probabilidad de acertar la dirección, subjetiva) | 0.40 | 0.40 | sin pred. | 0.45 (empeora) | |

## 4. Fuera de rango y riesgos

- **Extrapolación (J, E, I, y F en menor grado):** razón de escalas de salida 2e10 a 4e10 frente a 1 a 5 (la red trabaja adimensional y los bordes duros reescalan, pero no hay dato que la ancle); dispersión de trazas entre aristas 3.7 a 3.9 décadas frente a ≤ 3.46 (max/min 4.6 a 5.1 frente a 2.7 a 4.0; en el directo la razón entre capas 1.39 queda bajo el banco, 1.9 a 3.0); 8 salidas frente a ≤ 3; término dominante único; float64 y 20 000 pasos frente a float32 y 2 000 a 4 000.
- **Riesgo de I:** chunk 2 solo probado en CPU (segfault con chunk ≥ 4 en CPU, float64, 8 salidas); en GPU puede fallar o costar ~8× más llamadas por recálculo y rozar el tope de 3.4 h. Un `failed` de I no se imputa ni cuenta en H-PRED-1.
- **Línea base congelada no comparable:** RBF16 0.092 ± 0.021 (converso) se obtuvo con el gancho de `daniel_arm.py`, que reinicia Adam (momentos y remuestreo) cada 2 000 épocas; el puerto corre un Adam continuo. El único control válido es el B pareado de este bloque; 0.092 es solo informativo. El directo no tiene línea base propia.
- Otros: Jacobiano con enrutamiento (converso) y sustituto de carga neta (directo); n = 5 semillas; el B no reproduce por semilla entre sesiones (fase 0); evidencia de receta completa pobre en NS (n = 3), Burgers (n = 2) y nula en DFG, J e I.

**Actualizar al llegar (celdas vacías; completar y revisar la sección 2 antes de firmar):**

| fuente | problema | F | J | E | I | nota |
|---|---|---|---|---|---|---|
| `phase1e_ns1`+`ns2` (n=5, llegó 2026-10-25) | NS2d-CG, receta completa | fin de Adam 0.970 (5/5) indiferente; L-BFGS 0.980 | fin de Adam 1.150 (0/5) daña; L-BFGS 1.056 indiferente | fin de Adam 0.956 (5/5) indiferente; L-BFGS 0.971 | fin de Adam 0.945 (4/5) indiferente; L-BFGS 0.952 | todo en 0.9-1.1 en Adam: según la regla fijada abajo, no cambia nada para NS |
| `phase1e_acf` (n=5, llegó 2026-10-25) | AC d=1e-3 completa (J, I); KPP completa (J, I); Burgers no corrido (aplazado) | - | AC: dependiente de semilla (1.14); KPP: sirve (0.20 Adam, 0.235 L-BFGS) | - | AC: sirve al fin de Adam pero daña tras L-BFGS (3.5, no aplica a piezo); KPP: dependiente de semilla (0.16, máx 22) | con B mal convergido (KPP, B 8.6e-2) J, E (1d: 0.077 Adam) e I ayudan; en NS, con B estancado en 6.6e-2, nada ayuda: el nivel de error de B no basta como explicación |

Reglas de actualización fijadas ahora: si NS completa (n ≥ 5) mantiene todo en 0.9 a 1.1 en Adam, no cambia nada; si F o E salen < 0.90 con ≥ 4/5, F y/o E pasan a «mejora» con magnitud tomada de NS; si J e I a receta completa en AC/KPP/Burgers coinciden en signo, I pasa de «sin predicción» a la dirección común solo si piezo no se separa por el factor acoplamiento; en cualquier otro caso I sigue sin predicción.

## 5. Qué se corre y cómo se leerá (registro previo)

Se corre: B, F, J, E, I en converso y directo, semillas 20260728 a 20260732 (la del paper es 20260728) pareadas por problema, más un brazo de referencia MLP-paper (semilla 20260728) fuera de H-PRED-1, receta del paper (Adam 20 000, float64, sin L-BFGS, remuestreo cada 200), recálculo cada 1 000 pasos, registro de error cada 500 y artefactos completos (v1.6); métrica primaria `l2re_u` (v y φ también). No se repite la línea base; la validación del MLP del paper (0.107/0.150/0.045 y 0.117/0.153/0.067) es previa, aparte.

Lectura, fijada antes de ver datos (por objetivo y brazo; razón mediana, k/5 mejoras, razón máxima, etiqueta uniforme):
1. **F y E dentro de 0.90 a 1.10:** respalda la dilución con presupuesto en un problema acoplado; con H-PRED-1 (sección 3) decide el acierto.
2. **F o E < 0.90 con ≥ 4/5:** la hipótesis de dilución **no** se extiende a piezo. Se reporta la propiedad que engañó (candidatas a priori: borde sin peso, escalas, término dominante).
3. **F o E > 1.10 con ≥ 4/5:** el mecanismo daña en acoplamiento con escalas dispares; se atribuye por D5 y J frente a B.
4. **D5:** E/F < 0.90 con ≥ 4/5 = aporte del residuo; E/F entre 0.90 y 1.10 con F efectivo = base; ambos ≈ 1 = nada que atribuir.
5. **J > 1.10:** coherente; **J < 0.90 con ≥ 4/5:** contradice y se señala la dispersión de trazas como propiedad que engañó.
6. **I:** solo descriptivo; su signo se anota como el séptimo punto de la tabla, sin inferir causa por un problema.
7. **Dilución observable:** razón mediana en el paso 2 000 y en 20 000 (solo si e_B < 0.9 en ese paso, para no leer la meseta cercana a la solución nula). Razón < 0.90 temprana y 0.90 a 1.10 final = eficiencia que se diluye; < 0.90 en ambos = ganancia que persiste.
8. **Contradicen «mejora modesta»:** F o E entre 0.90 y 1.10 con ≥ 4/5 (indiferente, como NS) o > 1.10 con ≥ 4/5 (empeora) al paso 20 000 en cualquiera de los dos objetivos; **mejora mayor a la prevista:** F o E < 0.30 (fuera del intervalo).
9. Converso y directo en direcciones distintas con propiedades iguales: se reporta como falta de poder de las propiedades del banco, sin promediar.

Un fallo de H-PRED-1 se reporta con la tabla de errores de predicción por brazo y objetivo, y este archivo no se reescribe.

## Firma del investigador

Nombre: Rodrigo Mansilla (firma dada en el chat; registrada por el agente)  Fecha: 25 / 10 / 2026

Firma: «Firmo por mejora modesta» y «firmo» (chat, 2026-10-25)

(Al firmar se confirma además la métrica primaria `l2re_u`, DIFERENCIAS.md fila 14, y la decisión indicada en la sección 2 sobre E.)
