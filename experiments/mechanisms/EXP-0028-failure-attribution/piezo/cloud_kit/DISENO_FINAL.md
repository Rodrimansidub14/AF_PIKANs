# Mecanismo final y su hipótesis (propuesta para firmar, 2026-10-26)

Estado: **propuesta; nada firmado ni lanzado en Modal.** Sustituye a `DISENO_ARANA.md` en lo que dice sobre pesos; no lo borra: cada pata sigue en el diseño con un **peso por evidencia**, que sube o baja cuando entra evidencia nueva. No hay patas «descartadas»; hay patas con peso 0 hoy y una razón registrada. Lo medido: `results/T1_T2_resultados.md`, `analisis_piezo/check_P2.md`, `check_P2_mezcla.md`, `check_Vf.md`, `ANALISIS_PIEZO_PROPIEDADES.md`.

## 1. Qué sabemos (resumen que sostiene el diseño)
1. **El cuerpo E hace el trabajo.** Converso: error l2re_u 0.198 → 0.053 (media de 3 semillas), CV 0.81 → 0.42; directo neutro (0.208 → 0.193). Los bloques GN de piezo son casi de rango uno (cond ≈ 1e4); E los lleva a cond 1.
2. **La dirección del bloque casi no depende de cómo se agrupen los términos.** Con la mezcla de P2 al b = 0.3 solo cambia la dirección dominante en 2 % de las aristas del converso y 0 % del directo (a b = 1: 38 % y 9 %; `check_P2_mezcla.md`). Lo que sí crece es la cola de la condición efectiva (p90: 23 en el converso a b = 0.3; 7.6 a b = 0.1; 1.4 en el directo a b = 0.1). Por eso en T1/T2 S y E difieren poco: u ×0.95 (mediana), φ ×1.06-1.33 en el converso con n = 3, dentro de lo que una perturbación pequeña produce en un entrenamiento caótico.
3. **Un cambio brusco de P daña.** P4 con amortiguación (a_body hasta 0.25, 3 veces por corrida) subió la pérdida del directo 2-3× frente a la misma configuración sin amortiguar (T2). Es la pérdida del condicionamiento de E, no del reinicio de Adam: todos los brazos reinician Adam en cada recálculo.
4. **El voltaje flotante ya está en su mínimo exacto** (`check_Vf.md`): P1 no tiene nada que corregir en piezo.
5. **F (Gram fijo) empeora el condicionamiento** en 82-88 % de las aristas (desajuste de dirección); no es candidato.

## 2. Diseño: E graduado («M»)
Un cuerpo y patas con **activación a_i = w_i · s_i**, donde w_i es el tope por evidencia (tabla) y s_i ∈ [0, 1] la señal medida en cada recálculo (continua, nunca un interruptor). Se instala P_ef = (1 − a)·I + a·P como en el diseño anterior.

| Pata | Señal s_i (medida en el recálculo) | Tope w_i hoy | Por qué ese peso (evidencia) | Qué hace en piezo |
|---|---|---|---|---|
| **Cuerpo E** | ninguna | 1 | única pieza con ganancia medida (§1.1) | P completo por arista |
| **P1 escalares** | \|V̂_f − V*\|/\|V*\| (dos evaluaciones de pérdida) | 1 | la señal es la que decide; medida ≈ 0.001 | s ≈ 0 → no actúa; en otro problema con escalar mal optimizado actuaría sola |
| **P2 balance por grupo** | mayor b en {0.05 … 1} con cond efectiva p90 ≤ 10 (medida con el bloque ya calculado, sin otro Jacobiano) | 0.5 | cambio de dirección pequeño y cola de cond que crece con b (§1.2); T1/T2: u ×0.95, φ sin mejora | b ≈ 0.12 en el converso, ≈ 0.5 en el directo |
| **P3 Gram empírico** | presupuesto: tiempo de recálculo > 10 % del paso | 1 si hay presupuesto, 0 si no | variante barata de E; no mejora a E; útil si E es caro | 0 en piezo (E cuesta ≈ 4-5 %) |
| **P4 guarda** | tasa de saltos (> 5× el mínimo de 200 pasos) | respuesta 0.1 (suelo de a_body 0.9); registro siempre | la respuesta fuerte daña (§1.3); el registro da la señal para calibrar | solo registra: respuesta 0.1 todavía sin probar, entra como sombra |
| **P5 arranque denso** | pérdida en llano (variación relativa en 100 pasos < 1e-3) | 1 | costo +2 recálculos (≈ 0.4 %); efecto sobre la salida del llano sin medir | recálculos extra en 250 y 500 |
| **P6 diagonal (J)** | mismo presupuesto que P3 | 0 si hay presupuesto | J: converso 0.089, directo 0.181 (mejor del directo); inestable en el banco | 0 en piezo |
| **X1-X3** | discrepancia constitutiva, desbalance, saltos | fuera de la receta del paper | cambian la pérdida o el optimizador | pista B, no se corre |

**Dónde se ejecuta cada pata:** ver `DISENO_EJECUCION.html` (el recálculo y las señales viven en R3-R6; P4 mira la pérdida de cada paso; P5 cambia el calendario). **Lo que M añade a E es pequeño por construcción** (P2 a b ≤ 0.5, P5 +2 recálculos, P4 en sombra), y eso es parte de la hipótesis.

## 3. Hipótesis final (borrador; cifras mías para que las cambies)
**H-FINAL.** *El beneficio del precondicionamiento por arista en piezo lo lleva el cuerpo E; las patas, con los pesos de §2, son de segundo orden: M no se aleja de E más de lo que la variación entre semillas permite y mantiene la ventaja sobre B del converso.*

Predicciones (semillas 20260728-30, receta del paper, l2re_u al paso 20 000, mediana de las 3 semillas):
1. **Converso:** l2re_u(M)/l2re_u(E) ∈ [0.85, 1.15]; l2re_u(M)/l2re_u(B) ≤ 0.5 (E da 0.27 con las 3 semillas; M debe conservarlo).
2. **Directo:** l2re_u(M)/l2re_u(E) ∈ [0.85, 1.15]; ninguna semilla de M > 1.5× E.
3. **φ del converso:** l2re_φ(M)/l2re_φ(E) ≤ 1.20 (la medición de T2 dio 1.06-1.33; no se predice mejora).
4. **Estabilidad:** saltos de pérdida de M ≤ los de E + 3 en el directo.
5. **Costo:** el tiempo por paso de M ≤ 1.10× el de E.

**Se falsa si:** la razón u(M)/u(E) sale fuera del intervalo en 2 de 3 semillas de un problema, o ninguna de las predicciones 1-2 se cumple. Nota para la tesis: esta hipótesis puede salir *respaldada con un resultado «M ≈ E»*; eso es un resultado sobre el diseño (las patas no mueven el error), no un fracaso.

## 4. Qué se corre en Modal (si firmas) y costo
| Tramo | Corridas | Costo (≈ 1.1 USD/corrida, cota alta) |
|---|---|---|
| **M**, semillas 28-30, converso y directo (pareadas con B y E ya medidas) | 6 | ≈ 7 USD |
| B y E, semillas 31-32 (completa la cohorte del paper, n = 5) | 8 | ≈ 9 USD |
| M, semillas 31-32 | 4 | ≈ 4.5 USD |

Con ≈ 19-20 USD caben los dos primeros tramos (≈ 16 USD). El tercero solo si la tarifa real sale por debajo de la supuesta (verificar en la cuenta antes). Orden: M primero; B y E 31-32 después. Se registra la pérdida por paso y las señales de cada recálculo (s_i, b elegido, cond p90, eventos de P4 en sombra).

## 5. Qué no se afirma
- Que P2, P4 o P5 mejoren el error: no hay medida a 20 000 pasos; T1/T2 son un proxy de horizonte corto con n = 3 y el converso de B sigue en el llano a mitad de entrenamiento.
- Que los topes w_i sean óptimos: son valores de partida que se recalibran con las señales registradas.
- Que M cierre la brecha del directo con el paper (0.19-0.21 frente a 0.117).
- Referencias de literatura: las del barrido (`REVISION_LITERATURA_PIEZO.md`) solo se vieron como resúmenes; verificar con el texto completo antes de citar. El sesgo de horizonte corto (Wu et al. 2018) y el hecho de que clasificar por una corrida corta puede engañar son los motivos por los que T1/T2 no se leen como veredicto.
