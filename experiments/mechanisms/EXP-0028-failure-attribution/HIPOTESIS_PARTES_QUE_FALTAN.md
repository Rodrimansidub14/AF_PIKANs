# Lo que el mecanismo no resuelve en piezo: hipótesis, y dónde no estamos mirando (2026-10-25)

Estado: **hipótesis de trabajo, ninguna probada, ninguna requiere cómputo para formularse.** Se apoyan en las 24 corridas de piezo y sus artefactos (n = 3 semillas por brazo y problema) y en los cálculos de [ANALISIS_PIEZO_PROPIEDADES.md](ANALISIS_PIEZO_PROPIEDADES.md) y [analisis_piezo/analisis_extra.txt](analisis_piezo/analisis_extra.txt). Para cada parte que E no resuelve: qué muestran los datos, qué se sospecha, qué cambiaría en el diseño y cuál sería la comprobación más barata.

Una advertencia de alcance. Lo que el mecanismo toca hoy es una sola cosa: los **coeficientes** de las capas RBF, reparametrizados con un P por arista. Quedan fuera de su alcance los demás parámetros (el voltaje flotante del directo, las bases, los centros y anchos), la forma de la pérdida, el optimizador y la formulación mixta. Casi todo lo que sigue vive ahí.

## 1. El potencial φ en el converso (E lo empeora: ×1.18 en mediana, 0.044 → 0.052)
**Datos.** El error de φ en el converso de B está en el interior de las capas (43 % y 45 % de la energía del error), no en los electrodos (1 %; son condiciones duras) ni en la interfaz (7 %), y crece hacia el extremo libre (36 % en el último cuarto). Con E queda repartido de forma pareja (23 / 24 / 26 / 27 %), pero el total sube un poco. En la pérdida, las constitutivas D son el 14 % y E las empeora ×1.33, mientras reduce las de σ (×0.73) y el equilibrio (×0.06). La curvatura del Gauss-Newton de la capa de salida está **casi toda en un solo canal, el de σ_yy: 82-86 % en el converso, 99 % en el directo**, y φ no llega a 0.1 %.
**Hipótesis H1.** E conoce la pérdida tal como está ponderada. La pérdida está dominada por σ y el equilibrio, y el precondicionador atiende esa parte; φ, con muy poco peso en la curvatura, queda a merced de lo que el entrenamiento le deje.
**Qué cambiaría.** Un P que equilibre las contribuciones por grupo físico (σ, D, φ) en vez de heredar los pesos de la pérdida; o bloques separados para los canales de salida con poca curvatura.
**Comprobación barata (sin entrenar).** Evaluar el Jacobiano por término sobre los estados guardados (un recálculo por estado, CPU) y ver qué parte de la traza de cada arista aporta cada término y cada canal. Si φ y D no aportan casi nada a la traza, H1 queda respaldada.

## 2. La brecha del directo con el paper (0.19-0.21 frente a 0.117) y el potencial flotante
**Datos.** El potencial medio del electrodo superior flotante sale **−2.3 a −2.6 V frente a −3.10 V del FEM** en B, E y J (error relativo 20-22 %; en F, entre −0.06 y −1.1 V). El parámetro del voltaje flotante converge a ≈ −0.18 (unidades escaladas) en B, E y J, igual en los tres: no lo mueve el mecanismo. Su gradiente es 3-4 órdenes menor que el de los coeficientes (≈ 1e-4 a 1e-3 frente a ≈ 6). La banda del electrodo superior concentra el 35-38 % de la energía del error de φ, más el 36 % de la capa superior: un error casi uniforme en esa zona es lo que produce un desplazamiento del voltaje común. En el directo, φ y u tienen errores casi iguales por semilla (0.18 / 0.18, 0.26 / 0.24, 0.25 / 0.20 en B): van acoplados. El paper reportó −2.92 V (6 % de error) con su MLP.
**RESULTADO DE LA COMPROBACIÓN (2026-10-25, sobre los estados finales guardados, 9 corridas del directo; `analisis_piezo/check_Vf.md`): H2 NO se sostiene.** El V̂_f aprendido ya es el mínimo exacto de la pérdida: el valor exacto coincide con el aprendido a 3 cifras en las 9 corridas (diferencias de 0.001) y sustituirlo no cambia φ, u, v ni la pérdida (variaciones en la tercera cifra). El potencial 20 % bajo no viene de un escalar mal optimizado, sino que es el valor que minimiza la pérdida dado el resto de la red; el error nace en los demás campos (o en el término de carga neta como aproximación discreta). 
**Hipótesis H2 (refutada en su forma original).** El voltaje flotante es un **escalar global** que el mecanismo no precondiciona y que la condición de carga neta cero determina de forma débil (promedio sobre 256 puntos, remuestreado cada 200 épocas). Se queda en un valor ≈ 20 % bajo el correcto, y como el directo acopla φ con u y v, parte del error del directo cuelga de ahí.
**Qué cambiaría.** Tratar el escalar fuera de las capas: su propia escala o paso, o resolverlo en forma cerrada (la carga neta es lineal en V_f dado el resto de la red: se puede fijar a su valor exacto en cada recálculo, proyectando V_f fuera de la optimización).
**Comprobación barata.** Con los campos guardados, calcular el V_f que anula exactamente la carga neta sobre la malla de evaluación y comparar con el aprendido; y estimar cuánto del error de u, v y φ desaparece al corregir solo ese escalar (cálculo sobre el campo guardado, sin entrenar).

## 3. Los saltos de pérdida en el directo (B 25-38; F 99-122; E 26-36; J 13-29)
**Datos.** El gradiente del voltaje flotante sube en los saltos (≈ 1e-3 a 2e-3 frente a su mediana de 2e-4 a 1e-3); la correlación entre el log de ese gradiente y el log de la pérdida es 0.4-0.54. Los saltos aparecen desde el paso 2 000 en adelante, con el error estable. En el converso casi no hay (0-2).
**Hipótesis H3.** Los saltos están ligados a la restricción global (la carga neta, el peso 10 de los bordes y el remuestreo de los puntos del electrodo cada 200 épocas) más que al condicionamiento por arista: E no los reduce, J algo. Es un problema de la formulación del directo, no de la geometría de los coeficientes.
**Qué cambiaría.** Estabilizar el escalar (H2), o medidas del optimizador (decaimiento del paso, recorte de gradiente, promedio de parámetros para evaluar), que quedan **fuera de la receta del paper** y habría que declararlas como extensión.
**Brecha de datos.** La pérdida se guardó cada 100 pasos; no se puede comprobar si los saltos coinciden con los remuestreos (cada 200). Un registro por paso lo resolvería en la próxima corrida; hoy no se puede afirmar.

## 4. La lotería de B en el converso (0.048 / 0.367 / 0.180; E reduce el CV de 0.81 a 0.42)
**Datos.** B pasa 6 000 a 8 000 pasos cerca de la solución nula antes de bajar de 0.5; E sale antes en dos de tres semillas (3 500 y 5 500 frente a 6 000-8 000). Pero el orden de salida **no** explica el error final: B salió a los 6 000 (0.048), 7 000 (0.367) y 8 000 (0.180).
**Hipótesis H4.** La lotería no es solo cuándo se sale del llano, sino hacia qué cuenca se baja; E acorta el llano y estrecha el resultado, pero no se sabe por qué mecanismo. Es la que menos soporte tiene.
**Qué cambiaría.** Recálculos más frecuentes al inicio (la primera fase decide la cuenca) o calentamiento del mecanismo.
**Comprobación barata.** Comparar, sobre los estados guardados cada 2 000 pasos, la distancia entre semillas en el espacio de coeficientes efectivos (¿B diverge entre semillas, E converge?).

## 5. Lo que ninguno de los brazos arregla: la consistencia constitutiva de la formulación mixta
**Datos.** La discrepancia entre las tensiones predichas por la red y las reconstruidas desde los gradientes de u, v y φ es **alta en todos los brazos**: σ_xx entre 0.79 y 1.75 (80-100 % o más) y D_x entre 1.0 y 2.7; D_y es buena en el converso (0.17-0.20) y mala en el directo (0.78-0.99). (σ_yy y τ_xy salen 1.00 en todos: probablemente un artefacto de magnitudes casi nulas, no informan.) E no la mejora de forma consistente.
**Hipótesis H5.** El límite de la formulación mixta que declara el propio paper ("la consistencia constitutiva queda como limitación") pesa en el error de u, v y φ, y es independiente del optimizador: mejorar el condicionamiento no la arregla.
**Qué cambiaría.** Fuera de nuestro alcance actual: pesos para los residuos constitutivos, o una penalización de consistencia. Es otra palanca (la forma de la pérdida), no una variante del mecanismo.

## Dónde no estamos mirando (resumen)
1. Los parámetros que el mecanismo no toca: el voltaje flotante (H2), las bases, los centros y los anchos.
2. La forma de la pérdida: pesos por grupo físico (H1) y consistencia constitutiva (H5).
3. El optimizador y su calendario: el paper usa Adam con paso constante y sin L-BFGS; nuestras palancas no lo tocan (H3).
4. El enrutamiento del gradiente del converso: el Jacobiano del Gauss-Newton usa el mismo `detach` que el entrenamiento, así que E ve una versión del problema distinta de la exacta. No sabemos si esto contribuye a φ (H1) y es verificable sin entrenar.
5. Los registros: pérdida por paso (H3) y los términos con poca masa.

## Qué cambia en el diseño del mecanismo
E sigue como base. Lo que estas hipótesis añaden es una lista de **extensiones candidatas, ordenadas por apoyo en los datos**: (i) tratar el voltaje flotante fuera de las capas (H2, apoyo medio-alto, cálculo barato); (ii) equilibrar el P por grupo físico (H1, apoyo medio); (iii) medidas de optimización para el directo (H3, apoyo bajo, fuera de la receta); (iv) consistencia constitutiva (H5, otra palanca). Ninguna se prueba hasta firmar su especificación.
