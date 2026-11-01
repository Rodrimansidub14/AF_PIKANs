# Análisis de propiedades de piezo: qué mecanismo acoplar y por qué (sin experimentos nuevos)

2026-10-25. Datos: las 24 corridas de piezo (B, F, E, J × semillas 20260728-30 × converso y directo, receta del paper, 20 000 pasos) y sus artefactos (estados, trazas y P por arista en cada recálculo, términos de la pérdida cada 100 pasos). Todo calculado en CPU con [analisis_piezo.py](analisis_piezo/analisis_piezo.py); tablas completas en [analisis_piezo.md](analisis_piezo/analisis_piezo.md). **n = 3 semillas: es descripción con números, no inferencia; no hay p-valores.**

## 1. Lo que hace cada brazo en piezo (error absoluto `l2re_u`)
| problema | B | F | E | J | MLP del paper |
|---|---|---|---|---|---|
| converso, media (por semilla) | 0.198 (0.367 / 0.048 / 0.180) | 0.236 (0.234 / 0.232 / 0.241) | **0.053** (0.050 / 0.077 / 0.033) | 0.089 (0.107 / 0.038 / 0.121) | 0.107 |
| directo, media | 0.208 (0.180 / 0.240 / 0.204) | **0.803** (0.973 / 0.634 / 0.801) | 0.193 (0.129 / 0.198 / 0.253) | 0.181 (0.170 / 0.180 / 0.193) | 0.117 |

Tres lecturas que la razón con B ocultaba:
- **En el converso B es una lotería** (CV 0.81: de 0.048 a 0.367). Las razones E/B que hacen «empeorar» a E en la semilla 29 vienen de que B resultó excepcionalmente bueno ahí, no de que E falle: E queda entre 0.033 y 0.077. **E reduce la media (0.198 → 0.053) y la variabilidad (CV 0.81 → 0.42)**, y queda por debajo del MLP del paper. J también mejora (0.089).
- **F en el converso no es inestable:** cae en el mismo valor (0.23-0.24, CV 0.02) en las tres semillas, un poco peor que la media de B. Es un llano determinista, no una explosión. En el directo sí es malo (0.80, cuatro veces B) y con muchos saltos de pérdida (99-122 frente a 25-38 en B).
- **En el directo nada del conjunto actual cierra la brecha con el paper** (todos 0.18-0.21 salvo F; el MLP del paper 0.117). E mejora 7 %, J 13 %, con semillas que se cruzan. El directo tiene un problema que no parece de condicionamiento: B ya da 25-38 saltos de pérdida.

Dónde actúan: la pérdida de B es 76 % constitutivas σ en el converso y 57 % σ más 41 % equilibrio y Gauss en el directo. E reduce las constitutivas σ (×0.73 converso, ×0.39 directo) y el equilibrio y Gauss (×0.06 converso); en el converso empeora las constitutivas D (×1.33) y el potencial φ (×1.18). F aumenta las constitutivas σ (×1.4 y ×2.5) y, en el directo, el equilibrio y Gauss ×7; a la vez mejora el error de σ_xx (×0.70): ajusta las tensiones y rompe los desplazamientos, la inconsistencia constitutiva que el paper señala como su limitación.

## 2. Por qué: cómo están condicionados los bloques de piezo
Con la traza y la P de E en cada recálculo se reconstruye el bloque Gauss-Newton de cada arista (secciones 4 y 5 del análisis completo):
- **Cada bloque es casi de rango uno:** su λmax concentra casi toda la traza (cond regularizado ≈ 1e4, el máximo posible con la regularización de 1e-3, en la mediana de las aristas; E reduce la dirección más rígida a ≈ 0.09-0.12 de la de B). La curvatura de la pérdida vive en una o pocas direcciones de 11.
- **Entre aristas la curvatura varía 3.3-4.4 décadas, y crece durante el entrenamiento** (dispersión de trazas del converso 3.28 → 3.95 entre los pasos 2 000 y 20 000; directo 3.75 → 4.40). La medida a la inicialización (3.6-3.8) subestima lo que el mecanismo enfrenta después.
- **F empeora el condicionamiento del bloque en el 82-88 % de las aristas** (cond mediano tras F 1.7e4 frente a 9e3 de B; hasta 1e5 en el p95). Su P es el Gram de la base para entradas uniformes en [-1, 1]; la dirección rígida real de cada arista depende de cómo se distribuyen sus entradas y del acoplamiento de la pérdida, y F no las ve. Es un desajuste de **dirección**, no de escala.

## 3. Una hipótesis que se retira
`F_diagnosis_piezo.md` y el borrador de diseño decían que F falla porque su P multiplica el paso hasta ~31 veces de forma uniforme. **Este cálculo no lo sostiene:** la rigidez de la dirección más rígida tras F es, en mediana, 0.65-1.5 veces la de B (p95 3.4-4.6; solo 1 % de aristas por encima de 10×). Tampoco sostiene que reescalar P a máximo singular 1 (la variante «Fn») arregle F: el problema es la dirección, no la escala. Esa variante queda sin respaldo y se retira del diseño.

## 4. Qué propiedades de piezo orientan la elección
| propiedad medida | valor en piezo | qué favorece |
|---|---|---|
| bloques casi de rango uno por arista | cond ≈ 1e4 en la mediana | un P que siga la dirección real (E); una diagonal (J) o un Gram fijo (F) la ven peor |
| dispersión de curvatura entre aristas y crecimiento en el entrenamiento | 3.3-4.4 décadas, creciente | P recalculado durante el entrenamiento (E, cada 1 000 pasos), no una regla fijada a la inicialización |
| pérdida dominada por constitutivas σ y equilibrio | 57-76 % más hasta 41 % | un mecanismo que actúe en esos términos: es donde E gana |
| B en el converso: lotería entre semillas | CV 0.81 | E reduce la variabilidad: su valor es también robustez |
| B en el directo: 25-38 saltos de pérdida | error estable, pérdida inestable | nada del conjunto actual toca esto; parece un problema de optimización, no de condicionamiento |

## 5. Estimación del mecanismo a acoplar en piezo
- **E es la base:** única pieza que baja la media y la varianza en el converso, y la única consistente con el condicionamiento medido. Costo: ≈ 10-15 s por recálculo, 20 recálculos, ≈ 4-5 % del tiempo.
- **J merece una segunda mirada** (mejora el converso a 0.089 y es el mejor en el directo, 0.181), con la salvedad de que en el banco dañaba en Burgers y KPP corto. Es más barato que E y sirve de control de cuánto aporta el acoplamiento entre coeficientes.
- **F y Fn quedan fuera:** F empeora el condicionamiento y Fn no ataca la dirección.
- **Lo que el mecanismo no resuelve, y hay que decirlo:** el directo (brecha con el paper de ~1.6×), el potencial φ del converso (E ×1.18) y la lotería del B inicial.
- **Para probar después, no ahora:** si un recálculo más frecuente o la reducción a las pocas direcciones rígidas (rango bajo) ayuda al directo; si E con medidas de estabilidad (recorte de gradiente, ya fuera de la receta del paper) reduce los saltos.
