> **CORRECCIÓN (2026-10-25, tarde): la hipótesis de abajo (F amplifica el paso hasta ~31x de forma uniforme) NO se sostiene con el cálculo directo de curvatura.** La rigidez de la dirección más rígida tras F es en mediana 0.65-1.5x la de B (solo 1 % de aristas > 10x). Lo que sí se mide: F empeora el condicionamiento del bloque en el 82-88 % de las aristas (desajuste de dirección, no de escala). Ver `ANALISIS_PIEZO_PROPIEDADES.md`. Se deja el texto original por trazabilidad.

# Diagnóstico del brazo F en piezo (CPU, desde artefactos; 2026-10-25) — hipótesis, n = 3

Datos: artefactos de las 24 corridas B, F, E, J (semillas 20260728-30, converso y directo). F = Gram fijo de la base (uniforme en [-1,1]), P fija e idéntica en todas las aristas y en todos los problemas: valores singulares de P 0.47 / 2.9 (mediana) / 31, cond ≈ 66.

| Señal | B | F | E | J |
|---|---|---|---|---|
| ‖c_eff‖ a los 2 000 pasos, directo | 9-13 | **116-125** | 124-146 | 128-162 |
| ‖c_eff‖ a los 2 000 pasos, converso | 3.2-3.3 | **21-27** | 32-50 | 27-32 |
| Eventos de pérdida > 5× el mínimo previo (tras el paso 2 000), directo | 25-38 | **99-122** | 26-36 | 13-29 |
| Pérdida final, directo | 0.027-0.055 | **0.044-0.148** | 0.017-0.025 | 0.014-0.038 |
| Eventos, converso | 0-2 | 0-1 | 1-2 | 1-2 |

Lectura (no probada):
1. F no es un error de implementación: su P es la misma que usó en los demás problemas del banco, donde ayudó o fue neutra.
2. Con Adam, la actualización por coordenada de c' es ~lr; en c = P c' el paso efectivo en las direcciones con valor singular alto se multiplica hasta ~31. Es un aumento uniforme del paso en **todas** las aristas, ciego a la curvatura de cada una.
3. Piezo tiene una dispersión de trazas del Gauss-Newton entre aristas de 3.6-3.8 décadas (el banco 1.7-3.5): hay aristas de curvatura alta donde ese aumento sobrepasa. B ya muestra picos de pérdida (25-38 eventos en el directo); F los triplica (99-122) y el directo termina en una solución casi nula en las tres semillas.
4. E y J tienen un rango de valores singulares parecido (0.3 a 32), pero su P sale del residuo del propio modelo y se recalcula cada 1 000 pasos: el aumento del paso va a las direcciones planas de la pérdida y no a las curvas. Esa es la diferencia que el banco no podía mostrar: en problemas con poca dispersión entre aristas, F y E coincidían (D5: la ventaja era de la base).
Predicción comprobable: F con P reescalada a máximo singular 1 (o con tasa de aprendizaje dividida) debería recuperar la estabilidad; E no debería depender de eso. No probada.
