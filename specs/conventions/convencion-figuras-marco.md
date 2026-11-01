# Convención de figuras del marco teórico

Convención solicitada por el investigador: cada problema se presenta mediante **una ilustración del problema** y **una gráfica de su solución**. Los diagramas de arquitectura comparten la técnica de dibujo de las ilustraciones. Esta convención visual no altera el banco de problemas ni sus condiciones matemáticas.

## Tres clases de figura

| Clase | Pregunta que responde | Técnica y apariencia | Contenido obligatorio |
|---|---|---|---|
| Ilustración física | ¿Qué sistema se estudia y qué le ocurre? | Fiziko/MetaPost; dibujo de libro de física, tinta y hachurado, fondo blanco. | Objeto o medio, dominio, coordenadas, fuentes, materiales, condiciones de frontera y flechas que expliquen el mecanismo. Indicar «sin escala» si se exagera la geometría. |
| Gráfica de solución | ¿Qué campo satisface el problema y con qué precisión se conoce? | Gráficas numéricas del estilo actual: blanco, verde azulado y naranja, ejes limpios, unidades y escalas visibles. | Campo, dominio, parámetros y tiempo; solución exacta o método numérico identificado; referencia y comprobación de precisión. |
| Diagrama de arquitectura | ¿Qué operaciones se conectan y qué se aprende? | Fiziko/MetaPost; mismo lenguaje de tinta y trazos de las ilustraciones. | Entradas, salidas, operaciones, conexiones y parámetros entrenables. Distinguir un peso escalar de una función aprendida. |

Los colores de las gráficas codifican magnitudes o comparaciones; en las ilustraciones el hachurado identifica materiales y la forma de las flechas identifica acciones. Las leyendas deben poder entenderse sin depender solo del color. No se usa una barra de color para un esquema sin datos calculados.

## Aplicación a cada problema

| Problema | Ilustración física | Solución y referencia |
|---|---|---|
| P1: Poisson | Placa cuadrada, fuente distribuida y valor impuesto en los bordes. Aclarar que el caso manufacturado con fuente de signo variable es una idealización matemática. | Solución exacta seno–seno. Comprobación independiente con diferencias finitas y refinamiento. |
| P2: Burgers | Medio unidimensional, transporte con velocidad dependiente del campo, difusión y frente. | Onda viajera exacta con las trazas iniciales y de frontera del contrato. Verificación por sustitución. |
| P3: Allen–Cahn | Dos estados de fase y zona de transición; competencia entre reacción y difusión. | Evolución del dato inicial y fronteras del banco mediante diferencias centradas y BDF. Refinamiento espacial y sensibilidad a tolerancias temporales. |
| P4: Navier–Stokes (DFG) | Canal, entrada parabólica, cilindro, paredes y salida. | FEM mixto para velocidad y presión. Identificar aproximación geométrica, convergencia y balances comprobados. |
| P5: bimorfo PVDF | Dos capas adheridas, empotramiento, electrodos, polarización y condiciones de interfaz. | FEM piezoeléctrico acoplado: desplazamiento y potencial. Refinamiento, flecha de punta y carga de electrodo; indicar el factor si se amplifica una deformada calculada. |

La varilla con transferencia de calor ilustra la ecuación de difusión, no la ecuación de Burgers. Puede introducir el operador difusivo antes del banco canónico, sin constituir un sexto experimento.

## Rigor de las soluciones

1. Usar una solución exacta cuando exista y comprobar ecuación, dominio y condiciones.
2. En una solución numérica, conservar las condiciones del problema y describir discretización, tolerancias y refinamiento. Un residual algebraico pequeño verifica el sistema discreto; no demuestra por sí solo precisión frente a la PDE.
3. Identificar la convergencia como evidencia de sensibilidad a la resolución, no como cota demostrada del error absoluto. Si no se alcanza la precisión deseada, declararlo con las cifras obtenidas.
4. Usar métodos del estado del arte cuando aporten una comparación pertinente y reproducible. La etiqueta SOTA no sustituye una verificación y no se atribuye a cualquier método reciente.
5. Reservar «solución del problema» para el campo que corresponde a sus datos. Un perfil ideal, una deformada cualitativa o una geometría con flechas se presenta como explicación del mecanismo.
6. Los cálculos de referencia para estas figuras explican el marco teórico; no constituyen por sí mismos resultados de validación del mecanismo PIKAN propuesto.

## Orden de lectura y pies

Cada caso sigue **sistema físico → ecuaciones y condiciones → solución → interpretación y precisión**. Los dos tipos de figura se colocan cerca de su discusión y se identifican por separado.

Ejemplo de pie físico: «Ilustración del problema P5: capas adheridas, empotramiento y electrodos. Dibujo propio con Fiziko; altura exagerada, sin escala. Las flechas indican la orientación del campo impuesto».

Ejemplo de pie de solución: «Solución del caso P3 mediante diferencias centradas y BDF, con las condiciones indicadas en el texto. Se muestran el campo, cortes temporales y la diferencia entre refinamientos. Los parámetros y las comprobaciones se conservan con el código de generación».

Los archivos editables, datos, versiones y evidencia se conservan junto a cada figura. La atribución y licencia de Fiziko se mantienen con su copia local. Las actualizaciones del texto científico se revisan mediante el procedimiento del proyecto; esta convención no modifica permisos.
