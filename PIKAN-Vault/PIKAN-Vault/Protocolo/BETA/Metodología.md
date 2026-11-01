
Desde el punto de vista computacional, la justificación también incluye el

costo y la reproducibilidad de la comparación. La literatura reciente advierte

que las PINNs no sustituyen automáticamente a métodos clásicos como el FEM. Su

valor depende de la formulación, del régimen del problema y de las métricas

utilizadas \cite{grossmann2024}. Por esta razón, el proyecto evaluará no solo

el error de aproximación, sino también el residual físico y el costo

computacional por configuración. Esta decisión permite comparar PINNs y PIKANs

de forma más rigurosa, evitando conclusiones basadas únicamente en un error

final sin considerar estabilidad, tiempo de entrenamiento o complejidad

arquitectural \cite{shukla2024fair,wang2025kinn}.

  

En términos metodológicos, el trabajo comprende la implementación y evaluación

de mecanismos de focalización sobre arquitecturas PIKAN aplicadas a problemas

seleccionados de PDE en una y dos dimensiones espaciales. Para ello, se partirá

de implementaciones base alineadas con el estado del arte. En particular, se

considerarán variantes con actualización adaptativa de grilla como punto de

referencia metodológico. A partir de esa base, se explorarán mecanismos

orientados a concentrar el aprendizaje en regiones de mayor complejidad física.

El desempeño se evaluará contra soluciones analíticas o referencias numéricas

reproducibles obtenidas mediante métodos clásicos

\cite{grossmann2024,lu2021deepxde,liu2024kan}.