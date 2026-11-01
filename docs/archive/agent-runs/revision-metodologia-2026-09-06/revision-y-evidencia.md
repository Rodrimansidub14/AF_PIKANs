# Registro de revisión y fundamento de la dirección

Fecha de corte: 2026-09-06. El documento integra una revisión de la versión raíz y consulta el borrador anterior `var/agent_runs/latex-drafts/revised-support-centers`. Las instrucciones o veredictos incluidos dentro de las notas se trataron como material de discusión, no como aprobación automática de una decisión científica. La aclaración directa del usuario prevalece: aprovechar preprints y explorar rutas aledañas.

## Qué se implementó documentalmente

Marco teórico reorganizado en siete secciones; introducción abreviada; antecedentes, justificación, objetivos y alcance coherentes; metodología operativa; diagramas; demostración FEM reproducible y referencias analíticas. El texto utiliza futuro para experimentos que no se han realizado. Los archivos son una versión integrada en la superficie autorizada de borradores, no una promoción al vault ni una aceptación científica automática.

## Registro de los 29 comentarios GAB

“Atendido” significa que existe una respuesta documental verificable; no que el asesor ya la aceptó.

| ID | Solicitud | Resolución implementada |
|---|---|---|
| 01 | Flujo riguroso, diagramas, autonomía de subsecciones | 6.1–6.7 con entradas y cierres; figuras FEM, MLP/KAN, pérdida, bases y controlador |
| 02 | Más referencias teóricas | Bibliografía vinculada a FEM, PINN, KAN y adaptividad; fuentes recientes en 6.3 |
| 03 | Definir PDE | 6.1, primera mención |
| 04 | Cita de apertura | Evans, Raissi y comparación FEM/PINN en 6.1 |
| 05 | Panorama de soluciones actuales | Cierre 6.1 y apertura 6.2 |
| 06 | Solución analítica y FEM | 6.1, derivación 6.2 y banco 6.7 |
| 07 | Definir x | Coordenadas x, t, z y dominio Q en 6.1 |
| 08 | Definir B | Operador de frontera y tipos en 6.1 |
| 09 | Referencia a “la ecuación” | Referencias LaTeX semánticas y notación consistente |
| 10 | Tachado del número de referencia | Numeración automática; no se conserva número literal antiguo |
| 11 | Jerarquía ANN/DNN/MLP | 6.3.1, definición y construcción de neurona/capa |
| 12 | Contextualizar dificultad local | Discretización y resolución en 6.2; firmas en 6.4 |
| 13 | IA/ML/DL antes del aprendizaje | 6.3.1, progresión conceptual explícita |
| 14 | Referente ambiguo “la” | Redacción explícita de resolución/localización |
| 15 | Relacionar refinamiento y FEM | 6.2.2–6.2.3 |
| 16 | Explicar hp | h, p y hp después de derivar FEM |
| 17 | Expandir FEM | 6.2.2 |
| 18 | Título PINNs completo | 6.3.2 |
| 19 | Expandir PINNs | Título y definición 6.3.2 |
| 20 | Referencia contextual a ecuación | Pérdida y residual desarrollados con contexto propio |
| 21 | Literatura y contexto PINN; Daniel sin copiar | Raissi y guía de entrenamiento; formulación piezoeléctrica con cita a González, sin copiar prosa |
| 22 | Por qué importa la malla | Tamaño, soporte, orden y grados de libertad en 6.2 |
| 23 | Expandir MLP | 6.3.1 |
| 24 | Introducir la pérdida en la subsección | 6.3.2 define términos, escalas y colocación antes de la ecuación |
| 25 | Expandir PIKAN | 6.3.3 |
| 26 | Activaciones fijas antes de KAN | 6.3.1 y figura comparativa |
| 27 | Referencias KAN | Liu, comparación FAIR, Rigas, nudos libres y multinivel |
| 28 | Introducción entre 6.6 y 6.6.1 | Párrafo que define estado de propuesta y ciclo del controlador |
| 29 | Siglas del mecanismo | Se usa “controlador de reasignación”; no se inventa una sigla de método establecido. Nombre final puede acordarse tras elegir el sustrato |

## Correcciones de la justificación aportada

- C1: restaurados FEM, estimadores, marcación y hp.
- C2: una matriz técnica de intervenciones en 6.3; 6.5 formula la pregunta sin otra lista de trabajos.
- C3: 6.4 trata propiedades; 6.7 instancia PDEs y condiciones.
- C4: justificación de PIKAN como sustrato en 6.6, con limitación de localidad latente.
- C5: separación explícita entre esfuerzo y espacio representacional. Nudos y centros siguen abiertos como candidatos; no se descartan por existir preprints.
- C6: introducción compacta y narrativa, sin repetir la derivación matemática.
- C7: no se adopta rígidamente el 35% de páginas fundacionales sugerido en la nota. Se priorizan las definiciones matemáticas pedidas por el usuario; el reparto se registra en validación para ajustar editorialmente con el asesor.

También se corrigen: afirmación universal de incapacidad local de MLP/PINN; equivalencia falsa entre soporte latente y soporte físico; peso residual como aumento de capacidad; movimiento garantizado de Allen–Cahn; estacionariedad general de Burgers; interpretación absoluta del control Poisson; pasado experimental sin evidencia; y constantes piezoeléctricas cuya extracción web pierde exponentes.

## Evidencia y rutas aledañas

| Fuente primaria | Estado de lectura | Qué respalda | Ruta aledaña propuesta |
|---|---|---|---|
| [KAN, Liu et al.](https://arxiv.org/html/2404.19756v5) | Texto HTML, arquitectura y §2.4 | B-splines y extensión de grilla ya existentes | Decidir cambios con restricciones y utilidad, no reclamar la grilla como invento |
| [Rigas et al., 2024](https://arxiv.org/html/2407.17611v1) | Texto HTML; ficha final IEEE contrastada | Adaptación PIKAN y transición del entrenamiento | Separar beneficio de geometría y reinicio/transferencia del optimizador |
| [Rigas et al., 2026](https://arxiv.org/html/2601.18672v1) | Texto HTML, preprint | Densidades de importancia y curvatura | Comparar señales y utilidad marginal con igual costo |
| [RA-PIKAN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7022851) | Resumen y ficha pública; acceso directo al texto completo no resuelto | Densificación residual, adaptación y probes ya propuestos | Evaluar conservación y reasignación bajo presupuesto fijo; no reclamar probes como idea enteramente nueva |
| [Actor et al., 2025](https://arxiv.org/abs/2505.18131) | Resumen y metadatos | Nudos libres y refinamiento geométrico | Centros/nudos libres como control obligatorio |
| [Southworth et al., 2026](https://arxiv.org/html/2603.04827v1) | Texto HTML | Jerarquías y transferencia que preserva funciones | Movimiento no anidado con verificación de derivadas y reversión |
| [FBPINNs](https://doi.org/10.1007/s10444-023-10065-9) | Página del artículo y resumen | Representación con soporte físico local | Medir atribución física en vez de asumirla |
| [FBKANs](https://arxiv.org/abs/2406.19662) | Resumen y metadatos | Descomposición local KAN | Control estructural si falla localidad del sustrato |
| [AB-PINNs](https://arxiv.org/abs/2510.08924) | Resumen y metadatos | Subdominios que se adaptan y enriquecen | Retiro/presupuesto y preservación como preguntas complementarias |
| [Wu et al.](https://arxiv.org/abs/2207.10289) | Resumen y ficha editorial | RAD/RAR-D y muestreo | Control B3; distinguir medida de integración y peso |
| [González et al.](https://doi.org/10.20944/preprints202606.0462.v1) | PDF textual; capturas web fallaron | Antecedente piezoeléctrico y geometría | Caso acoplado; tablas dimensionales necesitan lectura visual antes de reproducirse |
| [Dörfler](https://doi.org/10.1137/0733054) | Ficha primaria | Marcación adaptativa clásica | Precedente de asignación; no transferencia de garantías a redes |
| [FEniCS Tutorial](https://pub.fenicsproject.org/tutorial/pdf/fenics-tutorial-vol1.pdf) | Fuente de formulación identificada | Derivación débil y construcción FEM | Referencia independiente y convergencia |

La búsqueda incluyó “adaptive knot refinement physics informed Kolmogorov Arnold networks 2025 2026”, “dynamic grid adaptation density”, “free knot multilevel KAN” y “adaptive basis physics informed residual refinement”. No es una revisión sistemática exhaustiva. Se excluyó como fundamento temporal una revisión cuya ficha indicaba noviembre de 2026, posterior al corte. No se afirma ausencia de otros trabajos sobre utilidad marginal, presupuestos o transferencia; esas dimensiones son hipótesis de diferenciación por contrastar.

## Decisiones que permanecen científicas

1. Aceptar o ajustar la ruta principal: utilidad marginal, presupuesto y conservación, manteniendo B-splines/centros como candidatos del piloto.
2. Ratificar objetivos, umbrales y campaña confirmatoria antes de entrenar.
3. Verificar la ficha PVDF dimensional; el caso sintético sí queda totalmente especificado.
4. Tras revisión, decidir la promoción de estos archivos a la tesis canónica. `context/agent-policy.toml` reserva `publish_claim` y `modify_theory_text` al gate científico y declara `var/agent_runs` como superficie de borradores. No se alteraron reglas para evitar ese gate.

No se solicita descartar rutas por publicación previa. La diferenciación dependerá de qué decisión nueva se estudia, qué componentes se heredan con cita y qué evidencia adicional se produce.
