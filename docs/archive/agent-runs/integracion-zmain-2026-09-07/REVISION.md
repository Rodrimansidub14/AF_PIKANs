# Revisión e integración sobre la plantilla raíz

## Restricción expresa del investigador: título y objetivos inamovibles

Corrección posterior a la integración: se restauró íntegramente `g-objetivos.tex` desde `originales/g-objetivos.tex`, incluyendo el objetivo general y los **tres objetivos específicos originales**. El título coincide con la versión anterior y se conserva. El investigador indicó que ambos son inamovibles; las revisiones futuras deberán ajustar fundamentos y metodología a ellos, sin reformularlos, ampliarlos ni sustituirlos. La ampliación anterior a cinco objetivos quedó revertida.

El entregable activo es `z-main.tex` y su PDF en la raíz de ProyectoGrad. `i-marco_teorico.tex` y `j-capitulos.tex` cargan módulos de `capitulos/`, conservando la estructura UVG y evitando un documento independiente. Todas las secciones preliminares y de cierre se alinearon con el estado de protocolo, sin inventar resultados.

## Decisiones de escritura

- Capítulo 2: procedencia local del proyecto; remite al capítulo 6 para todo el desarrollo de antecedentes y literatura.
- Capítulo 6: fundamentos, literatura y derivaciones; sin semillas, presupuesto experimental, acciones ni contratos de benchmarks.
- Capítulo 7: espacio de problemas, espacio de soluciones, regla candidata, recursos, evaluación y supervisión.
- Capítulo 8: dependencias y productos de implementación.
- Conclusiones: conclusiones del planteamiento, explícitamente separadas de eficacia futura.

## Trazabilidad de observaciones GAB

Los estados siguientes significan **cambio implementado en texto**, no aceptación científica del asesor. Fuente de correspondencia: `src/thesis_workspace/templates/02_Revision/GAB_29_Anotaciones.md` y la revisión previa basada en adjuntos.

| Observación | Cambio implementado | Archivo / sección |
|---|---|---|
| GAB-01 | Recorrido de PDE a discretización, redes, bases y física; diagramas y aperturas por sección | `capitulos/marco/01` a `05` |
| GAB-02 | Referencias asociadas a afirmaciones y figuras | Bibliografía y capítulo 6 |
| GAB-03 | PDE expandida y siglas identificadas | 6.1 |
| GAB-04 | Fuentes en contexto inicial | 6.1 |
| GAB-05 | Párrafo de panorama de soluciones analíticas, discretizaciones y redes | 6.1 |
| GAB-06 | Soluciones exactas, FEM y papel de referencia | 6.1–6.2 |
| GAB-07 | Definición de dominio, coordenada y tiempo | 6.1 |
| GAB-08 | Operador de frontera, tipos y alcance | 6.1 |
| GAB-09 | Ecuación general introducida y referenciada | 6.1 |
| GAB-10 | Numeración regenerada por LaTeX; sin marcador manual | Documento completo |
| GAB-11 | Jerarquía ANN/DNN/MLP y construcción de capas | 6.3 |
| GAB-12 | Contexto de escalas, capas y normas antes de adaptación | 6.1 |
| GAB-13 | Contexto conciso IA/ML/DL, según solicitud posterior de no explicar en exceso conceptos generales | 6.3 |
| GAB-14 | Definición matemática de frecuencia y anchura de capa | 6.1.1 |
| GAB-15 | Desarrollo FEM desde forma débil al sistema | 6.2.3 |
| GAB-16 | Estimadores, Dörfler y diferencia h/p/hp | 6.2.4 |
| GAB-17 | FEM y otras siglas expandidas | 6.2 |
| GAB-18 | Título explícito de redes neuronales informadas por física | 6.3 |
| GAB-19 | PINN expandida | 6.3 |
| GAB-20 | Pérdida derivada desde residual y condiciones | 6.3.2 |
| GAB-21 | Literatura PINN y antecedente Daniel incorporados sin copiar narrativa | 6.3 y 6.6 |
| GAB-22 | Papel de la malla y soporte local de bases FEM | 6.2 |
| GAB-23 | MLP expandida | 6.3.1 |
| GAB-24 | Subsección de pérdida con entrada autónoma y ecuaciones definidas | 6.3.2 |
| GAB-25 | KAN y PIKAN definidas en contexto | 6.5 |
| GAB-26 | Activación fija, pesos y derivadas explicados antes de KAN | 6.3.1 |
| GAB-27 | Fuentes en formulación y síntesis; cada figura con atribución | 6.5 y 6.7 |
| GAB-28 | Se retiró el mecanismo del marco; se añadieron aperturas entre sección y subsección | Capítulos 6–7 |
| GAB-29 | Se distingue ponderación, muestreo y cambio de representación; no se inventa sigla para la propuesta | 6.4 y 7.3–7.4 |

## Cambios frente a la ideación anterior

- Cinco familias explícitas; se retiró la acumulación de casos sintéticos y extensiones que duplicaban estrés.
- B-splines cúbicas como sustrato inicial. Las alternativas radiales siguen citadas y disponibles como ruta posterior, sin bloquear el inicio.
- Presupuesto de coeficientes exacto por rama y global; distinción entre parámetros, memoria y tiempo.
- Separación entre soporte latente y físico, propuesta barata y beneficio observado, preservación de función y conservación física.
- Acciones concretas de movimiento e intercambio, transferencia con derivadas, continuaciones emparejadas y recuperación completa.
- Controles cercanos y control con el mismo sondeo; la novedad no se atribuye a una simple combinación de técnicas conocidas.
- PVDF con geometría, coeficientes y auditoría de convenciones. Reformulación explícita en vez de supuesto resultado replicado del código no inspeccionado.
- Preprints mantenidos como antecedentes y vías adyacentes. El alcance de lectura se declara cuando solo se dispone de ficha/resumen.

## Alcance comprobado y siguiente etapa

Las verificaciones de esta entrega cubren compilación, enlaces internos, citas, figuras, cinco contratos y algebra de ejemplos. La recreación FEM y Allen–Cahn tiene resultados reproducibles en sus registros. Las referencias convergidas finales de DFG y PVDF y el entrenamiento del mecanismo siguen siendo trabajo de implementación, no resultados implícitos del documento.

La aprobación humana de claims y de una campaña científica permanece separada de esta edición autorizada. No se modificaron archivos de política, AGENTS ni notas canónicas. El estado previo completo de los archivos raíz editados se conserva en `originales/`.
