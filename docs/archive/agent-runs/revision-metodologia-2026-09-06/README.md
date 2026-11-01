# Revisión integrada del protocolo

Versión de trabajo para revisión científica, 6 de septiembre de 2026. Superficie: `var/agent_runs`, workflow `synthesis`, con consulta de literatura. No activa hipótesis, no promueve notas al vault y no modifica políticas.

El usuario autorizó implementar la revisión documental y preparar la metodología previa al desarrollo del mecanismo. Su aclaración indica usar los preprints como citas y apoyo para explorar rutas aledañas, no como motivo para descartar automáticamente la adaptación local.

## Entregables

- `protocolo-revisado.pdf`: documento integrado de revisión.
- `i-marco_teorico.tex`: capítulo 6 desarrollado, con fundamentos, ecuaciones y figuras.
- `j-capitulos.tex`: metodología y plan de desarrollo con comparadores, controles y criterios verificables.
- `revision-y-evidencia.md`: resolución de comentarios, antecedentes y límites de novedad.
- `figuras/`: diagramas y figuras científicas reproducibles.
- `generar_figuras.py`: cálculo FEM ilustrativo y visualizaciones analíticas; no entrena el mecanismo.
- `validacion.json`: resultados reales de verificación del material generado.

La ruta recomendada es un controlador de reasignación de representación con presupuesto fijo y transferencia protegida. Nudos móviles y centros compactos permanecen como alternativas. Se investigará la utilidad marginal del cambio y la asignación entre campos, tomando como precedentes la adaptación residual, por densidad/curvatura y multinivel. Esta combinación es una hipótesis de diseño, no una afirmación de prioridad ni de superioridad.

La compilación incluye introducción, antecedentes, justificación, objetivos, alcance, marco teórico, metodología, plan y bibliografía. No incluye resultados de entrenamiento inexistentes. Las figuras de soluciones exactas se identifican como ilustraciones analíticas; la figura FEM proviene de un cálculo verificable de Poisson, no de la simulación piezoeléctrica futura.

## Verificación realizada

El PDF final contiene 33 páginas, 8 figuras y 28 fuentes citadas. Compila sin advertencias LaTeX, citas indefinidas ni figuras faltantes. Se revisaron las 33 páginas renderizadas y páginas de ecuaciones/metodología a mayor tamaño. La demostración FEM presenta órdenes 1.99 en L2 y 1.00 en gradiente; las verificaciones independientes del banco mediante diferencias finitas de cuarto orden cumplen tolerancia 1e-4. Los valores detallados están en `validacion.json`.

Para reconstruir: ejecutar `compilar.ps1 -PythonExe <ruta-a-python>` desde PowerShell. Requiere NumPy, SciPy, Matplotlib, pypdf, LaTeX con Latin Modern y Biber. En este entorno se utilizó el Python incluido en el runtime de Codex; el Python predeterminado del sistema no inició correctamente. No hace falta entrenar una red para reconstruir estas figuras.

Las decisiones que sí requieren revisión del investigador se enumeran en `revision-y-evidencia.md`. La aprobación del documento y su promoción a la tesis raíz se mantienen separadas de los chequeos técnicos realizados.
