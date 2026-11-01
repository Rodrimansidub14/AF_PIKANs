# Marco teórico: ilustración física y solución por problema

Esta versión aplica la [convención solicitada](../../../reports/convencion-figuras-marco.md): Fiziko/MetaPost para ilustraciones físicas y diagramas; gráficas en blanco, verde azulado y naranja para campos y comparaciones numéricas.

## Entregables

- `atlas-figuras.pdf` (7 páginas): comparación compacta de ilustración y solución para P1–P5, con diagramas de arquitectura y la varilla de difusión.
- `marco-teorico-ilustrado.pdf` (40 páginas): capítulo ampliado con explicación física, ecuaciones, figuras y bibliografía.
- `tesis-revision.pdf` (77 páginas): integración en una copia de la tesis completa.
- `illustrations/`: fuentes MetaPost y figuras físicas/arquitecturas.
- `solutions/`: soluciones P1–P3, datos y comprobaciones.
- `dfg/`: FEM de P4, campos y evidencia de verificación.
- `pvdf/`: FEM piezoeléctrico acoplado de P5, campos y evidencia.

Los nombres de los PDF anteriores son salidas del proceso de compilación. La evidencia numérica se conserva junto al cálculo que la produce; los archivos fuente de la tesis se mantienen separados de esta versión de revisión.

## Métodos y significado de los datos

P1 usa la solución exacta seno–seno y una comparación independiente con diferencias finitas. P2 usa la onda viscosa exacta de Burgers, incluidas sus trazas de frontera e iniciales. P3 integra el problema completo de Allen–Cahn con diferencias centradas y BDF, con Jacobiano disperso analítico. Las soluciones P4 y P5 se generan mediante FEM para los respectivos contratos del proyecto; sus comprobaciones y límites están en `dfg/evidence.json` y `pvdf/evidence.json`.

Una ilustración describe datos y mecanismos: no es un campo de solución. Un perfil ideal puede acompañar una derivación, pero no reemplaza la evolución con las condiciones del banco. La varilla explica difusión térmica y se distingue explícitamente de Burgers. La clasificación «SOTA» no se aplica a estas referencias clásicas: se eligen por su verificabilidad.

## Reproducción

Desde la raíz de ProyectoGrad, con Python, NumPy, SciPy y Matplotlib:

```powershell
python var/agent_runs/teoria-visual-v2/solve_scalar_problems.py
python var/agent_runs/teoria-visual-v2/pvdf/solve_pvdf_fem.py
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Resolve-Path var/agent_runs/teoria-visual-v2/vendor).Path
$env:MPLCONFIGDIR = Join-Path (Resolve-Path var/agent_runs/teoria-visual-v2).Path '.cache/matplotlib'
python var/agent_runs/teoria-visual-v2/dfg/solve_dfg.py
# Para las ilustraciones, seguir illustrations/README.md (MetaPost + Fiziko).
python var/agent_runs/teoria-visual-v2/assemble_review.py
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -output-directory=var/agent_runs/teoria-visual-v2 var/agent_runs/teoria-visual-v2/tesis-revision.tex
biber --input-directory var/agent_runs/teoria-visual-v2 --output-directory var/agent_runs/teoria-visual-v2 tesis-revision
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -output-directory=var/agent_runs/teoria-visual-v2 var/agent_runs/teoria-visual-v2/tesis-revision.tex
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -output-directory=var/agent_runs/teoria-visual-v2 var/agent_runs/teoria-visual-v2/tesis-revision.tex
python var/agent_runs/teoria-visual-v2/extract_review.py
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -output-directory=var/agent_runs/teoria-visual-v2 var/agent_runs/teoria-visual-v2/atlas-figuras.tex
python var/agent_runs/teoria-visual-v2/verify_review.py
```

En esta sesión se usa el Python suministrado con el entorno. Las versiones efectivas figuran en cada archivo de evidencia. Las dependencias adicionales y las cachés se alojan dentro de esta carpeta, sin instalación global. Los originales usados para montar la revisión se registran por SHA-256 en `source-snapshot.json`.

## Procedencia

- Biblioteca [Fiziko](https://github.com/jemmybutton/fiziko), copiada en `../teoria-visual/fiziko/upstream/`, con su licencia y commit documentados allí.
- [Caso Allen–Cahn de DeepXDE 1.12.2](https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/allen.cahn.html): ecuación, parámetro y datos iniciales/de frontera. La solución aquí se calcula independientemente con BDF; no se atribuye a un entrenamiento DeepXDE.
- [Documentación primaria de SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html): integrador BDF y control de tolerancias.
- [Benchmark oficial DFG 2D-1, Re=20](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html): geometría, fronteras y valores de referencia para arrastre, sustentación y diferencia de presión.
- Contratos P1–P5 y bibliografía de `capitulos/metodologia/01-problemas.tex` y constitutivo piezoeléctrico de `capitulos/marco/04-piezo.tex`.
- [Página de OpenAI indicada por el investigador](https://openai.com/index/navier-stokes-solution/): referencia cromática para las gráficas. No se incorporan sus afirmaciones sobre Navier–Stokes al contenido científico.

Las comparaciones de refinamiento son diagnósticos numéricos, no cotas demostradas de error absoluto. Las soluciones de referencia explican los problemas; no acreditan por sí mismas el mecanismo de adaptación PIKAN. Esta versión se prepara para revisión del texto científico bajo la política existente.

## Comprobaciones obtenidas

| Caso | Evidencia de la solución presentada |
|---|---|
| Poisson | Error FDM dividido aproximadamente por cuatro al duplicar la resolución; solución exacta disponible. |
| Burgers | Onda exacta; residual analítico evaluado hasta redondeo de doble precisión. |
| Allen–Cahn | Tres mallas; razón de diferencias 3.94, energía decreciente y sensibilidad a tolerancias de aproximadamente 6.14 × 10⁻⁹. |
| Navier–Stokes | FEM P2/P1; cambios medio–fino de 0.24 % en velocidad y 0.99 % en presión de media cero; cierre relativo de caudal 2.54 × 10⁻¹⁵. La sustentación cambia de signo y la diferencia de presión aún cambia 5.18 %: la referencia integral no está convergida. |
| PVDF | FEM P2 acoplado; flecha de −10.598 µm; cambios finales de 0.3105 % en flecha y 0.00167 % en carga. Pruebas de movimiento rígido y deformación afín satisfactorias. |

La auditoría de Navier–Stokes comprobó la orientación de normales y el uso de derivadas físicas en la tracción. La malla local asimétrica y los anillos radiales sin refinamiento proporcional son una explicación plausible de la sensibilidad de la sustentación; no se identificó un fallo de interpolación en ese funcional. El criterio automático del script no incluye la sustentación: su estado nunca debe interpretarse como aceptación de todos los observables sin revisar también `Cl_rel_change`. En esta entrega el estado es `not_certified`.

La compilación, integridad de los originales y ausencia de referencias sin resolver se registran en `review-verification.json`. Se revisaron visualmente las páginas renderizadas y el atlas. Los cálculos y esta revisión fueron realizados con tres subagentes LUNA MAX y revisión de integración por el agente principal.
