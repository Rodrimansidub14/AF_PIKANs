# Marco teórico ilustrado: versión para revisión

Esta ampliación explica los mecanismos de las cinco familias ya definidas en el proyecto. Mantiene los contratos experimentales y conserva los capítulos fuente. Los fragmentos nuevos se integran en una copia del documento para revisión científica humana conforme a `context/agent-policy.toml`.

**Entrega:** `marco-teorico-ilustrado.pdf` contiene 30 páginas de marco teórico y 4 de bibliografía. `tesis-revision.pdf` permite revisar la integración en la tesis completa (71 páginas). Se añadieron siete figuras: seis generadas por fórmulas/esquemas y una con Fiziko. La compilación final no presenta referencias indefinidas ni cajas desbordadas; se revisaron las páginas renderizadas y las fórmulas pasaron las comprobaciones reproducibles.

## Contenido nuevo

- Poisson: fuente, curvatura, frecuencia distribuida y cortes exactos con amplitud común.
- Burgers: frente estrecho, gradiente, movimiento y cancelación de los términos de la PDE.
- Allen–Cahn: potencial de doble pozo, reacción y perfil ideal de interfaz en la recta. Este perfil no resuelve el problema inicial y de frontera del banco.
- DFG: canal y cilindro a escala, entrada parabólica y significado del balance de momento. No se dibujan líneas de corriente calculadas.
- PVDF: geometría, fronteras, orientación del campo, constitutivo por capa y mecanismo de flexión. La deformada es esquemática.
- Interfaz: ejemplo dieléctrico normalizado que muestra potencial continuo, pendiente discontinua y flujo continuo. Sus dos permitividades ilustrativas no son las del caso PVDF.
- Fiziko: ilustración física producida con la biblioteca MetaPost solicitada, con fuentes y licencia en `fiziko/`.

## Diseño visual

Fondo blanco, trazos limpios, anotaciones breves, paneles comparables y paleta verde azulado/naranja. La página [de OpenAI compartida](https://openai.com/index/navier-stokes-solution/) se usa como referencia visual; no se incorporan sus afirmaciones científicas al marco teórico. La paleta implementada es una interpretación propia, no una guía oficial de marca. [Fiziko](https://github.com/jemmybutton/fiziko) se usa para esquemas físicos; Matplotlib evalúa y grafica las fórmulas analíticas y la geometría exacta.

Cada gráfica se entrega como PDF vectorial (mapas rasterizados dentro del PDF), SVG y PNG. Las ecuaciones y los pies explican qué representa cada figura y qué parámetros utiliza.

## Archivos y reproducción

Desde la raíz del repositorio, usando un Python con NumPy y Matplotlib:

```powershell
python var/agent_runs/teoria-visual/generate_figures.py
python var/agent_runs/teoria-visual/verify_math.py
python var/agent_runs/teoria-visual/assemble_review.py
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -output-directory=var/agent_runs/teoria-visual var/agent_runs/teoria-visual/tesis-revision.tex
biber --input-directory var/agent_runs/teoria-visual --output-directory var/agent_runs/teoria-visual tesis-revision
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -output-directory=var/agent_runs/teoria-visual var/agent_runs/teoria-visual/tesis-revision.tex
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -output-directory=var/agent_runs/teoria-visual var/agent_runs/teoria-visual/tesis-revision.tex
python var/agent_runs/teoria-visual/extract_review.py
```

El generador fija la caché gráfica dentro de esta carpeta. `math-verification.json` registra comprobaciones de las fórmulas: convergencia del Laplaciano discreto de Poisson, residual de la onda Burgers en todos sus parámetros, equilibrio del perfil ideal Allen–Cahn, velocidad media de DFG y continuidad del flujo dieléctrico. La comprobación de deformación libre del PVDF valida una tendencia local bajo campo supuesto; no es una simulación del bimorfo unido.

`source-snapshot.json` identifica por SHA-256 los originales usados para montar la copia. Los cambios propuestos están en `pdes-expansion.tex` y `piezo-expansion.tex`; los archivos `01-aproximacion-ampliada.tex`, `04-piezo-ampliada.tex` e `i-marco-ampliado.tex` muestran cómo quedan integrados. No es necesario copiar una tesis completa para promover los cambios: basta revisar esos dos fragmentos y sus figuras, y aplicar las inserciones correspondientes a los dos capítulos.

## Estado científico

Las figuras explican ecuaciones y condiciones ya presentes; no son evidencia de mejora de PINN o PIKAN. La aceptación de la nueva interpretación y su promoción al texto canónico quedan sujetas a revisión humana. No se modifican el vault, la política del proyecto ni sus definiciones de PDE.
