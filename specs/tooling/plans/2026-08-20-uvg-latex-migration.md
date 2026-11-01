# UVG LaTeX Repository Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promover la plantilla UVG 2019 a la raíz de `ProyectoGrad`, separar el código bajo `src/` y dejar un documento principal compilable y verificable.

**Architecture:** La raíz será el límite del documento LaTeX y contendrá `z-main.tex`, sus módulos, la bibliografía y los recursos de plantilla. El código operativo vivirá en `src/`, mientras notebooks, presentaciones, vault y recursos auxiliares permanecerán como límites independientes. Las versiones antiguas de LaTeX se moverán a `reports/archive/latex-legacy/` y las salidas nuevas a `artifacts/latex/`.

**Tech Stack:** LaTeX/MiKTeX, `pdflatex`, `biber`, `biblatex`, PowerShell, Python, Graphify y Poppler (`pdftotext`/`pdftoppm`).

**Spec:** `specs/tooling/2026-08-20-uvg-latex-migration-design.md`

## Global Constraints

- La raíz contiene el documento LaTeX principal y sus archivos de entrada.
- `src/` contiene código ejecutable y sus manifiestos/documentación operativa.
- `notebooks/`, `presentaciones/`, `PIKAN-Vault/` y `recursos/` conservan su ubicación actual.
- `reports/archive/latex-legacy/` conserva las versiones antiguas de LaTeX sin eliminar datos.
- `artifacts/` contiene salidas regenerables y `var/` contiene estado temporal; ambos quedan ignorados por Git.
- No se modifica el repositorio `obs-data`.

---

### Task 1: Definir mapas de navegación y configuración de salidas

**Files:**
- Create: `README.md`
- Create: `src/README.md`
- Create: `reports/README.md`
- Modify: `.gitignore`
- Modify: `.vscode/settings.json`

**Interfaces:**
- Consumes: la estructura de límites aprobada en `specs/tooling/2026-08-20-uvg-latex-migration-design.md`.
- Produces: reglas de ubicación documentadas y una carpeta de salida LaTeX estable en `artifacts/latex/`.

- [ ] **Step 1: Crear el mapa raíz** con las responsabilidades de `z-main.tex`, `src/`, `notebooks/`, `presentaciones/`, `PIKAN-Vault/`, `recursos/`, `reports/` y `artifacts/`.
- [ ] **Step 2: Crear los mapas de `src/` y `reports/`** indicando que `src/` contiene código operativo y que `reports/archive/` es histórico, no fuente normativa.
- [ ] **Step 3: Actualizar `.gitignore`** para ignorar `artifacts/latex/`, `var/`, `graphify-out/` y auxiliares LaTeX generados en la raíz.
- [ ] **Step 4: Actualizar LaTeX Workshop** para usar `%DIR%/artifacts/latex` y mantener la receta `pdflatex → biber → pdflatex ×2`.
- [ ] **Step 5: Verificar** que los archivos nuevos no contienen rutas a `obs-data` ni comandos de escritura externa.

### Task 2: Promover la plantilla UVG y sus contenidos

**Files:**
- Create/replace at repository root: `z-main.tex`, `0-datos_estudiante.tex`, `1-opciones_adicionales.tex`, `2-paquetes_y_comandos_usuario.tex`, `a-prefacio.tex`–`o-glosario.tex`, `m-bibliografia.bib`
- Create/replace directories: `figuras/`, `plantilla/`
- Modify: `z-main.tex`

**Interfaces:**
- Consumes: archivos equivalentes de `C:/Users/rodri/Downloads/plantilla-uvg-2019-v4`.
- Produces: `z-main.tex` como único punto de entrada principal, con `i-marco_teorico.tex` y `j-capitulos.tex` conectados por `\input`.

- [ ] **Step 1: Copiar los módulos fuente** de la plantilla a la raíz y conservar los nombres que usa `z-main.tex`.
- [ ] **Step 2: Copiar `figuras/` y `plantilla/`** incluyendo `cronograma_gantt.png`, la portada y los PDF de logos preconvertidos.
- [ ] **Step 3: Ajustar `z-main.tex`** para que `\includegraphics` use `plantilla/logoUVGblanco-eps-converted-to.pdf` y `plantilla/escudoUVGnegro-eps-converted-to.pdf` en lugar de disparar conversión EPS automática.
- [ ] **Step 4: Confirmar rutas y entradas** con `rg -n '\\(input|includegraphics|addbibresource|graphicspath)' z-main.tex *.tex`.
- [ ] **Step 5: Verificar contenido académico** comprobando que `i-marco_teorico.tex`, `j-capitulos.tex` y `m-bibliografia.bib` mantienen sus secciones y claves bibliográficas.

### Task 3: Mover el código a `src/` y actualizar referencias

**Files:**
- Move: `codigo/research_pipeline/` → `src/research_pipeline/`
- Move: `codigo/harness_loop/` → `src/harness_loop/`
- Modify: `src/research_pipeline/pipeline.py`
- Modify: `src/research_pipeline/README.md`
- Modify: `src/harness_loop/run_experiment.py`
- Modify: `src/harness_loop/README.md`
- Modify: `src/harness_loop/experiment_manifest.yaml`

**Interfaces:**
- Consumes: comandos existentes del pipeline y del harness.
- Produces: los mismos comandos operativos con rutas `src/...`.

- [ ] **Step 1: Mover ambas carpetas** y eliminar únicamente el directorio `codigo/` si queda vacío.
- [ ] **Step 2: Sustituir rutas** `codigo/harness_loop/...` por `src/harness_loop/...` y `codigo/research_pipeline/...` por `src/research_pipeline/...` en Python, YAML y README.
- [ ] **Step 3: Buscar referencias antiguas** con `rg -n 'codigo[/\\]' src README.md .vscode` y confirmar cero coincidencias activas.
- [ ] **Step 4: Validar sintaxis Python** con `python -m compileall -q src`.
- [ ] **Step 5: Validar el manifiesto** con `python src/research_pipeline/pipeline.py validate --manifest src/harness_loop/experiment_manifest.yaml`.

### Task 4: Archivar LaTeX legado sin pérdida y cerrar límites

**Files:**
- Move: `escrito/` → `reports/archive/latex-legacy/`
- Modify: `.gitignore`
- Create: `reports/archive/latex-legacy/README.md`

**Interfaces:**
- Consumes: fuentes, PDF y auxiliares de `escrito/`.
- Produces: histórico accesible que no compite con `z-main.tex` como punto de entrada.

- [ ] **Step 1: Confirmar que `reports/archive/latex-legacy/` no existe** y listar el contenido de `escrito/` antes del movimiento.
- [ ] **Step 2: Mover `escrito/` completo** sin borrar archivos compilados ni fuentes.
- [ ] **Step 3: Crear el README histórico** indicando las dos variantes archivadas y que el documento vigente está en la raíz.
- [ ] **Step 4: Confirmar** que `escrito/` ya no existe, que el destino contiene el mismo número de archivos y que no se tocaron `obs-data` ni otras carpetas de apoyo.

### Task 5: Compilar y revisar el documento

**Files:**
- Generate ignored: `artifacts/latex/*`
- Generate temporary: `tmp/pdfs/*`
- Verify: `artifacts/latex/z-main.pdf`, `artifacts/latex/z-main.log`, `artifacts/latex/z-main.bcf`

**Interfaces:**
- Consumes: `z-main.tex` y sus entradas en la raíz.
- Produces: PDF compilado con bibliografía resuelta y evidencia visual de las páginas clave.

- [ ] **Step 1: Compilar una primera pasada** con `pdflatex -interaction=nonstopmode -file-line-error -output-directory=artifacts/latex z-main.tex`.
- [ ] **Step 2: Ejecutar `biber`** sobre `artifacts/latex/z-main` y compilar dos pasadas adicionales con `pdflatex`.
- [ ] **Step 3: Revisar el log** con `rg -n '(^!|Error|undefined|Citation|There were)' artifacts/latex/z-main.log` y clasificar cualquier advertencia restante.
- [ ] **Step 4: Comprobar el PDF** con `pdfinfo`/`pdftotext` y confirmar que contiene `Marco teórico`, `Metodología`, `Plan de trabajo` y referencias.
- [ ] **Step 5: Renderizar páginas representativas** con `pdftoppm -png` en `tmp/pdfs/` y revisar portada, índice, marco teórico y cronograma visualmente.

### Task 6: Auditar relaciones y cerrar la migración

**Files:**
- Generate ignored: `graphify-out/*`
- Verify: `src/`, root LaTeX and repository status

**Interfaces:**
- Consumes: árbol migrado y código bajo `src/`.
- Produces: auditoría de relaciones, búsqueda final de rutas obsoletas y reporte de cambios.

- [ ] **Step 1: Ejecutar Graphify** sobre `src/` con la salida de visualización desactivada si el corpus no contiene documentación suficiente.
- [ ] **Step 2: Revisar el reporte** para detectar endpoints faltantes o referencias huérfanas.
- [ ] **Step 3: Buscar rutas obsoletas** con `rg -n '(^|[/\\])(codigo|escrito)([/\\]|$)' --glob '!artifacts/**' --glob '!reports/archive/**'`.
- [ ] **Step 4: Revisar `git status --short`** y confirmar que solo hay cambios dentro de `ProyectoGrad`.
- [ ] **Step 5: Entregar** las rutas del punto de entrada, PDF compilado, histórico archivado y cualquier pendiente académico explícito.

