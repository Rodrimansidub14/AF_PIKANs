# Migración a la plantilla UVG 2019 — Diseño

**Fecha:** 2026-08-20  
**Repositorio afectado:** `C:/Users/rodri/Documents/ProyectoGrad`  
**Repositorio de referencia:** `obs-data` (solo lectura)

## Objetivo

Convertir la plantilla UVG 2019 ubicada en `plantilla-uvg-2019-v4` en la fuente
principal del trabajo de graduación, conservar el contenido actualizado del
marco teórico y metodología, separar el código reproducible bajo `src/` y
mantener las versiones anteriores como material histórico sin eliminar datos.

## Límites de repositorio

- La raíz contiene el documento LaTeX principal y sus archivos de entrada.
- `src/` contiene código ejecutable y sus manifiestos/documentación operativa.
- `notebooks/`, `presentaciones/`, `PIKAN-Vault/` y `recursos/` son material de
  apoyo y conservan su ubicación actual.
- `reports/archive/latex-legacy/` conserva las versiones antiguas de LaTeX y
  sus PDF/auxiliares como histórico.
- `artifacts/` contiene salidas regenerables; `var/` contiene estado temporal.
- El repositorio `obs-data` no se modifica.

## Estructura LaTeX

Se promueven a la raíz los archivos `z-main.tex`, `0-datos_estudiante.tex`,
`1-opciones_adicionales.tex`, `2-paquetes_y_comandos_usuario.tex`, los módulos
`a-` a `o-`, `m-bibliografia.bib`, `figuras/` y `plantilla/` de la plantilla
UVG. El documento usa `report`, `biblatex` con `biber`, y las rutas se resuelven
relativas a la raíz del repositorio.

El marco teórico de `i-marco_teorico.tex` y la metodología/plan de trabajo de
`j-capitulos.tex` se conservan como contenido vigente. La adaptación no añade
contenido académico nuevo en esta etapa. Los nombres pendientes del asesor y
tribunal permanecen explícitos en `0-datos_estudiante.tex`.

Los logos de portada se referencian mediante los PDF ya convertidos incluidos
en `plantilla/`, evitando que MiKTeX vuelva a generar un PDF EPS vacío. Las
salidas de compilación se dirigen a `artifacts/latex/`.

## Código y rutas

Las carpetas existentes se renombran mediante movimiento, sin borrar su
contenido:

```text
codigo/research_pipeline  -> src/research_pipeline
codigo/harness_loop       -> src/harness_loop
```

Las rutas literales en Python, YAML y README se actualizan a `src/...`. No se
introduce una capa de importación nueva ni se modifica la lógica científica.

## Verificación

La migración se verifica con:

1. búsqueda de referencias antiguas (`codigo/`, `escrito/`) en las fuentes
   activas;
2. compilación `pdflatex → biber → pdflatex ×2` de `z-main.tex`;
3. comprobación de ausencia de errores de citas, referencias y figuras;
4. extracción rápida de texto del PDF y render de páginas representativas para
   revisar portada, índice, marco teórico y plan de trabajo;
5. auditoría de relaciones del código con Graphify sobre `src/`.

