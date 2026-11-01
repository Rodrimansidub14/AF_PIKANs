# Prueba reproducible de Fiziko para el bimorfo

Esta carpeta contiene una prueba local de la biblioteca MetaPost **Fiziko**
para una sección esquemática de un bimorfo piezoeléctrico. La fuente de la
figura es [`bimorph_fiziko.mp`](bimorph_fiziko.mp); los productos principales
son [`bimorph_fiziko.pdf`](bimorph_fiziko.pdf) y
[`bimorph_fiziko.png`](bimorph_fiziko.png).

## Procedencia y licencia

- Repositorio leído en modo externo de solo lectura:
  <https://github.com/jemmybutton/fiziko.git>
- Rama descargada: `master`.
- Commit observado: `54a63dba8e6700a5e70d3508838edebcbf0f45fe`.
- La biblioteca declara GNU GPL v3 en `upstream/LICENSE.md`.
- El manual `upstream/fiziko.tex` declara CC-BY-SA 4.0 para el documento;
  se conservan ambos archivos para que la integración mantenga la atribución
  y las condiciones correctas.

## Qué se comprobó

Fiziko expone `tube.l(path)(offset)` para tubos de ancho variable con sombreado
longitudinal; también documenta `tube.t`, `tube.s`, `sphere.c`, `sphere.s`,
`flatSurface` y objetos compuestos. La prueba usa únicamente `tube.l` porque
produce una textura de capa continua y reproducible en este esquema.

La figura mantiene dos capas rectas adheridas, un empotramiento a la izquierda,
electrodos con `V=100 V` y `V=0 V`, y flechas de campo con `E_y<0`. La escala
física rotulada es `L=0.1 m` y `H=1 mm`; la altura está dibujada con una
exageración indicada como «altura exagerada, sin escala» para que la sección sea legible. No se dibuja
una deformación ni se afirma el signo de una eigenstrain: esta lámina es un
esquema de geometría, condiciones eléctricas e interfaz.

La página de referencia de OpenAI, [On the Navier–Stokes Millennium Prize
Problem](https://openai.com/index/navier-stokes-solution/), describe su imagen
como una instantánea de movimiento incompresible donde el naranja marca mayor
rotación angular, el teal menor rotación, y las trayectorias muestran espiral
interior y estiramiento axial. La paleta naranja/teal de esta prueba toma esa
convención cromática observada; la geometría del bimorfo es propia y no se
presenta como una reproducción de la figura de Navier–Stokes.
La fuente se consultó como contenido textual; por eso no se atribuyen a la
referencia un fondo, una tipografía o una composición visual que su texto no
documenta.

## Reproducción local

Desde esta carpeta, con MiKTeX 24.1 y Poppler disponibles:

```powershell
mpost -interaction=nonstopmode -halt-on-error bimorph_fiziko.mp
epstopdf --outfile=bimorph_fiziko.pdf bimorph_fiziko-1.eps
pdftocairo -png -singlefile -r 180 bimorph_fiziko.pdf bimorph_fiziko
```

La primera orden produjo un EPS de una página; las dos órdenes siguientes
produjeron el PDF y el PNG usados para inspección visual. `epstopdf` mostró
avisos de mapa de fuentes de MiKTeX, pero completó el PDF y el PNG; el texto
de las etiquetas se ve en la inspección visual.

## Integración sugerida

Para el informe puede insertarse `bimorph_fiziko.pdf` como esquema de
condiciones de contorno y citarse el repositorio Fiziko con su commit. Conviene
mantener `bimorph_fiziko.mp`, `upstream/LICENSE.md` y este README junto con el
artefacto generado. La advertencia del manual de Fiziko —sus algoritmos pueden
dar resultados decentes solo en circunstancias simples— es relevante si se
intenta reutilizar `tube.l` para una curvatura física cuantitativa; esta prueba
se mantiene deliberadamente recta para no confundir textura ilustrativa con
una simulación.
