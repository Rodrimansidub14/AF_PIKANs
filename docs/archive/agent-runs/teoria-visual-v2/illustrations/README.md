# Ilustraciones Fiziko para el marco teórico

Esta carpeta contiene ocho ilustraciones reproducibles: los problemas P1--P5, una comparación MLP/KAN y dos apoyos introductorios (varilla térmica y pérdida PINN). Cada fuente .mp importa la copia local de Fiziko en ../../teoria-visual/fiziko/upstream/fiziko.mp; no se escribe ni instala nada fuera de esta carpeta.

Fiziko se usa realmente para los tubos de ancho variable, las esferas sombreadas y el hachurado fino. El resultado mantiene tinta negra y grises de libro físico. Las cinco figuras de problemas son geometría, condiciones de frontera o estado conceptual; no contienen campos numéricos resueltos. P2 es una instantánea espacial: no es un eje tiempo ni una trayectoria de solución. P4 conserva solo la entrada prescrita, el cilindro, las paredes y la salida.

## Referencias observadas

- [Fiziko en GitHub](https://github.com/jemmybutton/fiziko) se describe como una biblioteca MetaPost para ilustraciones físicas en blanco y negro, con líneas de ancho variable, esferas sombreadas y tubos.
- [La página de Navier--Stokes de OpenAI](https://openai.com/index/navier-stokes-solution/) describe un snapshot de movimiento incompresible local con marcas de rotación rápida/lenta y trayectorias espirales con estiramiento axial. Esa descripción se registra para trazabilidad; las figuras de esta carpeta conservan el encargo B/N y no importan colores ni una solución de ese texto.

## Reproducción

Desde var/agent_runs/teoria-visual-v2/illustrations/, con MetaPost, epstopdf y pdftocairo disponibles:

~~~powershell
$figures = @('p1-problem','p2-problem','p3-problem','p4-problem','p5-problem','architecture','heat-rod','pinn-loss')
foreach ($stem in $figures) {
  mpost -interaction=nonstopmode -halt-on-error "$stem.mp"
  epstopdf --outfile="$stem.pdf" "$stem-1.eps"
  pdftocairo -png -singlefile -r 160 "$stem.pdf" "$stem"
}
~~~

La prueba se ejecutó con MiKTeX MetaPost 2.02 (MiKTeX 24.1). La conversión puede mostrar advertencias de mapa de fuentes ICXWUR de MiKTeX; en esta ejecución cada comando terminó con código 0 y produjo el PDF/PNG indicado en manifest.json.
