# EXP-0009 — piezo indirecto: guía de lectura

Pre-registro en `spec.md` y cierre en `decision.md`.

## Qué se extrajo de la receta de Daniel y cómo se configuró

El código de Daniel (PINNs_piezoelectricity, González Carrillo) **no se redistribuye aquí**. De él se extrajeron solo los parámetros necesarios para reproducir la receta, que están en [`PARAMETROS-DANIEL.md`](PARAMETROS-DANIEL.md):

- Procedencia, commit fijado `e7856f1` y copia en un fork.
- Geometría, escalas de referencia y constantes del material PVDF.
- Receta de entrenamiento: 20 000 épocas de Adam, tasa 1e-3, doble precisión, muestreo por capas y remuestreo cada 200 épocas.
- Configuración de cada corrida: brazo (`mlp` o `rbf` de ancho 16), parámetros, semillas y sesión.

Para volver a correrlo, `daniel_arm.py` importa el código desde un clon del fork en ese commit (instrucciones en `PARAMETROS-DANIEL.md`). Los resultados y su configuración (`results/daniel-*/results/*/run.json`) sí están versionados.

La licencia del repositorio original no está confirmada. Por eso las copias de su código que había en `results/daniel-*/payload/daniel/` se retiraron del árbol.

## Código propio de esta carpeta

`daniel_arm.py` (adaptador que ejecuta su receta con un tronco MLP o RBF), `piezo_daniel.py` (reimplementación propia de sus ecuaciones y receta), `piezo_pinn.py` y `piezo_layered.py` (reimplementación propia, diagnóstico negativo), `summarize.py`.
