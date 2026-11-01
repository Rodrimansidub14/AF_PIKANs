# EXP-0024 · ronda 1 · nivel T0 (2026-10-20)

Exploratorio. `scientific_accepted=false`. Ajuste supervisado sin PDE (arnés del paso 0 del EXP-0015), presupuesto T0: 600 pasos de Adam, 250 iteraciones de L-BFGS, 3 000 puntos de entrenamiento. Semillas 1–5 (bloque T). CPU, 2 núcleos, ~30 s por ajuste. Datos en `results/t0/`, resumen en `t0_summary.json` (`python3 analyze_t0.py`), cada corrida anotada en `registry/designs.jsonl`.

**Idea A:** reubicar los centros de las capas ocultas con los cuantiles de las activaciones (`regrid.py`; solo usa activaciones, sin pérdida ni referencia) y reajustar los coeficientes por mínimos cuadrados para conservar la función de cada arista (error relativo del reajuste ≈ 1e-3).

| Diseño | Qué es |
|---|---|
| D1 | cuantiles puros, una vez al 25 % de Adam |
| D2 | mezcla 0.5 cuantiles / 0.5 uniforme, una vez al 25 % |
| D3 | cuantiles puros, cuatro veces (10, 30, 50, 70 %) |
| D4 | cuantiles puros, una vez al 5 % |

## Razón del error final frente al control (misma semilla, mismo presupuesto)

| Problema | Diseño | Mediana (tras L-BFGS) | Mejora | Por semilla | Fin de Adam (mediana) |
|---|---|---|---|---|---|
| Allen–Cahn | D1 | **0.68** | 5/5 | 0.64 0.71 0.71 0.68 0.65 | 0.66 |
| Allen–Cahn | D2 | **0.69** | 5/5 | 0.59 0.70 0.64 0.75 0.69 | 0.62 |
| Allen–Cahn | D3 | 0.81 | 4/5 | 0.81 0.92 0.77 1.09 0.68 | 0.76 |
| Allen–Cahn | D4 | 0.81 | 3/5 | 1.05 1.25 0.73 0.75 0.81 | 0.76 |
| Heat | D1 | 1.44 | 1/5 | 0.94 1.96 1.44 1.05 2.67 | 0.45 |
| Heat | D2 | 0.92 | 4/5 | 0.75 0.92 0.93 0.68 1.02 | 0.38 |

El mecanismo hace lo que debe: en un ensayo mínimo, la Gram de la base en la capa oculta pasó de condición 2.7e13 a 4.9e4 y el rango efectivo de 1.4 a 2.6.

## Lectura

- En Allen–Cahn la reubicación única (D1, D2) baja el error de ajuste a ~0.7× del control en 5 de 5 semillas. Hacerla muy temprano (D4) o repetirla (D3) rinde menos y es inestable.
- En Heat (suave, control negativo del proyecto) D1 **empeora** tras L-BFGS (1.44×) aunque en Adam mejoraba (0.45×): reaparece el patrón de EXP-0011/12, donde la ganancia de Adam se pierde con L-BFGS. D2, con mezcla, lo atenúa (0.92×).
- Esto es T0: ajuste supervisado, 5 semillas, 4 diseños probados y uno elegido mirando resultados (maldición del ganador). No sostiene nada; sirve para decidir qué pasa al nivel T1.

## Siguiente paso (propuesto, según la regla 3 del protocolo)

Pasan a T1 (corridas cortas con la pérdida de la PDE, Burgers y Allen–Cahn) **D1, D2 y D3**; D4 se descarta (3/5 y mediana 0.81). T1 requiere el entrenador con pérdida de PDE y el gancho de reubicación; en Burgers (RBF48) es costoso en CPU, así que probablemente va a GPU. No existe aún.

## Pruebas previas del protocolo

1. Arnés T0: el camino de control es **idéntico, bit a bit**, a `fit_step0.fit` (`tests/test_wind_tunnel_t0.py`). No se reprodujo la tabla n = 3 del paso 0 a presupuesto completo (el ajuste de 2 000 + 2 000 pasos no cabe en una llamada de este entorno); queda pendiente.
2. `registry/designs.jsonl`: es un archivo nuevo, no leído por verificadores. La prueba de que `evals/` no lo escribe sigue pendiente.
3. Semillas: bloques disjuntos verificados (`tests/test_wind_tunnel_seeds.py`).

## Registro de incidencias

- Un fallo de código en la rama `mix < 1` (forma de un tensor) abortó el primer intento de D2 antes de producir resultados; se corrigió, se añadió una prueba y se corrió D2 completo. No hay resultado descartado.
- Una corrida de control de medición de tiempos (presupuesto 1 000/500/4 000, semilla 1) quedó en `results/t0/_timing/` y sigue en el libro con la marca `purpose: medicion de tiempos`; no entra en ningún análisis.
