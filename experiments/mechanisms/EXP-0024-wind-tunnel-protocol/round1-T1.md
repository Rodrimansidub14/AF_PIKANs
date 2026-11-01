# EXP-0024 · ronda 1 · nivel T1 — resultados (2026-10-20)

Exploratorio (`scientific_accepted=false`). Bloque en Kaggle T4×2 (`rodrimandub/af-pikans-exp-0024-t1-block`): 40 de 40 corridas, 36 min (estimado 0.7 h, tope 5 h). Diseños D1, D2, D3 frente al control CTRL de la misma sesión, semillas 1–5, convención T1 (Adam 10 % del control, L-BFGS 500 iteraciones, pérdida de la PDE, sin referencia en el entrenamiento). Las 40 corridas están en `registry/designs.jsonl`. Datos en `results/t1-block/`.

## Razón de error final (tras L-BFGS) del diseño / control, por semilla

| Problema | Diseño | Razón por semilla (1–5) | Mediana | Mejora | Fin de Adam (mediana) | Reloj (mediana) |
|---|---|---|---|---|---|---|
| Burgers | D1 | 21.17, 8.85, 16.72, 44.57, 146.29 | **21.168** | 0/5 | 1.80 | 1.00 |
| Burgers | D2 | 3.01, 4.52, 11.16, 7.44, 1.23 | **4.518** | 0/5 | 0.93 | 1.01 |
| Burgers | D3 | 11.51, 9.21, 12.29, 3.41, 15.47 | **11.514** | 0/5 | 1.41 | 1.01 |
| Allen–Cahn | D1 | 0.72, 0.62, 0.87, 0.70, 0.31 | **0.695** | 5/5 | 0.84 | 1.02 |
| Allen–Cahn | D2 | 0.71, 0.66, 0.75, 0.67, 0.45 | **0.675** | 5/5 | 0.87 | 1.03 |
| Allen–Cahn | D3 | 0.92, 0.72, 0.87, 0.76, 0.45 | **0.765** | 5/5 | 0.91 | 1.01 |

Controles (error final): Burgers 0.0019, 0.0127, 0.0024, 0.0016, 0.0012; Allen–Cahn 0.0517, 0.0534, 0.0481, 0.0514, 0.0495.

## Criterio de la regla 3 (firmada)

Pasa a V un diseño con **mediana ≤ 0.90 en Burgers y en Allen–Cahn** en T1.

- **Allen–Cahn:** D1 (0.695), D2 (0.675) y D3 (0.765) cumplen, con mejora en 5/5 semillas cada uno. Es coherente con T0 (D1 y D2 ≈ 0.68×).
- **Burgers: ninguno cumple, y por mucho.** La reubicación empeora el error final de las 15 corridas (0/5 en cada diseño): medianas 21.2× (D1), 4.5× (D2) y 11.5× (D3). El control de Burgers tampoco es estable a este presupuesto (error final 0.0012–0.0127).
- **Ningún diseño pasa a V.** Ninguno cumple en los dos problemas.
- Guardia de reloj (≤ 1.2× el control): se cumple (mediana 1.00–1.03 en los tres diseños y los dos problemas).

## Lectura

- La reubicación de centros ocultos por cuantiles **es problema-dependiente**: ayuda de forma consistente en Allen–Cahn (≈ 23–33 % menos error) y daña de forma consistente y grande en Burgers. Un mecanismo con ese perfil no cumple el objetivo de ser general.
- Esto es coherente con lo observado en la prueba corta (Burgers D3: 0.0218 frente a 0.0019 del control) y con el diagnóstico de sensibilidad (D1 amplifica perturbaciones diminutas en Burgers; la reubicación en sí es numéricamente inocente: CPU–GPU 4.2e-11). Una hipótesis de trabajo, **no probada aquí**, es que reubicar en Burgers mueve los centros hacia la zona del frente y deja a las capas desajustadas justo cuando el optimizador ya no tiene tiempo de reajustarlas en una corrida corta.
- D2 (mezcla 0.5) daña menos que D1 y D3 en Burgers (4.5× frente a 21× y 11.5×): la intensidad del movimiento importa.

## Límites

- T1 es una corrida corta (Adam 10 %, L-BFGS 500 iteraciones): un efecto de Burgers que se disipe con presupuesto completo no se ve aquí.
- n = 5 por celda; el control de Burgers varía más de 10× entre semillas.
- Una sola idea (A, reubicación de centros ocultos) y tres variantes.

## Opciones (decide el investigador)

1. **Cerrar la idea A** como específica de Allen–Cahn (resultado T1 negativo en Burgers) y pasar a otra idea de la ronda 1 (variantes de precondicionamiento).
2. **Una ronda de variantes de A pensadas para Burgers** antes de cerrarla: reubicar con menor intensidad (mezcla 0.25), solo la capa 2, o con más pasos de Adam tras reubicar. Cuesta ≈ 40 min de T4 por bloque.
3. **Probar A con presupuesto completo en Burgers** (un bloque más largo) para ver si el daño de T1 es solo transitorio.
