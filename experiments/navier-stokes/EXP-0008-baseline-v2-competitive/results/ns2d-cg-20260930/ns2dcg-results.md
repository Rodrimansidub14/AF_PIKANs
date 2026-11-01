# NS2d-CG (PINNacle, backward step Re=100) — resultados (2026-09-30)

Kernel `rodrimandub/af-pikans-ns2d-cg-20260930`: COMPLETE, sesión 11 388 s (3.16 h), 13/13 corridas con `return_code=0`, todas con los 20 000 pasos de Adam completos. Exploratorio; `scientific_accepted=false`. Por semilla: `ns2dcg-per-seed.csv`; artefactos en `outputs/ns/`.

Contrato: PINNacle NS2d-CG (8 192 dominio + 2 048 frontera, pesos 1, Adam 1e-3, 20 000 pasos, lote completo, float32), referencia COMSOL `ns_0_obstacle.dat` (12 910 nodos). Semillas 701–705 (RBF ancho 16 solo 701–703); pools compartidos por semilla.

## L2RE contra COMSOL (media ± sd)

| Brazo | Parámetros | n | **Aplanado** (todas las salidas) | Media de componentes | u | v | p |
|---|---|---|---|---|---|---|---|
| MLP tanh 5×100 (vanilla) | 41 003 | 5 | **0.0977 ± 0.0067** | 0.332 ± 0.007 | 0.072 | 0.510 | 0.414 |
| RBF-PIKAN ancho 56 | 40 992 | 5 | **0.0655 ± 0.0058** | 0.246 ± 0.024 | 0.042 | 0.383 | 0.312 |
| RBF-PIKAN ancho 16 | 4 032 | 3 | **0.0667 ± 0.0023** | 0.250 ± 0.009 | 0.041 | 0.374 | 0.333 |

Referencia publicada (PINNacle, tabla principal según su página HTML): vanilla `1.19e-1`, mejor método (LAAF) `8.24e-2`.

Diferencia pareada por semilla, L2RE aplanado, RBF − MLP:
- RBF ancho 56 (701–705): −0.028, −0.037, −0.035, −0.036, −0.025 → media −0.0322 ± 0.0053 (5/5 a favor del RBF).
- RBF ancho 16 (701–703): −0.032, −0.030, −0.035 → media −0.0322 ± 0.0023 (3/3).
Con la media de componentes el signo es el mismo en todas las semillas (ancho 56: −0.087 ± 0.021).

## Aplicación de las reglas fijadas antes de ver resultados

1. **Reproducción del vanilla:** el MLP da `0.098` en L2RE aplanado frente a `0.119` publicado (dentro de ×2, −18 %): **reproducido bajo la lectura "aplanada"**. Bajo la lectura "media de componentes" da `0.33` (×2.8 del publicado): **no reproducido**. No pude verificar en el código de PINNacle cuál de las dos agregaciones usa su tabla; la coincidencia con la aplanada es un indicio, no una prueba. Las conclusiones de abajo valen solo bajo esa lectura.
2. **Pareado:** RBF menor error que el MLP en todas las semillas (regla 3, primera parte: cumplida).
3. **Orden de magnitud frente a lo publicado:** el RBF (0.066) queda por debajo del mejor método publicado (0.082) y a ~0.55× el vanilla publicado. Pero nuestro propio vanilla ya queda 18 % por debajo del publicado, así que el margen real frente a LAAF es pequeño y sujeto a diferencias de pipeline (Sobol vs pseudoaleatorio, implementación propia, agregación de la métrica). **Etiqueta: `screening`, no claim ni "SOTA".**

## Otras lecturas

- **El tamaño casi no importa:** ancho 16 (4 032 parámetros) `0.0667` vs ancho 56 (40 992) `0.0655`; la ventaja sobre el MLP viene de la familia RBF, no de tener más parámetros. Eso también significa que igualar parámetros con el MLP no compró nada, y que el ancho 16 cuesta 2.5× menos tiempo.
- **Coste:** por corrida, MLP ≈ 690 s, RBF ancho 16 ≈ 1 150–1 250 s, RBF ancho 56 ≈ 2 920 s. A igual tiempo de reloj el MLP hace 4× más pasos que el RBF-56; aquí se comparó a igual número de pasos, no a igual tiempo.
- **Errores por componente enormes en todos los brazos:** v `0.37–0.51` y p `0.31–0.41` relativos; la métrica aplanada está dominada por u, que es la componente grande. Que la pérdida de entrenamiento baje a ~1e-4 con error de campo alto indica que el problema es la calidad de la solución, no el residuo: coherente con los modos de fallo de PINN ya descritos (balance de pérdidas, condición p=0 de salida, esquina reentrante). Nada de esto se probó aquí.
- No hay L-BFGS en esta campaña (el contrato de PINNacle vanilla es solo Adam).
