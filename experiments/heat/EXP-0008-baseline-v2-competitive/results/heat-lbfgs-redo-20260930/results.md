# Heat: L-BFGS repetido desde los checkpoints de Adam (2026-09-30)

Kernel `rodrimandub/af-pikans-heat-lbfgs-redo-20260930`: COMPLETE, sesión 1 566 s (0.44 h), 5/5 semillas `rc=0`. Mismos checkpoints `checkpoint-adam-final` (20 000 pasos Adam) y mismas piscinas que S2 (semillas 901–905), float64, lote completo, L-BFGS lr=1, historia 100, Wolfe fuerte, bloques de 25 iteraciones con 125 evaluaciones (arregla el defecto de 32), tope 15 000 iteraciones o 700 s. Puntuado en la malla analítica 512×256. Exploratorio; `scientific_accepted=false`. Datos: `results-summary.json`, `gpu-output/`.

| Etapa | L2 relativo 512×256 (media ± sd, n=5) |
|---|---|
| Final de Adam (20 000 pasos) | `2.197e-4 ± 1.802e-4` |
| + L-BFGS archivado (truncado: 22–73 it. en 4 semillas) | `2.061e-6 ± 2.307e-7` |
| + L-BFGS completo (esta corrida) | **`1.857e-6 ± 1.815e-7`** |

Por semilla (Adam → L-BFGS completo): 901 2.44e-4 → 1.95e-6 (12 600 it., tope de tiempo); 902 1.56e-4 → 1.80e-6; 903 5.14e-5 → 1.63e-6; 904 1.29e-4 → 1.79e-6; 905 5.18e-4 → 2.11e-6 (las cuatro últimas llegan a las 15 000 it.).

Lecturas:
- **El defecto del arnés era real pero casi inocuo**: el L-BFGS completo mejora solo ~10 % sobre el archivado (2.06e-6 → 1.86e-6). Mi expectativa de que estuviera claramente sin optimizar no se cumplió.
- **El error ya está en su piso a las 50 iteraciones** (≈2.1e-6 en 901/902/904; 1.9e-6 en 903; 2.3e-6 en 905) y de ahí en adelante baja <10 % en 15 000 iteraciones. La pérdida llega a ~1e-8 y se aplana: el límite parece ser la discretización con 4 096 puntos de colocación, no la optimización.
- **L-BFGS elimina la dispersión entre semillas**: sd de 1.8e-4 (Adam) a 1.8e-7 (mismo orden que la media de 1.9e-6), y el orden de las semillas casi se pierde (la peor en Adam, 905, sigue siendo la peor, pero por 30 % y no por 10×).
- **Consecuencia para el diseño del control**: una diferencia entre brazos tras L-BFGS solo se podrá ver si supera ~1e-7 (sd) sobre un piso de ~1.9e-6; casi todo el efecto medible de un mecanismo estará en la etapa Adam (2.2e-4 ± 1.8e-4) o en el costo para llegar al piso. Reportar ambas etapas, como se acordó.
- Presupuesto para igualar: L-BFGS fue ~450–700 s de GPU por semilla en un T4 (16 000–25 700 evaluaciones); con 50 iteraciones (~1 s) ya se alcanza el piso.
