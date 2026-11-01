# DFG bajo la configuración de la tesis de Daniel (2026-09-30) — resultados

Kernel `rodrimandub/af-pikans-dfg-daniel-cfg-20260930`: COMPLETE, sesión 2 739 s (0.76 h), 20/20 corridas `rc=0`. Exploratorio; `scientific_accepted=false`. Por semilla: `dfg-daniel-cfg-per-seed.csv`; artefactos en `outputs/dfgd/`.

Configuración (tesis González Carrillo 2024, §5.1.2): DFG Re=20 estacionario, LHS (318 puntos de frontera, 22 345 de colocación), MLP 4×50 tanh (7 953 parámetros) contra RBF-PIKAN ancho 16 (4 032), salidas (u,v,p) crudas, pérdida sin normalizar, Adam 2 000 pasos lr 1e-3 + L-BFGS 1 000 iteraciones lr 0.1 (interpretado como 1 000 pasos de una iteración, sin búsqueda de línea), float32, semillas 801–805 con las mismas piscinas por semilla. Variante `literal`: p=0 en todos los puntos de frontera y sin condición de salida (como la Fig. 19); variante `corrected`: condiciones del DFG (entrada, paredes y cilindro en velocidad; salida libre). Puntuado contra el FEM L3 (referencia de trabajo, malla 221×42, gauge en la salida) y con la fórmula de fuerzas σ·n.

| Variante | Brazo | Etapa | u (L2 rel.) | v | p | error de Cd | error de Δp |
|---|---|---|---|---|---|---|---|
| corregida | MLP 4×50 | fin de Adam | 0.949 ± 0.032 | 1.027 | 0.994 | 100.4 % | 100.5 % |
| corregida | MLP 4×50 | + L-BFGS | 0.883 ± 0.033 | 1.020 | 1.030 | 101.6 % | 101.8 % |
| corregida | RBF-PIKAN 16 | fin de Adam | 0.536 ± 0.016 | 0.776 | 0.744 | 95.2 % | 98.1 % |
| corregida | RBF-PIKAN 16 | + L-BFGS | 0.513 ± 0.011 | 0.668 | 0.716 | 89.2 % | 92.0 % |
| literal | MLP 4×50 | fin de Adam | 0.953 | 1.056 | 1.024 | 100.5 % | 100.7 % |
| literal | MLP 4×50 | + L-BFGS | 0.874 | 1.006 | 1.006 | 101.5 % | 101.7 % |
| literal | RBF-PIKAN 16 | fin de Adam | 0.571 | 0.787 | 0.877 | 95.4 % | 98.2 % |
| literal | RBF-PIKAN 16 | + L-BFGS | 0.559 | 0.677 | 0.864 | 89.2 % | 92.0 % |

Diferencia pareada RBF − MLP en u (5 semillas, todas negativas): corregida −0.41 (Adam) y −0.37 (L-BFGS); literal −0.38 y −0.32. En error de Cd: −0.05 (Adam) y −0.12 (L-BFGS), 5/5 a favor del RBF.

## Lectura

- **Con este presupuesto la configuración no resuelve el problema**: el MLP 4×50 queda en error de ~90–100 % en las tres salidas y en Cd y Δp ≈ 0 (la solución trivial), y el RBF en ~50–55 % en u y ~90 % en Cd. Comparar: nuestras cohortes de 14 000–20 000 pasos llegan a 7–9 % (u) y 5–8 % (Cd) tras L-BFGS. La tesis no reporta ningún error cuantitativo para este caso (solo figuras), así que no hay un número contra el que reproducir.
- **El RBF gana en las 5 semillas y en ambas variantes**, con error en u ≈ 0.4 menor; pero ambos brazos están lejos de una solución, así que esto mide velocidad inicial de aprendizaje con ~3 000 pasos, no calidad.
- **La variante literal y la corregida casi no se distinguen** (u 0.536 vs 0.571 en RBF; igual en MLP): a 2 000 pasos el modelo aún no alcanzó la física donde importa la condición de salida o el p=0. La diferencia esperada de la condición p=0 en el cilindro no es visible todavía.
- **Límites de esta lectura**: mi interpretación del L-BFGS de la tesis (1 000 iteraciones, lr 0.1, sin búsqueda de línea), float32, inicialización Glorot y coordenadas crudas en el MLP pueden diferir de lo que hizo Daniel; la tesis no lo detalla. Puede que sus figuras cualitativas correspondan a un entrenamiento efectivo distinto. Por eso esto es una reproducción de la configuración *tal como está escrita*, no una reproducción de sus resultados.
- Costo por corrida (un T4): MLP ≈ 75 s Adam + 48 s L-BFGS; RBF ≈ 240–250 s + 130–136 s.
