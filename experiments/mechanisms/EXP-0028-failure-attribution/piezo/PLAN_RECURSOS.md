# Piezo: plan de recursos (2026-10-25), alineado con el paper. NO LANZADO

`predictions.md` está firmado. Nada de piezo se corre hasta que el investigador apruebe la etapa 0.

## Lo medido y lo estimado
- Medido (EXP-0009, T4 float64, RBF16 converso): 0.47 s/paso, 9 400 s por corrida de 20 000 pasos. CPU: 3.9 s/paso (22 h por corrida: inviable).
- Estimado y sin medir: directo (0.31 s/paso), recálculo del bloque (≈ 45 s en T4), brazo I (bloque de tamaño 2), cualquier otra GPU.
- Cuota de Kaggle: 8.2 h de sesión hasta el 31/oct. Colab: solo T4 y TPU desbloqueadas (A100/H100 bloqueadas).

## Por qué cuesta tanto y dónde se ahorra
1. **float64 en T4**: la T4 hace doble precisión a 1/32 de su velocidad de precisión simple. El paper exige float64, así que no se baja la precisión.
2. **Redes pequeñas**: RBF16 (4-16-16-8) sobre ~4 000 puntos puede estar limitada por lanzamientos de kernels y no por cálculo. Si es así, una GPU más potente ayuda poco y lo que ayuda es correr varios procesos por GPU. Se decide midiendo.
3. **P100: descartada.** El selector de aceleradores de Kaggle del investigador (2026-10-25) solo ofrece None, GPU T4 ×2 y TPU v5e-8. La TPU no sirve para float64. No hay GPU con buena doble precisión en Kaggle.

## Recortes que no tocan la alineación con el paper
| Recorte | Ahorro | Costo |
|---|---|---|
| Quitar el brazo I de la primera ronda (sin predicción, el más lento y el único con riesgo de segfault) | ~20 % | I queda exploratorio al final si sobra cuota |
| Brazos B, F, E, J (suficientes para H-PRED-1) | | |
| Semillas en orden de etapas: 20260728 (la del paper), luego 20260729-30, luego 20260731-32 | hasta 60 % si una etapa basta | `predictions.md` evalúa con la mediana de las semillas completadas; se declara n |
| MLP-paper (ancla del arnés) al final | 1.9 GPU-h | |
| Procesos concurrentes por GPU (si está limitada por lanzamientos) | 1.5 a 2.5× | se mide |
| Receta, pesos, entradas, bordes, evaluación | 0 | sin cambios: todo igual al paper |

Costo por corrida (T4 float64, un proceso): converso B 2.6 h, F 2.8 h, J y E 3.0 h; directo B 1.75 h, F 1.9 h, J y E 2.1 h.

## Etapas con puertas de decisión
**Etapa 0, medición (≈ 0.5 h de cuota).** Kernel de tiempos de unos 300 pasos de B, E y J, converso y directo, en (a) 1, 2 y 3 procesos por GPU en T4, con utilización de GPU (`nvidia-smi`) para saber si el límite es el lanzamiento de kernels, y (b) el mismo paso de entrenamiento capturado con CUDA graphs (`Adam(capturable=True)`, buffers estáticos) si (a) muestra poca utilización. Mide s/paso, recálculo y memoria, y proyecta el costo de cada etapa siguiente. Puerta: se elige la GPU y el nº de procesos con el menor costo proyectado.

**Etapa 1, piloto con la semilla del paper.** B, F, E, J, converso y directo, semilla 20260728 (8 corridas; 19 h de T4 en un proceso sin optimizar). Puerta: si no cabe en la cuota con la configuración elegida, se reduce a B y E primero (9.5 h de T4) y el resto va después.

**Etapa 2, semillas 20260729 y 20260730 (n = 3).** Puerta: si F y E quedan los dos dentro de una misma banda de `predictions.md` con 3 de 3, el veredicto se declara con n = 3 y no se sigue; si no, etapa 3.

**Etapa 3, semillas 20260731 y 20260732 (n = 5), luego I y MLP-paper** si queda cuota.

## Qué va a otra plataforma para guardar cuota de Kaggle para piezo
Lo que es float32 y corre bien en T4: Burgers de las fases 1d y 1e (`bu2`, `bu3`, ~10.6 h), DFG de la fase 1e (~8.4 h) y la rebanada con artefactos. Se pueden correr en Colab (T4 gratuita, con interrupciones), reutilizando los scripts de los kernels de Kaggle con las variables `BUNDLE_DIR` y `OUT_DIR`. Es trabajo diferido y no condiciona piezo.

## Riesgos
- Si el límite es el cálculo en float64, solo ayuda una A100/H100 (Colab de pago o alquiler). Si es el lanzamiento de kernels, ayudan los procesos concurrentes, CUDA graphs o entrenar varias semillas en un solo proceso (vmap).
- Brazo I en GPU: no verificado (segfault de torch en CPU con bloques de 4 o más).
- Con n < 5, el criterio (ii) de H-PRED-1 se evalúa con menos semillas; se declara.
