# Campañas GPU 2026-09-30: Allen–Cahn (contrato DeepXDE) y NS2d-CG (PINNacle)

Estado: **exploratorio, pre-registro sin firmar** (la firma es humana). Aprobación del investigador en el chat del 2026-09-30: pasos 1–5, 12 h restantes, dejar 6 h de reserva. Nada aquí acepta referencias, congela un baseline ni constituye un claim. `evals/`, `registry/` y umbrales no se tocaron.

Código: `pikan_campaign.py` (un solo archivo, PyTorch propio); notebooks generados por `build_notebooks.py` (`bench/`, `ac/`, `ns/`, cada uno con `approval.json` y hashes). Resultados descargados en `outputs/`. Bitácora: `progress.md`.

## Por qué estas dos campañas

- **Allen–Cahn:** el baseline previo era de *tiempo* fijo (600 s → 27.9k–29.3k pasos según semilla), minilote de 1 024 y un MLP de 4×20 (la demo oficial usa 3×20). Aquí se reproduce el contrato de DeepXDE con pasos fijos y lote completo, puntuando solo contra `Allen_Cahn.mat` (referencia de terceros, la que usa DeepXDE), dos endpoints (fin de Adam, fin de L-BFGS).
- **NS2d-CG:** único problema de Navier–Stokes con tabla publicada y referencia FEM de terceros (COMSOL) a la que compararse: PINNacle NS2d-CG (escalón hacia atrás, Re=100), vanilla L2RE 1.19e-1, mejor método publicado 8.24e-2 (LAAF). DFG no tiene comparador PINN publicado.

## Contratos

| | NS2d-CG | Allen–Cahn |
|---|---|---|
| Problema | dominio [0,4]×[0,2] menos [0,2]×[1,2]; ν=0.01; entrada u=4y(1−y), v=0; p=0 en x=4; no-slip en el resto | u_t = 1e-3 u_xx + 5(u−u³), IC x²cos(πx), u(±1,t)=−1 (transformada dura) |
| Puntos | 8 192 dominio + 2 048 frontera, Sobol con scramble | 8 000 dominio + 400 borde + 800 iniciales como puntos de residuo |
| Pérdida | 3 residuos + 5 condiciones, todos peso 1 (como `loss_weights=ones` de PINNacle) | residuo PDE |
| Optimizador | Adam 1e-3, 20 000 pasos, lote completo, float32 | Adam 1e-3, 40 000 pasos lote completo, float32; luego L-BFGS ≤300 s (≤15 000 it.) |
| Brazo control | MLP tanh 5×100, Glorot normal (41 003 parámetros) | MLP tanh 3×20 (921 parámetros) |
| Brazo RBF-PIKAN | ancho **56** (40 992 parámetros, casi igual al MLP), semillas 701–705; ancho 16 (4 032, tamaño del baseline actual), semillas 701–703 | ancho 16 (3 648 parámetros), semillas 601–605 |
| Métrica | L2RE vs `ns_0_obstacle.dat` (12 910 nodos COMSOL): por componente, media de componentes y aplanada | L2 relativo sin pesos vs `Allen_Cahn.mat` 201×101 |

Desviaciones declaradas respecto a los pipelines originales: implementación propia en PyTorch (no DeepXDE/PINNacle), pools Sobol en vez de pseudoaleatorios/Hammersley, y en NS la agregación exacta de L2RE de PINNacle sobre 3 salidas no se pudo verificar en su código; se reportan ambas variantes (media de componentes y aplanada). Ambos brazos comparten pools por semilla.

## Reglas de lectura fijadas antes de ver resultados

1. La comparación principal es **pareada por semilla** (mismos pools) entre brazos; diferencia = RBF − MLP; se reportan las 5 diferencias, media y sd (n=5, poco potente).
2. NS2d-CG: el vanilla (MLP) de esta implementación se compara con 1.19e-1 publicado solo como chequeo de que la receta está bien reproducida. Si el MLP cae lejos de ese valor (fuera de ~×2), se reporta como "reproducción no lograda" y no se saca ninguna conclusión de "competitivo".
3. "Competitivo" en NS2d-CG solo si el RBF, a parámetros iguales, queda por debajo del MLP en las cinco semillas **y** en el mismo orden de magnitud que la mejor cifra publicada; aun así sería `screening`, no claim.
4. Allen–Cahn: dos endpoints por brazo, sin mezclarlos. No se compara con AEH-PINN (8.1e-5: otra red, 100 000 épocas, currículo propio).
5. Corridas incompletas (`adam_time_cap`, `nonfinite_loss`, L-BFGS rechazado) se reportan como tales; nunca como 0.
6. No hay selección de checkpoint ni de hiperparámetros contra la métrica: un solo checkpoint por etapa.

## Presupuesto (medido en el bench de 144 s, `outputs/bench/`)

| Campaña | Estimación de trabajo GPU | Reloj (2 T4) | Tope de sesión |
|---|---|---|---|
| AC | 5×(RBF 13.7 min + MLP 6.2 min + 2×5 min L-BFGS) ≈ 150 min | ≈ 1.3 h | 7 200 s |
| NS2d-CG | 5×RBF56 (~48 min) + 5×MLP (10.8 min) + 3×RBF16 (18.7 min) ≈ 350 min | ≈ 2.9 h | 12 600 s |

Cuota antes de lanzar: 12.35 h restantes de 30 h (renueva 2026-10-03); topes suman 5.5 h; reserva proyectada ≥ 6.85 h.
