# Baseline v1 (exploratorio) — CONGELADO 2026-10-03

Estado: **CONGELADO el 2026-10-03** (no se edita; si cambia, se congela otro id y se anota la supersesión). Gate `eval_integrity`. Todo exploratorio (`scientific_accepted=false`). Un agente no escribe en `evals/baselines/`; al aprobarla, el investigador (o con su OK explícito) la copia allí. Los números se toman de `scorecard-current.csv` y de los resultados del 2026-09-30/10-01. Un baseline congelado no se edita: si cambia, se congela otro id y se anota la supersesión.

Regla de lectura: dos etapas por problema cuando aplica (fin de Adam / tras L-BFGS), nunca mezcladas; n=5 semillas salvo nota; media ± sd muestral.

| Problema | Control (brazo RBF-PIKAN) | Etapa 1: fin de Adam | Etapa 2: tras L-BFGS | Referencia | Comparador de literatura / paralelo | Notas |
|---|---|---|---|---|---|---|
| Heat 1D | RBF-PIKAN, semillas 901–905 | 2.197e-4 ± 1.80e-4 | **1.857e-6 ± 1.8e-7** (L-BFGS completo, `heat-lbfgs-redo-20260930`) | analítica 512×256 | DeepXDE α=0.1: 5.81e-4; B-spline S2 (seguimiento): 1.416e-6 | Piso ~1.9e-6 por discretización (4 096 pts); el efecto de un mecanismo se mide en Adam. El valor archivado 2.061e-6 queda supersedido (L-BFGS truncado) |
| Burgers 1D | RBF-PIKAN 48 (29 376 par.) vs MLP 5×100 (40 801), semillas 301–305, f32; Cole–Hopf 1025×129 | RBF48 1.435e-2 ± 9.4e-3; MLP 1.446e-2 ± 3.8e-3; RBF16 2.38e-2 ± 4.3e-2 | **RBF48 3.87e-4 ± 2.2e-4**; MLP 5.66e-4 ± 5.7e-4; RBF16 1.87e-3 ± 2.4e-3 | Cole–Hopf (Gauss–Hermite 128), evaluador del verificador | PINNacle publicado 1.45e-2 (Adam) / 1.33e-2 (+L-BFGS, FEM del autor); MLP de aquí reproduce 1.45e-2 al final de Adam | Control del baseline = RBF48. Diferencia pareada RBF48 − MLP tras L-BFGS −1.8e-4 ± 6.8e-4 (RBF gana 2/5): **empate dentro del ruido**, no ventaja. RBF16 es peor (3.3× el MLP). Reemplaza la fila anterior (0.0395 ± 0.0256, solo Adam, otro contrato) |
| Allen–Cahn 1D | RBF16 vs MLP 3×20, semillas 601–605 | RBF 2.84e-2 ± 7.4e-3; MLP 1.947e-1 ± 9.9e-2 | **RBF 2.686e-3 ± 9.5e-4**; MLP 1.773e-2 ± 8.7e-3 | `Allen_Cahn.mat` (DeepXDE, aceptado 2026-10-01) | DeepXDE oficial 2.00e-2 ± 6.1e-3 | RBF gana 5/5. Las filas contra FD8192/ETDRK4 son candidate-only y no cuentan |
| NS2d-CG (PINNacle) | RBF56 (40 992 par.), RBF16, MLP 5×100; semillas 701–705 (RBF16: 3) | **RBF56 0.0655 ± 0.0058**; RBF16 0.0667 ± 0.0023; MLP 0.0977 ± 0.0067 | no corrido | COMSOL `ns_0_obstacle.dat` (sha256 13e1d9fa…) | PINNacle vanilla 0.119, LAAF 0.0824 | L2RE plano (la agregación de PINNacle no es verificable); RBF16 n=3 |
| DFG Re=20 (cohortes previas) | RBF-PIKAN 301–305 | — | Cd 5.124 ± 0.053 (**error 8.16 %**), Δp error 9.89 % | FEATFLOW QoI (Cd 5.5795); campo: FEM L3 (referencia de trabajo) | MLP 5×64 (501–505): Cd error 5.3 % | σ·n corregido el 2026-09-30. Campo u,v 7–9 % |
| DFG Re=20 (receta larga, tesis de Daniel) | RBF16 vs MLP 4×50, semillas 801–805 | RBF u 0.461; MLP 0.671 | RBF u 0.191 ± 0.039 (Cd 34.7 %); MLP 0.421 (Cd 69.5 %) | FEM L3 | — | NO convergido (L-BFGS en tope de tiempo); informativo, no es el control del DFG |

## Qué se congela
1. La columna "Etapa" por problema como la única lectura admitida del control.
2. Las referencias listadas (con sus hashes en cada ficha); ninguna generada por agente salvo FEM L3, aceptado como referencia de trabajo.
3. La etiqueta `exploratory`: v1 no sostiene claims.

## Qué NO se congela
- Heat B-spline y toda fila `candidate-only`.
- Umbrales de gates: se derivan después, con un baseline firmado.

## Firma
- [x] Aprobado y congelado por el investigador — fecha: 2026-10-03 — nombre: Rodrigo Mansilla (mensaje de chat "firmo v1"). Copiado a `evals/baselines/` con su OK explícito.
