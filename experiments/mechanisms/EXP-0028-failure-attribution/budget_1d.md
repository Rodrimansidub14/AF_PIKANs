# EXP-0028 Fase 1d — presupuesto y receta (2026-10-24). NO LANZADO

Enmienda v1.2: confirmar con receta COMPLETA de la línea base los efectos de la fase 1. Código nuevo: `run_phase1d.py` (importa `run_phase1.py` sin modificarlo), `kaggle_block_1d.py`, `kaggle/build_bundle_1d.py`, `analyze_phase1d.py`. NS2d-CG queda fuera de este presupuesto (`run_phase1c.py --recipe full`).

## Receta exacta por problema (qué se corre y de dónde sale)

| | Allen-Cahn d=1e-3 | Burgers RBF48 | Fisher-KPP |
|---|---|---|---|
| Fuente | baseline v1/v2 AC: EXP-0008 `ac-rerun-20260930` (semillas 601–605; 2.686e-3) | baseline v2 Burgers, fila «grande»: EXP-0008 `burgers-big-20261003` (301–305; 7.165e-5) | **sin línea base v1/v2**: hereda la de AC (ronda 1-C: «receta heredada», no firmada) |
| Red | RBF16, 11 bases | RBF48, 11 bases | RBF16, 11 bases |
| Puntos | 8 000 + 400 + 800 Sobol (los de `pools`) | **16 384** Sobol (`n_col`; la fase 1 usó 8 192) | los de AC |
| Adam | lr 1e-3, **40 000** pasos, float32, lote completo | lr 1e-3, **24 000** pasos, float32 | **40 000** (= 10 × 4 000, el control «completo» de la convención T1) |
| L-BFGS | strong-Wolfe (history 100, tol_grad 1e-8), chunk 50, **float32**, tope **300 s** / 15 000 it., parada por meseta | idem pero **float64** (modelo y pool convertidos tras Adam), tope **1 800 s** / 15 000 it. | como AC |
| Fase 1 (para contrastar) | Adam 4 000, L-BFGS 500 it. | Adam 2 000, L-BFGS 500 it., 8 192 pts | Adam 4 000, L-BFGS 500 it. |

Justificación: es la receta de la línea base que ya es comparable con la literatura (spec v1.2); el lazo de L-BFGS (`sw.lbfgs_stage`) es el mismo del worker de `pikan_campaign.py` (mismos `LBFGS_PARAMS`, mismo criterio de meseta/rechazo), verificado línea a línea. Se conserva lo firmado en el mecanismo: recálculo cada 1 000 pasos de Adam y al arrancar L-BFGS, N = 2 048, ε = 1e-3 (41 recálculos en AC/KPP, 25 en Burgers frente a 5/3 en la fase 1). En Burgers el recálculo de arranque de L-BFGS se hace **después** de pasar a float64 (el GN siempre se calcula en float64); el error de fin de Adam se puntúa en float32 antes del cambio, como la línea base. C (regrid) se escala solo: 25 % de Adam = paso 10 000 (AC).

Qué no soporta el código / límites declarados: nada falta para AC, Burgers y KPP (B, F, E, C pasan por el mismo camino de la fase 1). (1) KPP no tiene línea base propia: su «receta completa» es una herencia, no una congelación; se declara en el análisis. (2) El tope de L-BFGS de Burgers es de **tiempo** (1 800 s): las iteraciones alcanzadas dependen de la GPU y del brazo (E es algo más lenta por iteración); es la receta de la línea base y se reporta `lbfgs_iterations`. (3) Puntos y semillas distintos de la línea base (4101–4105 frente a 301/601): se espera reproducir el orden de magnitud de B, no los dígitos; el análisis imprime B frente al congelado como control de sanidad, sin umbral.

## Brazos y semillas
AC: B + **E, F, C** (fase 1: E 0.162 sirve, F 0.352 sirve, C 0.783 sirve; R 0.793 queda fuera por pedido). Burgers: B + **F** (0.285 sirve) y **E** (0.240, «dependiente»: max 2.37). KPP: B + **E** (0.767 dependiente) y **F** (1.835 dependiente; control del daño: J 2.38 «daña» y S/R/C «dañan», pero no hay J en 1d). Semillas **4101–4105**: verificado por búsqueda en `experiments/`, `registry/`, `src/`, `docs/`, `tests/`, `evals/` (solo coincidencias numéricas ajenas a semillas: pérdidas, iteraciones) y por los JSON de la fase 1 (solo 4001–4010); `run_phase1d.py` rechaza semillas < 4101 salvo humo. 15 + 15 + 20 = 50 corridas.

## Tiempos (T4, medidos en la fase 1 y en la línea base completa; la CPU de esta máquina estaba saturada y no sirve para estimar)

| | s/paso Adam (T4, fase 1) | 1d: Adam | L-BFGS | corrida (con arranque/calentamiento ≈ 25 s) |
|---|---|---|---|---|
| AC B / F / E / C | 0.0196 / 0.0203 / 0.0213 / 0.0198 | 784 / 812 / 852 / 792 s | meseta, 86–127 s en la línea base (esp. 110; peor caso 300) | 920 / 950 / 990 / 930 s |
| KPP B / F / E | 0.0204 / 0.0212 / 0.0223 | 816 / 848 / 892 s | esp. 110 (sin medir a receta completa; peor caso 300) | 950 / 985 / 1 030 s |
| Burgers B / F / E | 0.0782 (medido a 16 384 pts en la línea base) / igual / +24 recálculos × 3.5 s | 1 878 / 1 878 / 1 963 s | 1 800 s (tope de tiempo) + recálculo f64 de E (≤ 35 s) | 3 710 / 3 710 / 3 825 s |

(El s/paso de E incluye el recálculo amortizado al mismo ritmo de 1 000 pasos, por lo que escala linealmente con los pasos.) Totales: AC ≈ 18 950 GPU-s (peor caso 22 750), KPP ≈ 14 810 (17 660), Burgers ≈ 56 230; **≈ 90 000 GPU-s = 25 GPU-h → 12.5 h de pared con 2 ranuras si el empaque fuera perfecto.** No cabe en un lanzamiento (tope 3.9 h).

## Partición en kernels (un proceso por corrida, 2 ranuras, despacho ≤ 3.4 h, kill 3.9 h)

| kernel (`rodrimandub/af-pikans-exp-0028-phase1d-…`) | corridas | pared estimada (peor caso) | margen al kill de 3.9 h |
|---|---|---|---|
| `ac` | B,E,F,C × 4101–4105 = 20 | 2.6 h (3.2 h) | 0.7 h |
| `kpp` | B,E,F × 4101–4105 = 15 | 2.1 h (2.5 h) | 1.4 h |
| `bu1` | B,F,E × 4101–4102 = 6 | 3.2 h | 0.7 h |
| `bu2` | B,F,E × 4103–4104 = 6 | 3.2 h | 0.7 h |
| `bu3` | B,F,E × 4105 = 3 | 2.1 h (una ranura libre en la 2.ª ronda) | 1.8 h |

Total ≈ 13.2 h de sesiones (cuota semanal de GPU de Kaggle a verificar antes de lanzar). Burgers no admite más de 6 corridas por kernel (3 rondas × 1.03 h); por eso son tres. Alternativa de 4 kernels (mezclar KPP con Burgers) llega a 3.2 h por ranura pero con margen < 0.7 h: no se recomienda. Orden de despacho: por semilla y B primero; si el tope corta, faltan semillas completas, no un brazo suelto. Las corridas son reanudables (un JSON por corrida), así que un kernel cortado se relanza.

## Pruebas hechas (CPU, humo; nada se lanzó ni se subió)
- `run_phase1d.py` de punta a punta: ac y kpp (B, F, E, C) y Burgers (B, F, E, incluido el cambio a float64 de L-BFGS), 12 pasos Adam / 5 it. L-BFGS / N = 64. Salida con `smoke: true`.
- **B de 1d == B de `run_phase1.run_one` bit a bit** (1 hilo, misma receta corta: error Adam 0.7203229003093471, final 0.7567394480095766, dos veces). Con 2 hilos aparece ruido de ~1e-9 por reducciones, ajeno al código.
- Kernel completo `bu3` desde el bundle plano (`X28_TINY=1`, `BUNDLE_DIR`), 3 corridas, `complete: True`.
- `analyze_phase1d.py` con datos sintéticos de resultado conocido (sostenido, cambia de signo, se pierde, aparece, provisional con n = 4).

## Comandos (los de subida/lanzamiento NO se han ejecutado)
```
PY="C:\Program Files\Python312\python.exe"
# construir (ya hecho; reescribe solo kaggle/dataset_1d, kernel_1d_*, approval_1d.json, SHA256SUMS_1d.txt)
$PY experiments/mechanisms/EXP-0028-failure-attribution/kaggle/build_bundle_1d.py
# humo CPU de una corrida
$PY run_phase1d.py --problems ac --arms B,E,F,C --seeds 4101 --out OUT --adam 12 --lbfgs-iters 5 --pre-every 5 --pre-n 64
# subida y lanzamiento (a ejecutar solo por orden del investigador)
kaggle datasets create -p kaggle/dataset_1d
kaggle kernels push -p kaggle/kernel_1d_ac     # ... kernel_1d_kpp, kernel_1d_bu1, kernel_1d_bu2, kernel_1d_bu3
# análisis tras descargar las salidas
$PY analyze_phase1d.py --dir OUT_ac --dir OUT_kpp --dir OUT_bu1 --dir OUT_bu2 --dir OUT_bu3 --out results/phase1d_scored
```
Estas cifras son proyecciones de corridas previas; el primer kernel (`kpp`, el más barato) las confirma o corrige antes de lanzar Burgers.
