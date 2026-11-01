# EXP-0028 Fase 1e — receta completa para J e I (Enmienda v1.4) — presupuesto y decisión de cuota (2026-10-24). NO LANZADO, NO SUBIDO

Código nuevo (nada de lo ya subido se modificó; los ejecutores 1c / 1c-dfg / 1d se usan tal cual): `run_phase1e.py` (fija la receta y delega por `runpy`), `kaggle_block_1e.py`, `kaggle/build_bundle_1e.py`, `analyze_phase1e.py`, `budget_1e_calc.py` (reproduce las tablas de abajo). Bundle: `kaggle/dataset_1e/` (dataset NUEVO `rodrimandub/af-pikans-exp0028-bundle-1e`) y `kaggle/kernel_1e_*`, `approval_1e.json`, `SHA256SUMS_1e.txt`.

## Recetas (semillas 4101–4105 en todo)

| problema | brazos | receta | cómo se invoca |
|---|---|---|---|
| NS2d-CG | **B, F, E, J, I** | Adam 20 000 f32, lote completo + L-BFGS 500 it. (la de `run_phase1c.py --recipe full`); GN 2048 por clave, 21 recálculos J/E/I | `run_phase1c.py --recipe full` |
| DFG | **B, F, E, J, I** | **Adam 20 000 (minibatch 1024 + 256/frontera, f32) + L-BFGS 4 000 it.** (ver abajo) | `run_phase1c_dfg.py --adam 20000 --lbfgs-iters 4000` |
| Allen-Cahn d=1e-3 | **J, I** | la de 1d: Adam 40 000 f32 + L-BFGS f32 (300 s / 15 000 it., meseta) | `run_phase1d.py --problems ac` |
| KPP | **J, I** | la de 1d (heredada de AC) | `run_phase1d.py --problems kpp` |
| Burgers | **J, I** | la de 1d: 16 384 pts, Adam 24 000 f32 + L-BFGS f64 (1 800 s) | `run_phase1d.py --problems burgers` |

**Receta completa de DFG (decisión mía, justificada; la ratifica o cambia el investigador).** La fase 1c usó Adam 2 000 + L-BFGS 500 it. (`dfg_blockers.md`). La línea base DFG con receta larga es `dfg_long.py` (kernel 2026-10-01): RBF16, float32, **Adam 20 000 pasos** lr 1e-3 + L-BFGS strong-Wolfe (hist. 100) con tope de 600 s, que se cortó siempre por **tiempo** a **~4 000 iteraciones** y «seguía mejorando» (u 0.191, Cd 35 %). Fijo: (i) **Adam 20 000** = los de `dfg_long` y 10× los de 1c, con el lazo y el minibatch de `train_dfg` ya validado (B == `train_dfg` paso a paso); (ii) **L-BFGS por iteraciones** (`sweep.lbfgs_stage`, como todo el banco 1c), tope **4 000 it.**, la profundidad que alcanzó RBF16 en `dfg_long`, sobre el pool completo de 16 384 + 5×512 y **sin tope de tiempo**, para que todos los brazos reciban las mismas iteraciones (un tope por reloj favorecería al brazo más rápido). Diferencias declaradas respecto de `dfg_long`: el problema es el de 1c (`DFGRBFPIKAN`, pools Sobol, pérdida de 11 términos con escalas), no el `RBFKAN` crudo de `dfg_long` con 22 345 puntos LHS de lote completo; por eso el B de 1e **no** se compara con las cifras de `dfg_long`, solo con el B pareado de 1e y con el B corto de 1c (informativo). `sweep.lbfgs_stage` puede terminar antes por meseta (se reporta `lbfgs_status` e iteraciones).

## Verificación del pareo con el B de 1d (ac, kpp, burgers)
- `run_phase1e.py` no reimplementa nada: delega en `run_phase1d.py` (byte a byte el subido en `dataset_1d/`). `build_bundle_1e.py` **aborta** si algún archivo que 1e comparte por nombre con `dataset_1d/` (23 archivos: arnés, `pre_trainer`, `sweep`, `t1_trainer`, problemas, referencias, `gram_preconditioning.py`…) difiere byte a byte. Pasa.
- Humo en CPU (1 hilo, Adam 12, L-BFGS 5): B vía `run_phase1e.py --include-b` **== B de `run_phase1d.py` directo, bit a bit** (error Adam, error final, pérdidas, `hist`, receta) y == al valor registrado en `budget_1d.md` (0.7203229003093471 / 0.7567394480095766).
- En GPU la igualdad bit a bit entre lanzamientos no está garantizada por el hardware (reducciones float32); lo que se garantiza es el mismo código y la misma receta. Si se quiere comprobar en T4, `X28_WITH_B`/`WITH_B = True` añade B a ac/kpp/burgers y `analyze_phase1e.py` imprime «B de 1e frente a B de 1d: IDÉNTICO / distinto» con el máx. |log razón|; el análisis usa siempre el B de 1d. Coste de esa comprobación: AC +0.64 h de sesión, KPP +0.67 h, Burgers +2.57 h.
- **Dependencia (hallazgo, no estaba en el encargo):** de 1d solo se corrió el kernel `ac` (`results/phase1d_ac_scored.md`: KPP y Burgers = 0 corridas). Los J,I de KPP y Burgers **solo tienen pareo cuando 1d `kpp` (~2.1 h) y `bu1–bu3` (~8.5 h) estén corridos**; ese coste NO está en las cifras de abajo. Alternativa barata para KPP: `WITH_B = True` en el kernel `acf` (+0.67 h) y listo.
- `residual_terms.py` del bundle 1e es la copia exacta subida con 1c/1c-dfg (la de `src/mechanisms/` ya difiere por el trabajo de piezo, solo en `piezo_residual_terms` y `_ADAPTERS`; NS y DFG no cambian). No se usa la del árbol de trabajo.

## Tiempos medidos (T4) y estimación por corrida (`budget_1e_calc.py`)
Base: NS 1c-t1 y `timing_ns.json` (Adam 0.0535 s/paso B, 0.0555 F, 0.0561 J/E/I incl. sobrecarga de aplicar P; L-BFGS 0.0776 s/it. [medido 0.0742]; recálculo 5.4 s); DFG 1c-dfg (B 154.3 s / 2 000 pasos = 0.0771; F 0.0801; J/E/I 0.082; L-BFGS 0.123 s/it.; recálculo 6.2–7.0 s); AC 1d medido (E 0.0218 s/paso, corrida 884 s; B 912 s); KPP/Burgers de la fase 1 y `budget_1d.md`. Sobrecarga por proceso (arranque, calentamiento de 50 pasos, pools, puntuación) 40 s en NS/DFG y 25 s en el resto.

| corrida (s) | B | F | J | E | I |
|---|---|---|---|---|---|
| NS (20 000 Adam + 500 L-BFGS) | 1 149 | 1 189 | 1 314 | 1 314 | 1 314 |
| DFG (20 000 Adam + 4 000 L-BFGS) | 2 075 | 2 133 | 2 311 | 2 266 | 2 331 |
| AC | (1d) | | 957 | | 957 |
| KPP | (1d) | | 1 037 | | 1 053 |
| Burgers (L-BFGS en el tope de 1 800 s) | (1d) | | 3 815 | | 3 854 |

## Partición en kernels (T4×2, 2 ranuras, despacho ≤ 3.4 h, kill 3.9 h) y cuota

Cuota de Kaggle = **horas de sesión** (pared del kernel; la T4×2 cuenta 1 h por hora de pared, p. ej. el kernel `dfg-long` de 3.34 h bajó la cuota 3.34 h). «GPU-h» ocupadas (las dos T4) = 2× esa cifra.

| kernel (`rodrimandub/af-pikans-exp-0028-phase1e-…`) | prio. | contenido | corridas | sesión estimada | margen al kill |
|---|---|---|---|---|---|
| `ns1` | 1 | NS B,F,J,E,I × 4101–4103 | 15 | 2.83 h | 1.07 h |
| `ns2` | 1 | NS × 4104–4105 | 10 | 1.85 h | 2.05 h |
| `dfg1` | 1 | DFG B,F,J,E,I × 4101–4102 | 10 | 3.23 h | 0.67 h |
| `dfg2` | 1 | DFG × 4103–4104 | 10 | 3.23 h | 0.67 h |
| `dfg3` | 1 | DFG × 4105 | 5 | 1.94 h | 1.96 h |
| `acf` | 2 | AC J,I × 4101–4105, luego KPP J,I × 4101–4105 | 20 | 2.79 h | 1.11 h |
| `bu1` | 3 | Burgers J,I × 4101–4102 | 4 | 2.14 h | 1.76 h |
| `bu2` | 3 | Burgers J,I × 4103–4104 | 4 | 2.14 h | 1.76 h |
| `bu3` | 3 | Burgers J,I × 4105 | 2 | 1.07 h | 2.83 h |

(La sesión incluye 0.07 h de la prueba B == ID en GPU en los kernels ns/dfg. NS+DFG se mantienen en kernels separados: mezclar `ns2` con `dfg3` ahorraría 0.35 h pero llevaría ese kernel a 3.36 h. Las corridas son reanudables y los kernels respetan semillas completas si el tope corta.)

**Totales:** 145 349 s de corrida = **40.4 GPU-h ocupadas (dos T4) = 21.2 h de sesión (cuota)**: prioridad 1 (NS+DFG) **13.1 h**; + prioridad 2 (AC+KPP) = **15.9 h**; + prioridad 3 (Burgers) = **21.2 h**. Error de estimación esperado ±5 % (≈ ±1 h en total); las cifras de AC están calibradas con las 20 corridas reales de 1d (884–912 s), las de DFG/NS con la fase 1c.

## Decisión de cuota (escrita aquí; el investigador la confirma)
Cuota restante ≈ 20 h, se renueva el 31/oct. **21.2 h la exceden en ~1.2 h y sin margen para reintentos.** Propuesta de recorte:
1. **No se recorta nada de prioridad 1 (NS y DFG, 13.1 h) ni de prioridad 2 (AC y KPP, 2.8 h): 15.9 h acumuladas, ~4 h de margen.**
2. **Burgers (prioridad 3, 5.35 h) se difiere a después del 31/oct**, junto con los kernels de 1d `bu1–bu3` de los que depende su pareo. Razones: es lo más caro por dato, J,I de Burgers no tienen pareo hasta que 1d Burgers exista (no corrido), y recortar semillas no sirve: con n = 4 el veredicto sale PROVISIONAL y con n = 3 («sin veredicto» por la regla n < 4 de `analyze_phase1d.verdict`) no se obtiene etiqueta. Si el investigador prefiere Burgers ya, lo más defendible es `bu1`+`bu2` (n = 4, PROVISIONAL, 4.3 h): 20.2 h en total, sin margen (no recomendado).
3. Con el margen (~4 h) caben los reintentos de `dfg1/2` (los más justos) y, si se quiere el pareo de KPP esta semana, el kernel 1d `kpp` (~2.1 h) o `WITH_B = True` en `acf` (+0.67 h).

## Comandos (NINGUNO ejecutado salvo el de construcción y los humos en CPU)
```
PY="C:\Program Files\Python312\python.exe"; cd C:\Users\rodri\Documents\AF_PIKANs\experiments\mechanisms\EXP-0028-failure-attribution
# construir (ya hecho; reescribe solo kaggle/dataset_1e, kernel_1e_*, approval_1e.json, SHA256SUMS_1e.txt)
$PY kaggle/build_bundle_1e.py
# calculo del presupuesto
$PY budget_1e_calc.py
# humo CPU de una corrida por problema (ejemplos)
$PY run_phase1e.py --problem dfg --arms B,I --seeds 4101 --out OUT --adam 12 --lbfgs-iters 5 --pre-every 6 --pre-n 64 --smoke-pool 128
$PY run_phase1e.py --problem ac --arms J,I --seeds 4101 --include-b --out OUT --adam 12 --lbfgs-iters 5 --pre-every 5 --pre-n 64 --threads 1
# subida y lanzamiento (a ejecutar solo por orden del investigador; en este orden de prioridad)
kaggle datasets create -p kaggle/dataset_1e
kaggle kernels push -p kaggle/kernel_1e_ns1     # luego kernel_1e_ns2, kernel_1e_dfg1, kernel_1e_dfg2, kernel_1e_dfg3, kernel_1e_acf
# Burgers (diferido): kernel_1e_bu1, kernel_1e_bu2, kernel_1e_bu3
# analisis tras descargar las salidas
$PY analyze_phase1e.py --dir OUT_ns1 --dir OUT_ns2 --dir OUT_dfg1 --dir OUT_dfg2 --dir OUT_dfg3 --dir OUT_acf --dir1d results/phase1d-ac/exp0028_1d --phase1d results/phase1d_ac_scored.json --out results/phase1e_scored
```

## Pruebas hechas (CPU, humo; nada se lanzó ni se subió)
- `run_phase1e.py`: ac (B, J, I; B == `run_phase1d.py` bit a bit), ns (B, J), dfg (B, I), guardas (semilla fuera de 4101–4105 y brazo no permitido rechazados).
- Kernels desde el bundle plano (`X28_TINY=1`, `BUNDLE_DIR=kaggle/dataset_1e`): `bu3` (Burgers J,I; `complete: True`), `dfg3` (prueba B == ID de DFG + 5 corridas) y `ns2` con `X28_SEEDS=4105` (prueba B == ID de NS + 5 corridas); `complete: True`.
- `analyze_phase1e.py` con datos sintéticos de resultado conocido (J,I de AC pareados con el B real de 1d; NS con I a n = 4 → PROVISIONAL): clases y veredictos como se esperaban; Adam y L-BFGS por separado.
