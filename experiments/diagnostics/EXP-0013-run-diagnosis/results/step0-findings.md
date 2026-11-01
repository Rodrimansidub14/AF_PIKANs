# EXP-0013 — hallazgos del paso 0 (inventario), 2026-10-05

Qué se hizo: `inventory.py` (solo lectura, CPU) listó los checkpoints que respaldan los baselines v1/v2, sus brazos pareados, EXP-0011/0012 y piezo; calculó sha256 y probó la carga estricta en la clase de modelo correspondiente. Detalle: `inventory.csv` (una fila por checkpoint) e `inventory-summary.md` (por celda). No se usó ninguna referencia ni etiqueta. No cambia el spec firmado: son datos para el análisis.

**Resultado (actualizado con el brazo FNN de Heat, a petición del investigador el 2026-10-05): 248 checkpoints listados; 238 cargan sin error (0 errores de carga); 10 no probados (piezo: necesita el código upstream de Daniel, `daniel_arm.py`); 0 faltantes.**

## Cobertura por problema (semillas × brazo × etapa)

| Problema | Brazos | Etapas | Semillas | Observaciones |
|---|---|---|---|---|
| Heat 1D | RBF (control S2; EXP-0011 RAD y réplica del control; EXP-0012) y, **añadido el 2026-10-05**, la campaña `heat-v4-isocost` con RBF (3 648 par.) y FNN 3×20 tanh con traza dura (921 par.) | S2/EXP-0011/0012: Adam y L-BFGS; **v4: solo Adam** | 901–905 | El pareo RBF−FNN existe **solo en v4** y es a **igual tiempo de GPU (500 s), no a igual número de pasos** (RBF 18 044 pasos, FNN 48 038 en la semilla 901; sin L-BFGS; parada por tope de tiempo durante Adam; pool Sobol compartido). Es un protocolo distinto al del control S2 y no se mezcla con él; la campaña v4 quedó `incomplete_validation`. El pareo en la etapa L-BFGS no existe en Heat. |
| Burgers | MLP 5×100, RBF16 (solo `close`), RBF48 | Adam y L-BFGS | 301–305 | 3 campañas con el mismo Adam y distinto L-BFGS/tamaño: sirven para estudiar L-BFGS en f32/f64 y con más puntos. |
| Allen–Cahn | MLP 3×20, RBF16 | Adam y L-BFGS | 601–605 | completo y pareado. |
| NS2d-CG | MLP 5×100 (n=5), RBF (8: RBF56 ×5 y RBF16 ×3) | **solo Adam** | 701–705 | sin L-BFGS: no se puede estudiar la transición Adam→L-BFGS. |
| DFG | cohorte A: solo RBF (Adam y L-BFGS, 301–305); cohorte B: RBF y MLP 5×64 (501–505, Adam y L-BFGS); cohorte C: MLP 4×50 y RBF16 (801–805, receta de Daniel y receta larga, Adam y L-BFGS) | ambas | 301–305, 501–505, 801–805 | Tres recetas distintas; no se mezclan entre sí. |
| Piezo | MLP 3×50 SiLU y RBF16 | **solo Adam** | 20260728–32 | sin L-BFGS; la carga requiere el código upstream (se prueba al calcular señales). |

## Consecuencias para el análisis (declaradas)

1. **Dependencia entre unidades:** corridas de la misma semilla comparten pools y arranque entre brazos y etapas; las permutaciones y los intervalos se hacen **por bloques de semilla**, y n efectivo por celda es 5 (no el número de filas).
2. **Etapa L-BFGS disponible** en Heat, Burgers, Allen–Cahn y DFG; no en NS2d-CG ni piezo (solo Adam).
3. **Pareo RBF−MLP disponible** en Burgers, Allen–Cahn, NS2d-CG, DFG (B y C) y piezo; en Heat solo en la campaña v4 (iso-tiempo, solo Adam); no en la cohorte A de DFG.
4. **Señales que dependen de los pools de entrenamiento** (muestreo, discrepancia) solo se pueden calcular donde se guardaron; las que usan solo modelo y PDE sobre una malla densa (residuo, concentración, ocupación de bases, jacobiano) están disponibles para los 228 checkpoints cargables.
5. **Cohorte A de DFG** requiere la clase `src.research_pipeline.sciml.dfg.DFGRBFPIKAN` (con `output_scales`), distinta de la usada en B y C.
