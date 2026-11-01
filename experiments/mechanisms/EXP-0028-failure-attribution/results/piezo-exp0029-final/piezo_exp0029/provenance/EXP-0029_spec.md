# EXP-0029, mecanismo final M en piezo: especificación y firma (2026-10-26)

**Firma.** El investigador, en el chat del 2026-10-26: «firmo N, no hacen falta pruebas, córrelo» (tras revisar DISENO_FINAL.md). Se toma como firma de la hipótesis y las predicciones de `DISENO_FINAL.md` §3, tal como están escritas, y de este bloque de corridas. Si la intención era otra, corregir aquí antes de leer resultados.

**Brazo M** (`src/mechanisms/gram_preconditioning.py`, arm `M`): cuerpo E (P por arista, recálculo cada 1 000 pasos) + P2 con b elegido en cada recálculo (mayor b de {0.05, 0.1, 0.2, 0.3, 0.5, 1} con p90 de la cond efectiva ≤ 10, tope 0.5) + P5 (recálculos extra en 250 y 500) + P4 solo en sombra (eventos registrados, sin amortiguar) + P1 sin señal en línea (medida fuera: ≈ 0). P3 y P6 en 0 (hay presupuesto). Receta del paper: Adam 20 000 pasos, float64, sin L-BFGS, pesos fijos.

**Bloque (12 corridas, Modal, `modal_piezo.py --mode exp0029`, 3 contenedores x 4 procesos):** M semillas 20260728-30 en converso y directo (6); B, E y M semillas 20260731-32 en el converso (6). El directo de las semillas 31-32 queda fuera por presupuesto (se declara en la tesis). Prioridad de recorte si se acaba el crédito: lo ya terminado se conserva (cada corrida se escribe al terminar y los logs parciales se publican cada 5 min).

**Predicciones (de DISENO_FINAL §3):** (1) converso: l2re_u(M)/l2re_u(E) ∈ [0.85, 1.15] y l2re_u(M)/l2re_u(B) ≤ 0.5; (2) directo: l2re_u(M)/l2re_u(E) ∈ [0.85, 1.15], ninguna semilla de M > 1.5× E; (3) l2re_φ(M)/l2re_φ(E) ≤ 1.20 en el converso; (4) saltos de pérdida de M ≤ los de E + 3 en el directo; (5) tiempo por paso de M ≤ 1.10× el de E. Se falsa si u(M)/u(E) sale del intervalo en 2 de 3 semillas de un problema. n = 3 (28-30) para M frente a B y E de la cohorte ya medida; semillas 31-32 amplían B y E a n = 5 y M a n = 5 en el converso.

**Hardware:** se pide A100 (`X28_MODAL_GPU=A100`) por costo; si la velocidad medida a los pocos minutos es peor que la de la H100 en la etapa 0, se relanza en H100. Comparabilidad: B y E de las semillas 28-30 corrieron en T4 (semilla 28) y H100 (29-30).
