# EXP-0029, pruebas previas del brazo S en Kaggle (T4x2, 2026-10-25/26): T1 y T2

Arranque en caliente desde el estado del paso 4 000 de B, 1 000 pasos, recálculo cada 500, semillas 20260728-30, Adam nuevo para todos. Fuente: `piezo-exp0029/` (T1: E y S), `piezo-exp0029-t2/` (T2: S0 = S con la guarda P4 en sombra). Cifras: mediana de las 4 evaluaciones (pasos 4250-5000) y mediana de la pérdida en los pasos 4500-5000. Cuota usada: 50 min (T0+T1) + 25 min (T2). Proxy de horizonte corto: da el signo, no el tamaño; en el converso B sigue en el llano a mitad de entrenamiento (u ≈ 0.8-0.95).

| problema | semilla | u S/E | u S0/E | φ S/E | φ S0/E | pérdida E / S / S0 |
|---|---|---|---|---|---|---|
| converso | 28 | 1.00 | 0.95 | 1.08 | 1.14 | 0.549 / 0.592 / 0.541 |
| converso | 29 | 0.97 | 0.93 | 1.01 | 1.06 | 0.426 / 0.463 / 0.413 |
| converso | 30 | 0.97 | 0.95 | 1.20 | 1.33 | 0.581 / 0.558 / 0.515 |
| directo | 28 | 1.09 | 0.97 | 1.03 | 0.97 | 0.036 / 0.070 / 0.031 |
| directo | 29 | 1.07 | 1.00 | 0.98 | 1.00 | 0.084 / 0.257 / 0.269 |
| directo | 30 | 1.01 | 0.85 | 1.12 | 0.75 | 0.323 / 0.752 / 0.272 |

Lectura. (1) La guarda P4 con amortiguación daña (pérdida 2-3× mayor en el directo; borra el estado de Adam 3 veces por corrida): se retira como respuesta activa, queda como registro en sombra. (2) Sin amortiguar (S0 = E + balance P2 b = 0.3 + arranque denso P5): u mejora poco (mediana 0.95, ninguna semilla > 1.00 salvo un empate), pero φ del converso **empeora** (×1.06-1.33), lo contrario de H-ARA-1. En el directo φ es mixto (0.75-1.00). (3) Criterio de φ del converso (mediana ≤ 0.95): no se cumple. Con n = 3 y un proxy de horizonte corto no se afirma que P2 daña; se afirma que **no hay apoyo para que mejore φ**.
