# EXP-0024 · ronda 1 · nivel T1 — plan y estado (2026-10-20)

**Estado (2026-10-20, noche): bundle armado y prueba corta hecha en T4; la prueba NO pasó por el criterio fijado antes de correr (un solo punto, por 0.11 puntos porcentuales); el bloque NO se lanzó. Decisión pendiente del investigador.**

## Qué hay

- `t1_trainer.py`: corrida corta de la PIKAN-PDE con la pérdida de la PDE y la receta del control del EXP-0016, con el gancho de reubicación (`regrid.regrid_kan`). Entrena sin referencia; puntúa después (error al final de Adam y tras L-BFGS). Cada corrida se anota en `registry/designs.jsonl`.
- Convención T1 (fijada en el docstring): Adam = 10 % del control (Burgers 2 000 pasos, Allen–Cahn 4 000); L-BFGS con tope de **iteraciones** (500) para que sea determinista; el reloj se reporta aparte (guardia: ≤ 1.2 × el del control); el costo de reubicar entra en el reloj.
- Pruebas (`tests/test_wind_tunnel_t1.py`, 3 pasan): el control es **idéntico bit a bit** a `sweep.adam_stage` + `sweep.lbfgs_stage`; la reubicación sobre el modelo de la campaña cambia la forma de los centros a (entradas, 11), los mantiene crecientes, conserva la función (< 5 % de cambio) y no baja el rango efectivo; el lazo de entrenamiento no toca la referencia.

## Costo medido en CPU (2 núcleos de tu computadora)

| Problema | s por paso de Adam | s por iteración de L-BFGS | Una corrida T1 |
|---|---|---|---|
| Allen–Cahn (RBF16) | 0.59 | 0.40 | ≈ 43 min |
| Burgers (RBF48) | 1.38 | 0.87 | ≈ 53 min |

Bloque completo: 4 brazos (CTRL, D1, D2, D3) × 5 semillas (1–5) × 2 problemas = 40 corridas ≈ 30 h de CPU. **No es viable en CPU: va a GPU** (T4, como el EXP-0018). El tiempo en GPU no está medido; hace falta una prueba corta antes de estimar.

## Lo que falta antes de lanzar

1. Empaquetar para Kaggle como el EXP-0018 (`experiments/mechanisms/EXP-0018-responsibility-ceiling/kaggle/build_bundle.py` sirve de plantilla): `t1_trainer.py`, `regrid.py`, `sweep.py`, `pikan_campaign.py` y las referencias.
2. Prueba en GPU: que el brazo CTRL de T1 reproduzca la corrida de la CPU hasta tolerancia razonable (la GPU no es bit a bit con la CPU) y medir segundos por paso.
3. Autorización explícita del investigador para lanzar el bloque (como el `approval.json` del 0018). El protocolo EXP-0024 autoriza explorar sin firma por corrida, pero el lanzamiento en Kaggle es una acción externa y se pide aparte.

## Criterio para pasar a V (regla 3 del protocolo, ya firmada)

Mediana de la razón (error tras L-BFGS del diseño / control) ≤ 0.90 en Burgers **y** en Allen–Cahn en T1; luego, en V (semillas 100–199, mismo presupuesto), mediana ≤ 0.90 con mejora en ≥ 4 de 5 semillas en ambos problemas. Pasa a C (preregistro firmado) un solo finalista.

Candidatos: D1, D2, D3 (D4 descartado en T0: 3 de 5, mediana 0.81). Lo que dijo T0: Allen–Cahn favorece D1 y D2 (≈ 0.68–0.69×, 5/5); Heat muestra que D1 puede perder la ganancia tras L-BFGS. T1 mide justo eso con la pérdida de la PDE.

## Autorización de lanzamiento (2026-10-20 17:32)

Rodrigo Mansilla (chat: «si firmo», respondiendo a «¿Autorizás desde ya el bloque de 40 corridas en GPU, condicionado a que la prueba corta pase?»; registrado por el agente) autoriza lanzar en Kaggle (T4, dataset privado) el bloque T1: diseños CTRL, D1, D2, D3 × semillas 1–5 × Burgers y Allen–Cahn (40 corridas), con la convención T1 de este archivo.

Condiciones (las dos primeras son las que pidió el investigador; la tercera la fija el agente como resguardo y puede cambiarse):
1. **La prueba corta en la T4 debe pasar** (el control de T1 reproduce la corrida de CPU dentro de tolerancia razonable y se mide el tiempo por paso). Si no pasa, no se lanza el bloque y se informa.
2. Sin referencia en el entrenamiento: la referencia solo puntúa, después (ya cubierto por las pruebas).
3. Si la prueba corta estima el bloque en más de **4 h**, no se lanza sin volver a consultar. Tope duro de 5 h: no se despacha trabajo nuevo pasada la hora 4 y se detienen los trabajos en curso a la hora 5, igual que el criterio del 0018. Lo que no termine se reporta como ausencia de medición, no como cero.


## Resultado de la prueba corta en T4 (2026-10-20)

Kernel `rodrimandub/af-pikans-exp-0024-t1-shorttest` (T4×2, 275 s, semilla 1; resultados en `results/t1-shorttest-gpu/`). Criterio fijado **antes** de correr (docstring de `t1_shorttest_gpu.py`): error relativo GPU frente a CPU de CTRL ≤ 5 % (fin de Adam) y ≤ 10 % (tras L-BFGS); de D1 ≤ 20 % en ambos; estimación del bloque ≤ 4 h.

| | Fin de Adam: GPU / CPU (dif.) | Tras L-BFGS: GPU / CPU (dif.) | Criterio |
|---|---|---|---|
| Allen–Cahn CTRL | 0.6885 / 0.6885 (0.00 %) | 0.5799 / 0.5826 (0.46 %) | cumple |
| Allen–Cahn D1 | 0.6095 / 0.6094 (0.01 %) | 0.5169 / 0.5178 (0.17 %) | cumple |
| Burgers CTRL | 0.2782 / 0.2782 (0.02 %) | 0.1467 / 0.1459 (0.57 %) | cumple |
| **Burgers D1** | **0.1649 / 0.2064 (20.11 %)** | 0.1306 / 0.1558 (16.17 %) | **no cumple** (fin de Adam: 20.11 % > 20 %) |

- **Estimación del bloque: 0.71 h de sesión** (Burgers ≈ 106 s por corrida de CTRL y D3 con el presupuesto T1; Allen–Cahn ≈ 100 s; 0.039 y 0.021 s por paso de Adam), muy por debajo de las 4 h y del tope de 5 h.
- Todas las corridas terminaron y son finitas; los ajustes lineales de la reubicación en GPU tienen error relativo ≤ 0.02.
- Lectura: el control se reproduce en GPU con diferencias < 0.6 % en ambos problemas, y D1 en Allen–Cahn también. **Burgers D1 diverge entre CPU y GPU** (20 %): la reubicación (cuantiles y mínimos cuadrados con ridge sobre 25 % del Adam) amplifica diferencias de redondeo en ese problema. No se relajó el criterio después de ver el dato.
- Observación fuera del criterio (una semilla, no concluyente): en Burgers D3 (cuatro reubicaciones) termina en 0.0218 frente a 0.0019 del control a presupuesto T1 completo.
- Por la condición 1 de la autorización («si no pasa, no se lanza el bloque y se informa») **no se lanzó el bloque**.

Opciones para el investigador: (a) aceptar la prueba como pasada con esta salvedad documentada y lanzar (0.7 h); (b) repetir primero Burgers D1 dos veces en GPU para saber si la divergencia con la CPU es sensibilidad numérica del diseño o falta de reproducibilidad en GPU (≈ 6 min); (c) fijar otro criterio para D1 antes de decidir.


## Diagnóstico de la falla de Burgers D1 (2026-10-20, opción 2 del investigador)

Presupuesto reducido de la prueba (semilla 1, Adam 200, L-BFGS 50). Resultados en `results/t1-d1-diag-gpu/`, `results/diag-d1-f64-cpu/` y `results/diag_sensitivity.json` (`diag_sensitivity.py`).

| Corrida de Burgers D1 | Error al final de Adam | Error tras L-BFGS |
|---|---|---|
| float32, CPU | 0.2064 | 0.1558 |
| float32, T4, corrida 1 | 0.164859 | 0.130571 |
| float32, T4, corrida 2 | 0.164859 | 0.130571 |
| float64, CPU | 0.139920 | 0.094710 |
| float64, T4 | 0.150491 | 0.098273 |

1. **La GPU consigo misma: idéntica bit a bit** (dos procesos, cada uno con calentamiento). La diferencia con la CPU no es ruido de la GPU.
2. **float64 no la elimina:** la diferencia CPU–GPU baja de 20.11 % a 7.56 % (fin de Adam) y de 16.2 % a 3.76 % (tras L-BFGS), pero no desaparece. No es solo redondeo del entrenamiento.
3. **Sensibilidad (CPU, float32):** se perturban los coeficientes de la capa 0 con ruido relativo 1e-7 (orden del redondeo de float32) y A y B reciben el mismo tratamiento hasta el paso 200. **El control cambia 7.5e-6 relativo en el error final; D1 cambia 12.3 %** (amplificación ≈ 1.2e6 frente a ≈ 75). Justo después de reubicar, los coeficientes de A y B difieren solo 1e-7 a 8e-7: la amplificación ocurre en el entrenamiento posterior a la reubicación.
4. Lectura: la divergencia CPU–GPU de D1 en Burgers es **sensibilidad numérica del diseño** (una perturbación de redondeo basta para mover el error final ≈ 10 %), no ruido de la GPU ni un error de código (el control y Allen–Cahn coinciden con la CPU a < 0.6 %; D1 en Allen–Cahn a 0.17 %). Queda sin explicar por qué en float64 persiste una diferencia de 7.6 % (una perturbación de 1e-16 no debería llegar ahí con la amplificación medida en float32); puede deberse a otra fuente de divergencia entre CPU y GPU (p. ej. el orden de las sumas en el ajuste lineal de la reubicación o en la búsqueda lineal de L-BFGS) y no se investigó más.
5. **Consecuencia para el bloque:** en la misma GPU y tras el calentamiento las corridas son reproducibles bit a bit, y el comparador CTRL corre en la misma sesión; pero el resultado de D1/D2/D3 en Burgers por semilla tiene un componente de «azar numérico» grande. El criterio de T1 (mediana sobre 5 semillas) lo trata como ruido, con la potencia limitada que eso implica en Burgers.


## Prueba de la reubicación en CPU frente a GPU (2026-10-20)

Pedida por el investigador tras el diagnóstico: mismo estado (Burgers RBF48, semilla 1, 50 pasos de Adam, pesos guardados en float64), `regrid_kan` una vez en CPU y una en T4, sin entrenar. Criterio del investigador: diferencia < ~1e-9 → la reubicación es inocente. Kernel `rodrimandub/af-pikans-exp-0024-regrid-check`; resultados en `results/regrid-check-gpu/`.

| Diferencia relativa CPU–GPU, float64 | |
|---|---|
| centros capa 1 / capa 2 | 2.1e-16 / 6.7e-14 |
| anchos capa 1 / capa 2 | 2.9e-16 / 2.3e-13 |
| coeficientes capa 1 / capa 2 | 2.3e-11 / 4.2e-11 |
| salida del modelo justo después de reubicar | 9.1e-14 |
| **máximo** | **4.2e-11 < 1e-9** |

En float32 (modelo en float32, reubicación en float64 internamente) la máxima es 1.6e-6, del orden del redondeo de float32. El número de condición de la Gram de la capa 1 antes de reubicar es ≈ 1.5–1.7e15 en ambos dispositivos y baja a ≈ 4.2e4 después.

**Veredicto: la reubicación es inocente** (4.2e-11). La brecha de Burgers D1 queda como sensibilidad del entrenamiento posterior a la reubicación (el diagnóstico anterior: una perturbación de 1e-7 cambia el error de D1 12.3 % y el del control 7.5e-6), con la salvedad documentada de que la persistencia de 7.6 % en float64 no está explicada del todo. Por la regla del investigador, el bloque puede lanzarse.

## Decision del investigador sobre la idea A (2026-10-20, 22:01)

Firmado en el chat: (1) la idea A se cierra como **especifica de Allen-Cahn** (mejora 0.675-0.765x, 5/5; en Burgers empeora 4.5-21x, 0/5, con dosis-respuesta D2<D3<D1). La causa del dano en Burgers es hipotesis no probada. No se hacen mas diagnosticos ni variantes de A. (2) **Regla de salida:** si la ronda 1-B (precondicionadores) no pasa T1 con el criterio ya firmado (mediana <= 0.90 en ambos problemas), la busqueda de mecanismo en Burgers/Allen-Cahn termina y la tesis se plantea con el diagnostico de condicionamiento (EXP-0022), la cadena negativa 0015-0018 y la dependencia del problema, validado en un problema real (NS/DFG, piezo). Un resultado dependiente del problema no se reetiqueta como mecanismo.
