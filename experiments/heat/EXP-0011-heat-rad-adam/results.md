# EXP-0011 — resultados (2026-10-04)

Kernel `rodrimandub/af-pikans-heat-rad-exp0011-20261004`: COMPLETE, sesión 4 327 s (1.2 h), 10/10 trabajos `rc=0` (5 RAD pre-registrados + 5 réplicas del control). Pre-registro firmado el 2026-10-03; los umbrales no se movieron. Exploratorio; `scientific_accepted=false`. Análisis: `analyze.py`, `results/run-20261004/analysis.json`, figura `figures/exp0011-paired.png`.

## Criterio primario (fin de Adam, pareado por semilla contra el control EXP-0008)

| Semilla | Control EXP-0008 | RAD | Razón RAD/control |
|---|---|---|---|
| 901 | 2.441e-4 | 6.23e-5 | 0.255 |
| 902 | 1.560e-4 | 9.11e-6 | 0.058 |
| 903 | 5.141e-5 | 1.80e-5 | 0.350 |
| 904 | 1.294e-4 | 9.10e-6 | 0.070 |
| 905 | 5.176e-4 | 7.53e-6 | 0.015 |

Media geométrica de las razones **0.088** (umbral ≤ 0.80) y **5 de 5** semillas con razón ≤ 0.80 (umbral ≥ 4). Media ± sd: RAD 2.12e-5 ± 2.33e-5 frente a control 2.20e-4 ± 1.80e-4. **El criterio primario se cumple con holgura** (~11× menos error). El tiempo de Adam es el mismo (RAD 482–489 s; el re-muestreo suma 0.8 s por corrida).

## Guardias

| Guardia | Resultado | ¿Cumple? |
|---|---|---|
| L2 tras L-BFGS completo ≤ 2× el control (≤ 3.7e-6) | RAD 7.2e-6 a 1.1e-5 frente a control 1.6e-6 a 2.1e-6; razón media geométrica **4.9** (3.4 a 6.8 por semilla) | **No** |
| Tiempo de GPU por corrida ≤ 1.5× el control | ~1.0× (Adam) | Sí |
| 5/5 corridas con pérdida finita | 5/5 | Sí |

## Veredicto según la regla pre-registrada

La condición de falsación dice que la hipótesis se rechaza si "(c) se violan las guardias". **La guardia de L-BFGS se violó, así que por la regla firmada H-RAD-HEAT-1 queda rechazada**, aunque el criterio primario se cumpliera. La decisión final es del investigador (`decision.md`); no se reinterpretó el resultado.

## Lo que hay que saber para leerlo (declarado, no cambia el veredicto)

1. **Qué se mide en L-BFGS.** La fase de L-BFGS se ejecutó sobre los últimos 4 096 puntos RAD (concentrados donde el residuo es alto), no sobre el pool uniforme del control. Esa elección era mía; el pre-registro decía "prueba de estrés" sin fijar el conjunto. Con puntos concentrados, el mínimo de la pérdida de L-BFGS no coincide con el error de campo uniforme: el error tras L-BFGS (~1e-5) queda por encima del piso del control (~1.9e-6). Esto es una **hipótesis explicativa, no verificada**. Probarla requiere un experimento nuevo con su propio pre-registro (p. ej. L-BFGS sobre el pool uniforme tras Adam con RAD); no se hizo ni se hará dentro de EXP-0011.
2. **Paridad del runner.** Las réplicas del control en este runner no reproducen por semilla los valores de EXP-0008 (razones réplica/EXP-0008: 0.96, 1.96, 1.93, 1.88, 0.46), aunque la media coincide (2.2e-4). La causa probable es no-determinismo de GPU/orden de operaciones frente a la sesión de S2; no se investigó. El control de EXP-0008 sigue siendo el comparador pre-registrado. Frente a las réplicas del mismo runner las razones son aún menores (media geométrica 0.070; 0.266, 0.030, 0.182, 0.037, 0.032).
3. **Potencia.** n = 5 semillas pareadas, un solo problema y una sola configuración de RAD (potencia 2, 50 % uniforme, refresco cada 1 000). No es un resultado general sobre RAD.

## Siguiente paso (decisión tuya)

- `decision.md` (humano): aceptar el rechazo por guardia, o abrir un experimento nuevo que preregistre L-BFGS sobre el pool uniforme.
- Nada de esto es un claim; cualquier uso en la tesis pasa por el gate `scientific_claim`.
