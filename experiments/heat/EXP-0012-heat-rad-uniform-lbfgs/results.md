# EXP-0012 — resultados (2026-10-04)

Kernel `rodrimandub/af-pikans-heat-rad-exp0012-20261004`: COMPLETE, sesión 1 968 s (0.55 h), 5/5 corridas `rc=0`. Pre-registro firmado el 2026-10-04; umbrales sin cambios. Exploratorio; `scientific_accepted=false`. Análisis: `results/run-20261004/analysis.json`.

L-BFGS completo (strong-Wolfe, tope 700 s) desde los checkpoints de fin de Adam de la rama RAD de EXP-0011, sobre el pool uniforme del control.

| Semilla | Control tras L-BFGS | EXP-0011 (L-BFGS en puntos RAD) | EXP-0012 (L-BFGS en pool uniforme) | Razón vs control |
|---|---|---|---|---|
| 901 | 1.949e-6 | 9.62e-6 | 3.74e-6 | 1.92 |
| 902 | 1.795e-6 | 8.72e-6 | 4.03e-6 | 2.24 |
| 903 | 1.634e-6 | 1.12e-5 | 3.36e-6 | 2.06 |
| 904 | 1.792e-6 | 9.04e-6 | 4.16e-6 | 2.32 |
| 905 | 2.113e-6 | 7.24e-6 | 3.50e-6 | 1.66 |

Media geométrica de las razones **2.025** (umbral ≤ 2.0); **2 de 5** semillas con razón ≤ 2.0 (umbral ≥ 4). 0 corridas no finitas.

## Veredicto según la regla pre-registrada

Se cumplen las condiciones de falsación (a) (media geométrica 2.025 > 2.0) y (b) (solo 2/5 semillas ≤ 2.0). **H-RAD-HEAT-2 queda rechazada.** La media geométrica pasa el umbral por 1.3 %, pero la condición (b) la rechaza con claridad, y no se reinterpreta.

## Lectura (no cambia el veredicto)

- Correr L-BFGS sobre el pool uniforme **explica una parte** del déficit: el error tras L-BFGS baja de ~9.7e-6 (puntos RAD) a ~3.8e-6, es decir ~2.5× mejor. Pero no cierra la brecha: queda ~2× por encima del piso del control (1.9e-6).
- Dos corridas (904, 905) llegaron al tope de 15 000 iteraciones y tres al tope de tiempo, así que L-BFGS seguía bajando; con más iteraciones el resultado podría cruzar el umbral. Eso es una hipótesis, no un resultado, y requeriría otro pre-registro.
- Conclusión honesta de la secuencia EXP-0011/0012 en este control: RAD reduce ~11× el error al final de Adam (estas 5 semillas) y deja un estado que, tras L-BFGS completo, queda ~2× peor que el control. Un solo problema, una configuración, n=5.
- Nada de esto es un claim; cualquier uso en la tesis pasa por el gate `scientific_claim`.

La decisión de qué hacer a continuación (cerrar el hilo RAD en Heat, o abrir EXP-0013) es del investigador (`decision.md`).
