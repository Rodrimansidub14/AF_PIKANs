# EXP-0027 — Resultados (2026-10-23)

Ejecutado `analyze_seed_predictor.py --dir ../EXP-0026-e-burgers-diagnostic/results/exp0026 --out results` tal como está firmado. Salida completa en `results/`. CPU, n = 20.

## P1 (E500, c500): **no respaldado**
| Condición | Valor | ¿Cumple? |
|---|---|---|
| 1. \|ρ\| ≥ 0.50 | ρ = +0.180 | no |
| 2. p ≤ 0.05 | p = 0.445 | no |
| 3. mismo signo en {3001–3010} y {1–5, 101–105} | +0.285 / −0.176 | no |
| 4. quitando las 2 semillas de mayor y, \|ρ\| ≥ 0.40 | +0.261 | no |

## P2 (E1000L, c1000): inconsistente con P1
ρ = −0.014, p = 0.956. Signo distinto al de P1 y |ρ| < 0.40.

## Exploratorio (sin veredicto)
ρ(y, error tras Adam de B) = +0.27; ρ(y, pendiente de log-pérdida de B entre pasos 300–500) = +0.30. Ninguna es base para un claim; solo podrían proponer un EXP nuevo.

## Salvedad de verificación
El script comprueba que hasta la primera intervención E y B comparten trayectoria (tolerancia relativa 1e-6). **No se cumple**: la desviación máxima es 1.67 (P1) y 1.89 (P2). Según el spec, se informa sin cambiar la covariable. Consecuencia: `c` (pérdida de B en el paso de intervención) no es exactamente la pérdida de E en ese paso, lo que debilita aún más a `c` como predictor. No cambia el veredicto (el contraste ya es nulo), pero habría que entender por qué las trayectorias divergen antes de la primera intervención (¿recálculo en el paso 0?).

## Veredicto según las reglas firmadas
P1 no respaldado → **Burgers se cierra como «dependencia de semilla sin predictor en el estado de la primera intervención»**. Falla caracterizada; no se abren más factoriales de cadencia o recálculo. Límite declarado: n = 20 con colas pesadas, poca potencia para |ρ| < 0.45; no prueba que no exista predictor, solo que este no lo es.

Este cierre pasa a EXP-0028 como dato de entrada: la variación entre semillas dentro de Burgers no se explica por la pérdida inicial; el análisis se mueve a nivel de componente del mecanismo y de propiedades del problema.
