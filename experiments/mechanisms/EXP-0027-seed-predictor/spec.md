# EXP-0027 — ¿Predice el estado en la primera intervención qué semillas dañan a E en Burgers?

**Estado: FIRMADO por el investigador en el chat el 2026-10-23** («Sí, lo firmo el XP»), sin cambios de umbral ni de covariable. Solo CPU, sobre datos ya guardados de EXP-0026 (`results/exp0026/burgers-*.json`). No lanza GPU. Se ejecuta `analyze_seed_predictor.py` tal como está escrito.

## Pregunta
En Burgers, E mejora a B en la mayoría de semillas y empeora mucho en unas pocas (semillas 1 y 105: >2× con ambas cadencias; 21.6× en 105 con E500). EXP-0026 descartó el recálculo al arrancar L-BFGS y la cadencia como explicación. ¿Hay una cantidad medible en el estado de la semilla **justo antes de la primera intervención** que prediga log(e_E/e_B)?

## Datos (fijados)
- Semillas: 1–5, 101–105, 3001–3010 (n = 20). Brazos con 20 semillas: B, E500, E1000L.
- Resultado por semilla: `y_s = ln(primary_final[E] / primary_final[B])`. Positivo = E peor que B.
- Covariable (una sola, fijada): `c_s = log10(loss de entrenamiento de B en el paso de la primera intervención)`. Es `hist[step==500].loss` de B para E500 y `hist[step==1000].loss` de B para E1000L. Se toma del brazo B porque hasta la primera intervención la trayectoria de E y la de B comparten semilla y estado; el script verifica esa igualdad (tolerancia relativa 1e-6) y, si no se cumple, lo informa sin cambiar la covariable.
- No se usa ninguna cantidad medida después de la primera intervención ni el error final de B como covariable (acoplaría ruido de B con y).

## Contraste primario (P1): E500
Spearman ρ entre `c500` y `y(E500)`, n = 20, a dos colas, p por permutación (10 000, semilla 0). **Respaldado** solo si se cumplen las cuatro:
1. |ρ| ≥ 0.50.
2. p ≤ 0.05.
3. ρ tiene el mismo signo en {3001–3010} y en {1–5, 101–105} por separado.
4. Quitando las dos semillas con mayor y, |ρ| ≥ 0.40 con el mismo signo.

## Contraste secundario (P2): E1000L
Misma construcción con `c1000` y `y(E1000L)`. Es réplica descriptiva: «consistente» si ρ tiene el mismo signo que P1 y |ρ| ≥ 0.40. No añade veredicto propio.

## Exploratorio (sin veredicto, sin p interpretada)
ρ de `y` contra `primary_adam` de B y contra la pendiente de log(loss) de B entre los pasos 300 y 500. Se reportan en una tabla marcada «exploratorio».

## Reglas de interpretación (fijadas antes de ver datos)
- **P1 respaldado:** hay un candidato a regla de uso de E en Burgers («aplicar E solo si c está en tal rango»). **No es un claim.** Requiere semillas nuevas y umbral de corte fijado antes, con otra firma, y comprobar que no rompe Allen–Cahn.
- **P1 no respaldado:** se cierra la pregunta en Burgers como «dependencia de semilla sin predictor en el estado de la primera intervención». Se documenta como falla caracterizada y no se abren más factoriales sobre cadencia o recálculo.
- No se bajan umbrales ni se añaden covariables después de ver resultados. Un hallazgo exploratorio solo puede proponer un EXP nuevo.

## Límites conocidos
n = 20 con colas pesadas; el contraste tiene poca potencia para |ρ| < 0.45. Un resultado no respaldado no prueba que no haya predictor, solo que este no lo es con estos datos.
