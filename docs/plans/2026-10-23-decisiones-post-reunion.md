# Decisiones después de la reunión (2026-10-23) y cómo se atan a D1–D5

**Estado:** registro de lo dicho por el investigador en el chat el 2026-10-23. **Ningún spec queda firmado por este documento**; firmar sigue siendo acto humano (AGENTS.md). Cada bloque nuevo necesita su spec preregistrado, firma y, si es GPU, autorización de lanzamiento.

## 1. Qué cambió con la reunión

Hasta ayer el camino era "buscar un mecanismo que mejore en Allen–Cahn, Burgers, KPP…" y, si no, cerrar en negativo. La reunión **cambia la pregunta**:

> De "¿qué mecanismo mejora?" a **"¿qué hace que un mecanismo funcione en un problema y falle en otro, y se puede predecir antes de correrlo?"**

Esto desbloquea el cuello de botella actual: E ayuda en Allen–Cahn (0.31 agrupado, 10/10), daña en Burgers según el bloque (1‑B V: 5.70; T1: 0.43; EXP‑0026: depende de la semilla) y empeora en Fisher–KPP. Sin explicar eso no se puede pasar a los dos problemas finales (piezoelectricidad y Navier–Stokes). La meta del fin de semana es **cerrar** este eslabón.

## 2. Las decisiones (lo dicho, ordenado)

| # | Decisión del investigador | Qué produce |
|---|---|---|
| R1 | Probar y reportar **qué parte del mecanismo, de las señales o del entrenamiento** hace que falle en un problema y no en otro. Se vale cualquier banco; lo importante es probarlo y reportarlo. | Experimento de **atribución de la falla** (descomposición por componente). |
| R2 | Desarrollar un **análisis de propiedades por PDE**: qué propiedades del problema hacen que un mecanismo de esta clase funcione o no, y cómo estimarlo/acoplarlo. | **Tabla de propiedades × problema** + ajuste de qué propiedades se correlacionan con el efecto. |
| R3 | **Piezoelectricidad:** probar el sistema de ecuaciones con el benchmark del paper. Si una decisión previa no está alineada con el paper, correrla igual y reportarla. | Corrida de piezo en el benchmark del paper + reporte de desviaciones. |
| R4 | Buscar **explicaciones** de por qué funcionó, por qué no, por qué pudo no haber funcionado. | Cada resultado lleva su explicación candidata *y* su prueba. |
| R5 | Con R1+R2, **predecir antes de correr** (batería Allen–Cahn, Burgers, KPP, Navier–Stokes) qué funcionará y qué no. | Predicciones **preregistradas** por problema. |
| R6 | **Experimento final:** el acople en piezo, ya respaldado por batería de pruebas + señales + estimaciones. | Un único experimento de piezo, cuya configuración sale de R1–R5. |

Lectura de la lógica: **R1 explica lo que ya pasó → R2 lo generaliza a propiedades → R5 lo convierte en predicción → R6 es la prueba final fuera de muestra.** Es un diseño de validación predictiva: la tesis no afirma "E mejora", afirma "sé cuándo E mejora".

## 3. Cómo se ata a D1–D5 (tabla del brief)

| D | Alternativas del brief | Efecto de las decisiones de reunión | Propuesta |
|---|---|---|---|
| **D1** Alcance del aporte con E | (a) hallazgo condicionado a Allen–Cahn + negativo caracterizado en Burgers/KPP; (b) ampliar el banco antes de afirmar | R1+R2 son exactamente "caracterizar el negativo" y R5 es "ampliar el banco" con predicción. **Se resuelve como (a) + (b) con orden:** el alcance se escribe condicionado a la familia AC *ahora*, y se amplía solo si la predicción R5 acierta en los problemas nuevos. | **(a) provisional, (b) condicionado a R5.** |
| **D2** Firma de EXP‑0027 (predictor de semillas) | (a) firmar y correr en CPU; (b) cerrar Burgers como dependencia de semilla | R1 pide explicar la falla; EXP‑0027 es la parte "dentro de Burgers, entre semillas" y cuesta CPU sobre datos ya guardados. Se vuelve **un caso del R1**, no un fin en sí. | **(a) firmar y correr** (barato, preregistrado); su resultado alimenta R2 con una propiedad del estado inicial. Si no se respalda, se cierra como dice el spec. |
| **D3** Confirmar E con la receta completa (nivel 2) | (a) campaña GPU 5–10 semillas; (b) reportar receta corta | Como R5/R6 se apoyan en E, un E solo con receta corta pone en duda las predicciones. Pero la campaña completa es el gasto grande. | **Diferir hasta ver R1/R2:** la receta completa solo vale la pena en los problemas donde la predicción diga "funciona". Mientras tanto reportar receta corta declarándolo (b). |
| **D4** Muestreador autogobernado (EXP‑0014) | (a) posponer; (b) correr | Mantiene lo mismo que ya decía el registro: su gate solo pesa cuando las señales coinciden. Puede entrar a R1 como **brazo de ablación** si es barato, no como línea. | **(a) posponer**, salvo que R1 lo pida como ablación. |
| **D5** Qué se afirma de E frente a F | F retiene parte de la ganancia sin activaciones entrenadas; la ventaja puede venir de corregir el solape de la base RBF | Esto **es** la primera ablación de R1: separar "corrección del solape de la base" de "activaciones entrenadas". Define cuál es el componente causante. | **Se vuelve el experimento 1 de R1** (ver §4). No se afirma E sobre F hasta tenerlo. |

## 4. Plan concreto para cerrar el fin de semana

Orden por dependencia y costo. Todo lo de abajo es **propuesta**; cada experimento necesita su spec y firma.

1. **R1 — Atribución de la falla (EXP‑0028, borrador).**
   Descomponer E en sus piezas y correrlas en Allen–Cahn (funciona), Burgers (depende de semilla) y KPP (falla), con semillas ya existentes cuando se pueda:
   - corrección del solape de la base RBF (F) vs activaciones entrenadas (E) → resuelve D5;
   - cadencia, recálculo, muestreo (R): EXP‑0026 ya descartó cadencia y recálculo como explicación de las semillas malas; no repetir;
   - el muestreo por reemplazo de R (≈45 % puntos únicos, 36–46 % en la banda del choque): **sospecha abierta**, barata de probar;
   - señal vs entrenamiento: ¿falla porque la señal es mala o porque el paso L‑BFGS la deshace? (antecedente EXP‑0011/0012).
2. **EXP‑0027 (D2a)** en paralelo: CPU, datos guardados, sin GPU.
3. **R2 — Tabla de propiedades por PDE.** Propiedades medibles *antes* de entrenar, para Allen–Cahn, Burgers, KPP, Heat, DFG/NS y piezo: presencia de choque/frente, rigidez, régimen difusión‑reacción vs advección, cond(B) inicial (EXP‑0022/0023), número de condición del residuo, concentración espacial del residuo, tipo de condición de borde (duro/blando). Variable respuesta: log(E/B) por semilla/problema. **Honestidad estadística:** con ~6 problemas y ≈4 puntos de contraste el ajuste es descriptivo; la fuerza viene de R5, no del ajuste.
4. **R5 — Predicciones preregistradas.** Antes de tocar NS/DFG y piezo: escribir en un archivo fechado y firmado qué predice la tabla para cada uno (mejora / empeora / indiferente, con magnitud aproximada). Tiene que quedar *antes* de correrlos; si acierta es la evidencia más fuerte de la tesis, si falla se reporta igual.
5. **R3 — Piezo con el benchmark del paper.** El inventario de `2026-10-20-faltantes-ns-dfg-piezo.md` §C sigue vigente y es el trabajo previo: `PiezoProblem`, convenciones de signo (polarización y e31/e33), licencia, referencia de terceros, etapa L‑BFGS inexistente, dtype. La instrucción de reunión es correr con las configuraciones del paper aunque una decisión previa nuestra se desvíe, y **reportar la desviación**.
6. **R6 — Acople en piezo:** un solo experimento, con la configuración derivada de R1–R5.

## 5. Riesgos y lo que no hay que prometer

- La regla de salida firmada el 2026-10-20 (22:01) decía que, si 1‑B no pasaba T1, terminaba la búsqueda de mecanismo en Burgers/Allen–Cahn. Esto **no la contradice**: no se busca un mecanismo nuevo, se explica uno ya medido. Conviene dejarlo escrito como aclaración para que no se lea como reabrir la búsqueda.
- Los resultados de la ronda 1‑B y los EXP 0014–0018 son negativos; no se re‑proponen sus familias.
- Con Burgers n=20 y colas pesadas, EXP‑0027 tiene poca potencia para |ρ|<0.45 (límite ya declarado en su spec).
- La explicación de por qué E falla en KPP es hoy hipótesis. No debe entrar a la tesis como causa hasta que R1 la pruebe.
- Piezo es lo más lejano (problema sin implementar en nuestro código, sin referencia de terceros). Si el fin de semana no alcanza para R3 completo, el plan degrada con gracia: R1+R2+R5 sobre los problemas ya implementados son un resultado cerrable por sí solo; R6 pasaría a trabajo futuro, declarado.

## 6. Lo que necesito que confirmes

1. D2(a): ¿firmas EXP‑0027 tal cual (sin cambios de umbral)?
2. ¿Autorizas redactar el spec borrador de **EXP‑0028 (atribución de la falla)** con D5 como primer brazo?
3. ¿Orden de piezo vs NS/DFG para R5: predecir en ambos antes de correr, o solo en piezo?
