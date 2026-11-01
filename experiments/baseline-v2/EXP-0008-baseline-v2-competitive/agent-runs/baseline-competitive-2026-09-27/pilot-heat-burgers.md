# Paquete de decisión: piloto competitivo Heat y Burgers

Estado: **propuesta, no aprobada ni ejecutada**. Este documento prepara protocolos y presupuestos; no implica autorización de GPU, no envía corridas y no cambia evals/. Cada campaña Tier2 tiene su propio cap máximo de 4 GPU-h (14,400 GPU-s), incluyendo smoke tests y reserva. La reproducción de cada paper es un carril separado con presupuesto aparte; no se mezcla con la comparación isocosto AF.

La matriz bibliográfica y sus caveats están en [literature-heat-burgers.md](literature-heat-burgers.md). El alcance general está en [plan.md](plan.md).

## Pregunta y regla común

En cada contrato, medir si el RBF-PIKAN AF es competitivo con una alternativa definida antes de entrenar: PINN tanh DeepXDE para Heat y MLP PINNacle para Burgers. Heat añade el B-spline local como control AF. Los resultados impresos de los papers son contexto hasta reproducirlos; las conclusiones del piloto usan los reentrenamientos puntuados bajo el contrato AF común.

Los smoke tests de 120 GPU-s indicados más abajo solo verifican ejecución, malla, pérdida finita y logging. No puntúan calidad, no filtran candidatos y no determinan qué pasa a Tier2.

## Contratos y métricas

**Heat.** \(u_t=0.1u_{xx}\), \(x,t\in[0,1]\), \(u(x,0)=\sin(\pi x)\), \(u(0,t)=u(1,t)=0\), referencia \(u=e^{-0.1\pi^2t}\sin(\pi x)\). Métrica primaria: error relativo de campo en el grid AF held-out \(512\times256\). Secundarias: error máximo y RMSE de IC/BC, RMS de residuo PDE, energía \(\int u^2dx\) y diferencia frente a la curva exacta, parámetros, updates, wall time y GPU-s.

**Burgers.** \(u_t+u u_x-\nu u_{xx}=0\), \(\nu=0.01/\pi=0.003183098861837907\), \(x\in[-1,1]\), \(t\in[0,1]\), \(u(x,0)=-\sin(\pi x)\), \(u(\pm1,t)=0\), referencia Cole–Hopf con cuadratura gaussiana de orden 128. Métrica primaria: error relativo de campo en \(1025\times129\). Secundarias: error máximo y RMSE de IC/BC, RMS de residuo PDE, error de posición de \(\arg\max_x|u_x|\) por tiempo, pendiente máxima, parámetros, updates, wall time y GPU-s.

Los grids held-out solo se puntúan para el reporte de confirmación; no seleccionar hiperparámetros a partir de esos scores o mapas. Crear grids de desarrollo independientes con la misma referencia: Heat \(257\times129\), Burgers \(513\times129\). Si esas referencias de desarrollo no están disponibles, prepararlas sin cambiar los evaluadores protegidos.

Guardar pools deterministas comunes por seed. Heat: 4,096 interiores, 1,024 IC y 512 por cada frontera espacial. Burgers: 8,192 interiores, 1,024 IC y 512 por cada frontera, como el protocolo de EXP-0006. Usar float64, las mismas coordenadas, pesos y normalización de pérdidas, tratamiento soft de IC/BC y stop rule en todos los brazos de una PDE. Las PINN DeepXDE y PINNacle se adaptan al pool AF; son comparadores emparejados, no reproducciones literales. Registrar como desviación el cambio de precisión, pool o referencia respecto del paper.

## Campaña independiente: Heat

| Brazo | Configuración congelada | Uso |
|---|---|---|
| H-RBF | RBF-PIKAN AF con receta vigente y pool común. | Control del método propuesto. |
| H-BS | B-spline local con receta de su bundle existente y pool común. | Control local alternativo. |
| H-DX | FNN [2,20,20,20,1], tanh, Glorot normal; Adam lr \(10^{-3}\), luego L-BFGS. Target \(\alpha=0.1\), pool AF común. | Control arquitectónico de la receta DeepXDE. |

La receta oficial DeepXDE proviene de un tutorial para \(\alpha=0.4\), 2,540 interiores, 80 frontera, 160 IC, 20k Adam y luego L-BFGS. En el comparador AF se cambia a \(\alpha=0.1\) y al pool común. Por tanto, es una adaptación del método al contrato AF, no reproducción literal de la instancia publicada.

**Smoke dentro del cap Heat:** una seed por brazo, 120 GPU-s por corrida, máximo 3 ×120 =360 GPU-s. Solo verificar que corre, produce campos finitos y escribe logs; no comparar error de campo.

**Tier2 isocosto:** cinco seeds comunes y pareadas 901–905 para H-RBF, H-BS y H-DX. Cap de 900 GPU-s por corrida; máximo 15 ×900 =13,500 GPU-s. Guardar checkpoints a 120, 300, 600 y 900 GPU-s. H-DX usa Adam→L-BFGS según receta adaptada; si no alcanza L-BFGS dentro del cap, reportar el checkpoint como truncado. Reserva dentro del cap: 540 GPU-s. Total Heat, incluido smoke y reserva: **13,500+360+540 =14,400 GPU-s (4 GPU-h)**.

**Auditoría de cohortes Heat:** reutilizar resultados RBF/B-spline existentes como parte de la cohorte solo si la inspección confirma mismo hash/código, PDE, arquitectura y parámetros, pool, loss/weights, optimizador, precisión, unidades, referencia y grid/métrica de evaluación; para análisis pareado, las seeds deben alinearse entre brazos. Si falla una condición, los históricos quedan como contexto y se corren los brazos necesarios con las cinco seeds nuevas 901–905. No concatenar cohortes semánticamente distintas ni escoger cohortes por su score. La auditoría es previa a Tier2.

**Carril separado, receta/tutorial DeepXDE:** tres seeds nuevas 991–993; fijar versión/código, 2,540 interiores/80 frontera/160 IC, 20k Adam y L-BFGS según código del tutorial, target \(\alpha=0.1\). Puntuar en el grid tutorial \(256\times201\) y, adicionalmente, en evaluación AF. Presupuesto separado propuesto: hasta 1,800 GPU-s por seed, 5,400 GPU-s total. Si no termina el schedule, queda truncado y no se llama reproducción completa. Aunque mantiene la receta del tutorial, el cambio \(\alpha=0.4\to0.1\) impide presentarlo como reproducción literal de la instancia publicada. Este carril no consume el cap Tier2 de Heat y no interviene en selección.

## Campaña independiente: Burgers

| Brazo | Configuración congelada | Uso |
|---|---|---|
| B-RBF | RBF-PIKAN Sobol AF: 8,192 interiores, 1,024 IC, 512 por frontera, float64, receta de 20k pasos/tope 1,200 s de EXP-0006. | Control AF. |
| B-PINN | MLP tanh Vanilla PINNacle, cinco capas, ancho 100, Adam \(10^{-3}\), \(\beta=(0.9,0.999)\), hasta 20k updates; pool común Burgers y float64. Sin L-BFGS en este brazo. | Comparador isocosto literario; adaptación al contrato AF. |

La cifra publicada \(1.45\times10^{-2}\pm1.59\times10^{-3}\) corresponde a Vanilla PINN (Adam, 20k; tres corridas). La cifra \(1.33\times10^{-2}\pm5.30\times10^{-5}\) corresponde a otra variante L-BFGS, no a un paso final que deba concatenarse al schedule Vanilla. Mantenerlas como dos configuraciones separadas en cualquier reproducción. Ambas cifras usan referencia FEM discretizada y no son scores Cole–Hopf AF.

**Smoke dentro del cap Burgers:** una seed por brazo, 120 GPU-s por corrida, máximo 2 ×120 =240 GPU-s. Solo revisar salida finita y logging; no usar scores para retirar métodos.

**Auditoría y reutilización Sobol:** el bundle existente declara seeds 201–205 y tope 1,200 s/seed. Reutilizarlo como control solo si coinciden exactamente código/commit, manifiesto, coordenadas guardadas del pool por seed, loss, float64, updates, PDE, referencia, grid y definición de score, y si hay checkpoints/predicciones y GPU-s por dispositivo. El tope histórico está expresado en segundos; no convertirlo a GPU-s sin saber número de GPUs y telemetría. Si falta pool/hash o GPU-s, la auditoría falla. En tal caso volver a entrenar ambos brazos con cinco seeds nuevas 811–815, pools pareados y caps iguales.

**Tier2 isocosto:** cinco pares de seeds. Si pasa la auditoría, reutilizar B-RBF 201–205 y entrenar B-PINN con esos mismos IDs; si falla, usar RBF y Vanilla PINN nuevos con 811–815. Asignar hasta 1,200 GPU-s por seed a cada brazo; el RBF nuevo recibe el mismo tope y el PINN Vanilla es Adam hasta 20k o el límite, sin L-BFGS. Máximo 10 ×1,200 =12,000 GPU-s; reserva dentro del cap: 2,160 GPU-s. Total Burgers, incluido smoke y reserva: **12,000+240+2,160 =14,400 GPU-s (4 GPU-h)**. Puntuar curvas a 120, 300, 600, 900 y 1,200 GPU-s si hay checkpoint. Si Vanilla no completa 20k Adam al cap, su último checkpoint cuenta para isocosto y se marca truncado; no se afirma reproducción completa ni se atribuye la cifra publicada. Contar también el costo histórico del RBF reutilizado en el informe total aunque no consuma la nueva asignación.

**Carril separado, reproducción fiel PINNacle:** tres seeds independientes 821–823. Ejecutar por separado (a) Vanilla PINN con 20k Adam y (b) la variante L-BFGS según las configuraciones/repositorio publicados; no combinar ambos schedules salvo que la definición oficial de esa variante lo indique expresamente. Usar la referencia/grid FEM del paper, registrar hardware y detalles exactos del código. Presupuesto separado propuesto: hasta 1,800 GPU-s por seed y por configuración; máximo 10,800 GPU-s total para las dos configuraciones. Si el cap impide terminar la configuración, marcarla truncada, no fiel. Esta reproducción no consume el cap Tier2 Burgers ni reemplaza la comparación B-PINN Vanilla en Cole–Hopf AF.

vRBA/SSBroyden queda fuera de ambos Tier2: 5k Adam +60k SSBroyden no cabe razonablemente en el cap; además falta confirmar su condición de frontera contradictoria y el grid/hardware publicado. Su error \(O(10^{-8})\) no se usa como umbral.

## Regla de decisión fijada antes de scores

Todos los brazos listados en Tier2 quedan fijados de antemano; el smoke no elimina ninguno y no hay selección de arquitectura por el score final. Una corrida no finita, incompleta o con desvío de contrato permanece en la tabla con su motivo; no se sustituye por otra seed después de ver resultados.

Por seed, calcular \(r_s=\log(L2_{\mathrm{RBF},s}/L2_{\mathrm{literatura},s})\), reportar los cinco valores por brazo, diferencia pareada media e intervalo bootstrap pareado del 95% sobre seeds (10,000 remuestreos). El margen de no inferioridad del 10% (razón RBF/literatura ≤1.10) es una propuesta de decisión interna que requiere firma antes de Tier2; no es un umbral universal ni SOTA. Con el margen firmado: RBF pasa no inferioridad si el límite superior unilateral del 95% es ≤1.10; se falsifica si el límite inferior es >1.10; si el intervalo cruza el margen, resultado inconcluso. Superioridad requiere límite superior <1.00. Cinco seeds dan evidencia de piloto, no una afirmación SOTA.

**Falsación:** la hipótesis “RBF-PIKAN es competitivo con el comparador literario dentro del presupuesto isocosto” se falsifica para un PDE si el límite inferior unilateral de la razón RBF/control excede el margen preregistrado, o si hay fallos repetidos del RBF mientras el comparador completa válidamente el contrato. Si el intervalo cruza el margen, no se acepta ni se falsifica. Superioridad se falsifica si el intervalo incluye razones ≥1.00. Si auditoría detecta diferente PDE, pool, precisión, referencia o scoring, no existe comparación válida hasta reentrenar bajo un contrato único.

## Artefactos para reutilizar y preparación faltante

**Reutilizar sin alterar**

- [Definición protegida Heat](../../../evals/problems/heat_1d/problem.yaml) y [definición protegida Burgers](../../../evals/problems/burgers_pinnacle_1d/problem.yaml): PDE, condiciones, referencias y grids held-out.
- [Manifiesto EXP-0006](../../../experiments/EXP-0006-burgers-sobol-rad/manifest.json) y [decisiones de parámetros](../../../experiments/EXP-0006-burgers-sobol-rad/parameter-decisions.md), sujetos a la auditoría de seed/pool/código anterior.
- Bundles Heat RBF/B-spline ya medidos y resultados exploratorios del [plan](plan.md), solo si pasan la auditoría semántica; de otro modo quedan como contexto descriptivo.
- Protocolos y caveats publicados de DeepXDE y PINNacle en [matriz de literatura](literature-heat-burgers.md).

**Falta preparar antes de solicitar ejecución**

- Fichas de auditoría por cohorte con hashes, seed, puntos, loss/weights, precisión, optimizador, updates, referencia/grid/métrica, predicciones/checkpoints y GPU-s real.
- Adaptador DeepXDE Heat y MLP PINNacle Burgers fijados por commit/versión; runner que consuma pools compartidos y puntúe con referencia AF sin tocar evals/.
- Pools deterministas, grids de desarrollo con referencias independientes del held-out, y exportador de predicciones/checkpoints para ambas mallas.
- Manifiesto por corrida, entorno congelado, parámetros y reglas de censura; logger de GPU-s por dispositivo, wall time, memoria, hardware/driver, updates y checkpoints.
- Scorecard por seed, cálculo reproducible de secundarias e intervalo pareado; conservar fallos y corridas incompletas.
- Preregistro con hipótesis, margen estadístico, seeds, caps, contratos y falsación, firmado por el responsable; aprobación presupuestaria para Tier2. La reproducción fiel tiene presupuesto separado. Este documento no concede ninguna autorización.

No enviar GPU hasta completar los artefactos, revisar preregistro y recibir la aprobación presupuestaria correspondiente.
