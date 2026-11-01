# Handoff — EXP-0028 y diseño del mecanismo para piezo (2026-10-25, noche)

Para quien continúe en una sesión nueva. Leer esto primero, luego `experiments/mechanisms/EXP-0028-failure-attribution/` (todas las rutas de abajo son relativas a esa carpeta salvo indicación). Idioma de trabajo: español.

## 1. Qué es el proyecto y dónde estamos
Tesis (UVG) sobre mecanismos de precondicionamiento para PINN con KAN de bases RBF (PIKAN), con piezoelectricidad (bimorfo PVDF del paper González et al. 2026, MCA 31:214) y Navier–Stokes como problemas finales. Tras la reunión del 23-oct se reorientó a: explicar por qué un mecanismo funciona en un problema y falla en otro, predecir para piezo antes de correrlo, y diseñar el mecanismo con datos. **Hoy:** el banco (Allen–Cahn, Burgers, Fisher–KPP, NS2d-CG, DFG) está medido; las predicciones para piezo se firmaron y se evaluaron (n = 3); se hizo el análisis de propiedades de piezo, un barrido de literatura y un diseño «araña» (cuerpo + patas con activación graduada). **Fase actual del investigador: definir hipótesis y diseñar el mecanismo; sin experimentos nuevos hasta tener especificación firmada.**

## 2. Preferencias y reglas del investigador (importantes)
- Quiere un trabajo cuidadoso y «humano», no automatizado; análisis con datos y estadística descriptiva, sin p-valores con n = 3.
- **Ahorrar uso de Claude** (plan Pro; a las 20:35 del 25-oct: 20 % de la ventana de 5 h, 42 % de la semanal; la semanal se reinicia el 31-oct 06:00 UTC). Evitar agentes grandes y sondeos; los agentes pequeños (Haiku) se usan solo para búsquedas.
- No repetir línea base ni MLP (salvo la referencia del paper en piezo, de baja prioridad). Todo en torno a PIKAN.
- Piezo debe alinearse con el paper (receta, pesos fijos, semillas 20260728-32, Adam 20 000 float64 sin L-BFGS).
- **Firmar es acto del investigador** (preregistros en `registry/proposals.jsonl`, specs, `predictions.md`). No firmar por él.
- No manejar credenciales. **El investigador pegó su id y secreto de Modal en el chat: rotarlo/revocarlo cuando termine con Modal.** `.env` (gitignored) tiene claves de Kaggle y W&B; no leerlas ni mostrarlas. W&B: offline en kernels, sincronizar en local con `.env` en memoria (hay `wandb_tools/` a medias; el agente se detuvo; no se sabe si se subió algo al proyecto `AF_PIKANS`, verificar en el panel).
- Pidió no cerrar ideas por «solo lo que más sirve»: todas las palancas se consideran y se ponderan (diseño en araña).

## 3. Resultados clave (todo registrado en `registry/proposals.jsonl`)
- **Banco, receta corta y completa** (`results/phase1*_scored.md`, `results/phase1e_all_scored.md`): E sirve o es neutro a receta completa en Allen–Cahn (0.21 al fin de Adam), KPP (0.08), NS (0.96); F es robusto en el banco corto (4 de 5) pero se diluye a receta completa; J inestable; I daña en Burgers. NS a receta completa: todo ≈ 0.95-1.0 (el efecto de la receta corta desaparece). Burgers a receta completa solo n = 2 (`bu2`, `bu3` y J, I sin correr; L-BFGS corta por tiempo: correr solo en T4). DFG fase 1e en Colab T4: se cortó por un apagón de luz; el runner es reanudable.
- **Predicción firmada** (`predictions.md`, 2026-10-25): mejora modesta para F (0.6) y E (0.5), J empeora leve, I sin predicción; criterio H-PRED-1 v1.3. **Resultado con n = 3** (`results/piezo_vs_predictions.md`): H-PRED-1 **no respaldada** en converso ni directo (mejor brazo E, peor F, no J).
- **Piezo, error absoluto l2re_u (media de 3 semillas):** converso B 0.198 (CV 0.81), F 0.236, **E 0.053**, J 0.089; directo B 0.208, F 0.803, E 0.193, J 0.181; MLP del paper 0.107 / 0.117. Pareo de la semilla 28: B y E corrieron en Kaggle T4, F y J en Modal H100.
- **Análisis de propiedades de piezo** (`ANALISIS_PIEZO_PROPIEDADES.md`, `analisis_piezo/`): bloques GN casi de rango uno (cond ≈ 1e4), dispersión de curvatura 3.3-4.4 décadas y creciente en el entrenamiento, F empeora el condicionamiento en 82-88 % de las aristas (desajuste de dirección). **Se retiró** la hipótesis de amplificación uniforme de F y la variante Fn.
- **Hipótesis H1-H5** (`HIPOTESIS_PARTES_QUE_FALTAN.md`): φ del converso (curvatura de salida 82-99 % en σ_yy), voltaje flotante del directo 20 % bajo el FEM (escalar fuera del mecanismo), saltos de pérdida del directo, lotería de B, consistencia constitutiva. Ninguna probada.
- **Literatura** (`REVISION_LITERATURA_PIEZO.md`): barrido con agentes Haiku, **sin texto completo; verificar cada referencia antes de citar**; el investigador repitió el barrido de los huecos y dice que lo encontrado está bien.
- **Diseño** (`DISENO_ARANA.md`): cuerpo E + patas P1-P6 (escalares globales en forma cerrada, balance por grupo físico, Gram empírico por arista, guarda de estabilidad, arranque denso, J ligera) + extensiones X1-X3 fuera de la receta.

## 4. Pendiente, en orden
1. **Comprobación de P1 (H2), HECHA: refutada.** `analisis_piezo/check_Vf.md`: el V̂_f aprendido ya es el mínimo exacto de la pérdida (coincide a 3 cifras en 9 corridas del directo) y sustituirlo no cambia φ, u, v ni la pérdida. El potencial 20 % bajo es el valor que minimiza la pérdida dado el resto de la red: el error nace en los demás campos o en la aproximación discreta de la carga neta. Lo que sigue en esa línea: ¿la aproximación discreta de la carga neta (256 puntos) sesga V̂_f? (comparar con la integral exacta sobre la malla guardada; sin entrenar).
2. Comprobar P2 (Jacobiano por término sobre los estados guardados) y P3 (distribución de entradas por arista), también sin entrenar.
3. Redactar capítulos de Resultados y Discusión de la tesis (`ProyectoGrad`: `j2-resultados.tex`, `j3-discusion.tex`; hay un brief en `reports/brief-resultados/`): tablas de banco, predicciones y resultado, análisis de piezo, hipótesis, literatura, diseño. Es la tarea que más usa; hacerla por partes.
4. Especificación y firma de `EXP-0029` (mecanismo en araña) cuando las comprobaciones lo justifiquen; incluir registro de la pérdida **por paso** (hoy cada 100).
5. Pendientes de cómputo: Burgers a receta completa (≈ 21 horas de T4: Colab T4 o Kaggle tras el 31), DFG de la fase 1e (reanudar el runner de Colab), semillas 20260731-32 de piezo (≈ 19 USD en Modal; el veredicto difícilmente cambia), sincronizar W&B.

## 5. Recursos
- **Kaggle:** 1.59 h de GPU de 30 hasta el 31-oct 00:00 UTC; CLI con `"C:\Program Files\Python312\python.exe" -m kaggle ...` (sesión estable; falló una vez de forma transitoria).
- **Modal:** crédito ≈ 19-20 USD (de 42; el bloque de piezo costó ≈ 22). `modal_piezo.py` (en `piezo/`) corre 4 procesos por H100; perfil `rodrimansidub14`; volumen `af-pikans-exp0028-piezo`; comando con `--detach`. Se descargan resultados con `modal volume get`.
- **Voltage Park** (H100 a 1.99 USD/h) como respaldo con `piezo/cloud_kit` y `cloud_runner.py`. **Colab:** solo T4/TPU desbloqueadas; el investigador comprará unidades si hace falta; no es camino para piezo.
- **Hallazgos de rendimiento:** piezo en T4 está limitado por float64 (0.436 s/paso converso); en H100 por lanzamientos de kernels (utilización 20 %), así que 3-4 procesos por GPU rinden ×3-4; CUDA graphs ×6 pero no integrado. La cuota de Kaggle cuenta por sesión, no por GPU.
- Tests: `tests/test_piezo_problem.py` (24), `test_residual_terms.py`, `test_gram_preconditioning.py`, `test_mechanism_isolation.py` pasan (45 en total). Python con numpy/torch: `C:\Program Files\Python312\python.exe` (no usar `-I`).

## 6. Advertencias de método
- Un error propio: al agrupar archivos por patrón, los de humo (300 pasos) contaminaron la semilla 28; el análisis actual filtra `adam_steps == 20000` y comprueba 24 corridas.
- Probar los guardados de artefactos y cualquier ruta de GPU **antes** de gastar cuota (un tensor en CPU mató dos kernels en Kaggle).
- Las referencias del barrido de literatura son por verificar; ninguna se abrió completa.
- Las estimaciones de costo con H100 en Modal usan una tarifa supuesta (~4 USD/h por GPU); verificar en la cuenta.
- No se debe afirmar que «no existe literatura» sobre H2, H5 u otros sin repetir la búsqueda con cuidado.

## 7. Índice de documentos (EXP-0028-failure-attribution/)
`spec.md` (v1.1-v1.6), `predictions.md` (firmada), `piezo/` (porte del paper: `piezo_problem.py`, `run_piezo.py`, `ALINEACION_PAPER.md`, `DIFERENCIAS.md`, `PLAN_RECURSOS.md`, `ETAPA0_RESULTADOS.md`, `RESULTADOS_ETAPA1.md`, `modal_piezo.py`, `cloud_kit/`), `analisis_piezo/`, `cruce/`, `fase0/`, `fase2/`, `results/`, `ANALISIS_PIEZO_PROPIEDADES.md`, `HIPOTESIS_PARTES_QUE_FALTAN.md`, `REVISION_LITERATURA_PIEZO.md`, `DISENO_ARANA.md` (vigente), `DISENO_MECANISMO_FINAL.md` (superado).

## Actualización 2026-10-26 (sustituye lo que contradiga arriba)
- **Decisión del investigador:** no más cómputo; se **reporta lo encontrado**. Modal sin crédito (el bloque EXP-0029 costó ≈ 11 USD más de lo que estimé; mi tarifa supuesta de A100 estaba mal). No hay apps corriendo en Modal (verificado). No se lanza nada en Voltage Park.
- **Resultados nuevos:** `results/EXP0029_vs_predicciones.md` (M frente a predicciones firmadas: no respaldada en converso; directo M ≤ E en 2 de 3; φ converso cumple), `LECCIONES_EXP0029.md` (métrica robusta, E sale antes del llano en 5/5 pares y recorta la cola mala, P5 fijo no sirvió, P2 por grupo mal planteada en el converso), `analisis_piezo/check_P2.md`, `check_P2_mezcla.md`, `results/T1_T2_resultados.md`, `DISENO_FINAL.md`, `DISENO_EJECUCION.html`.
- **Hallazgo central para la tesis:** con n = 5, la ventaja de E sobre B en el converso no está en la media del último paso (0.158 frente a 0.152) sino en salir antes del llano y en la cola (error robusto E 0.033-0.102 frente a B 0.039-0.391). Métrica robusta = post hoc, declararla así.
- **Pendiente solo de escritura:** capítulos de Resultados y Discusión (`ProyectoGrad`), por partes. Datos completos en `results/piezo-exp0029-final/` (651 MB, estados cada 1 000 pasos, pérdida por paso, bloques GN, ficha por corrida). M converso 31-32 no corrió (OOM).
- **Idea abierta, sin correr:** M2 = P5 por cambio de cuotas + P2 solo en grupos con cuota ≥ 1 %; probar con corridas cortas (8 000 pasos del converso) si algún día hay cómputo.
- **Pendiente del investigador:** revocar el token de Modal que se pegó en el chat.
