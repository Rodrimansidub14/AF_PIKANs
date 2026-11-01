# Decisiones de parámetros — Heat + Burgers, primera campaña

Petición humana: ejecutar Heat y Burgers en una misma campaña; registrar en
W&B y documentar toda decisión. No se declara superioridad ni se congela baseline.
Presupuesto v2 solicitado antes del envío: tres semillas estiman dispersión inicial,
sin afirmar potencia estadística confirmatoria ni comparabilidad directa con SOTA.

| Parámetro | Valor | Justificación y origen |
|---|---|---|
| Problemas | Heat primero, Burgers después; redes independientes | Petición humana; Heat verifica recorrido y Burgers aporta frente no lineal. |
| Heat | alpha=0.1, x/t en [0,1] | Contrato existente; IC sin(pi*x), fronteras cero. |
| Burgers | nu=.02, c=0, A=.5, x0=-.25; x[-1,1], t[0,1] | Primera configuración del protocolo, onda estacionaria exacta; no entrenar con verdad interior. |
| Bases | 11 gaussianas por arista, centros [-1.05,1.05], ancho .21 | Geometría fija; ancho igual a separación. Elección de control, no óptimo demostrado. |
| Red | [2,16,16,1], SiLU por arista, sin sesgos, tanh ocultas | Continuidad con diseño del proyecto; 3648 parámetros por problema. |
| Inicialización | Normal, std=0.1/sqrt(entradas), pesos base y RBF | Elección conservadora para limitar saturación inicial; se medirá, no se presume óptima. |
| Coordenadas | Mapeo afín a [-1,1] antes de capas | Derivadas calculadas respecto a coordenadas físicas; no LayerNorm de entrada. |
| Condiciones Heat | sin(pi*x)+t*x*(1-x)*N | Impone IC y ambas fronteras exactamente. |
| Condiciones Burgers | Penalizaciones separadas IC, izquierda, derecha | Trazas exactas; cada término se promedia, peso fijo 1. |
| Normalización | Heat residual/1.1; Burgers residual/.76 y condiciones/.5 | Escalas U/T+alpha U/L² o U/T+U²/L+nu U/L²; U_heat=1,U_burgers=.5,L=T=1. |
| Precisión | float64 | Control inicial de derivadas segundas; registrar costo GPU, no presumir más rápida. |
| Muestreo | Sobol aleatorizado con scramble, fijo | Cobertura cuasi-Monte Carlo; alternativa fundada a iid sin adaptación. |
| Pool | Heat4096/Burgers8192 interiores, 512 por frontera, 1024 iniciales | Potencias de 2; Heat es prueba suave, Burgers conserva mayor cobertura del frente. |
| Semillas | 101,102,103; semillas Sobol=10*semilla+offset, offsets1/2/3/4 | Evita colisiones de streams entre semillas consecutivas. Pools exactos guardados. Tres repeticiones son piloto. |
| Diagnóstico | Pool hasta 2048; observar primeros 256; sensibilidad en 32 | Recorte explícito para costo; prefijos potencia de 2. |
| Batch | 1024 por condición según PINA common_batch_size | Compromiso inicial memoria/ruido; verificar comportamiento real y registrar pasos. |
| Optimizador | Adam lr=.001; betas=.9,.999; eps=1e-8; sin decay | Defaults explícitos de Adam, elección inicial común; sin scheduler ni L-BFGS en esta versión. |
| Presupuesto | Heat5000 pasos/300s; Burgers20000 pasos/1200s por semilla | Lo que ocurra antes. 75min entrenamiento/diagnósticos máximo, seis runs. |
| Timeout exterior | 2100s por semilla, 900s instalación | 105min runtime+15min instalación; cola del proveedor fuera de control. Tiempo límite conserva pasos reales; no implica convergencia. |
| Diagnósticos | Paso0, cada250 y final | Reducir costo sobre presupuesto ampliado; registrar tiempo diagnóstico por separado. |
| Saturación | abs(tanh)>0.99 | Definición descriptiva, no umbral de aceptación. |
| Sensibilidad | Norma L2 de du/dtheta por punto, 32 puntos | Primera señal física; no se interpreta como causalidad ni localización probada. |
| Validación | Malla independiente 129x65, cuadratura trapezoidal | V exploratorio; E no se abre. Error integral final requiere refinamiento. |
| Visualización | Plotly HTML autocontenido | Heat y Burgers son mapas 2D; PyVista se reserva para geometrías/mallas. |
| W&B | Un run por problema, group común, artefactos y copia local | Offline durante el cálculo; el ZIP conserva los runs y artefactos. `fetch` los sincroniza con `wandb sync` usando `WANDB_API_KEY` del `.env` solo en memoria local. |
| CPU local | 2 hilos; prueba de 3 pasos, pool64, batch32, BC32 | Solo integración antes de gastar GPU; no son resultados del baseline. |
| Determinismo GPU | Lightning deterministic=True, CUBLAS_WORKSPACE_CONFIG=:4096:8 | Requisito de cuBLAS; no garantiza identidad bit a bit entre hardware/versiones. |
| Dependencias | Pins en manifest; versiones transitivas guardadas por corrida | Torch GPU lo proporciona Kaggle; el entorno local usa torch CPU. Comparar hardware/versiones antes de claims. |

Corrección de integración previa al baseline: CartesianDomain de PINA ordena
variables alfabéticamente (t,x); se fija explícitamente (x,t), el orden físico del
modelo y los arrays guardados. La comparación PINA/autograd independiente detectó
la diferencia; los checkpoints previos de tres pasos son pruebas inválidas para
física y NO se usarán como evidencia. Añadida prueba de regresión.
SciPy remoto se fija en 1.15.3 para soportar Python 3.11 de runtimes habituales;
el entorno local resolvió inicialmente 1.18.1. Los puntos exactos se guardan para
evitar atribuir reproducibilidad únicamente a la semilla entre versiones.

El presupuesto v1 de 2000pasos/900s/semilla101 quedó sustituido ANTES de enviar a
Kaggle por petición humana de equilibrar precisión y tiempo (<8h). El nuevo
presupuesto no copia un benchmark: Free-RBF-KAN usa K=50, 4000interiores,
200/frontera y 15000épocas con arquitectura [2,5,5,1]/30bases en su problema
térmico, distinto de nuestro Heat. Épocas full-batch no equivalen a nuestros
pasos minibatch. La comparación futura exigirá PDE, precisión, hardware y costo
comunes, además de curvas error/tiempo y repeticiones. No se ajusta presupuesto
observando la evaluación final E.

Fuentes: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html
(usar 2^m puntos, no truncar arbitrariamente); https://arxiv.org/abs/2207.10289
(alternativas de muestreo y RAD/RAR-D); https://arxiv.org/abs/2601.07760
(RBF fija frente a geometría aprendible); https://arxiv.org/abs/2402.01868
(optimización como factor de confusión); https://pina-org.github.io/PINA/_rst/trainer.html
(semántica de batching). Estos trabajos motivan decisiones, no garantizan SOTA.

Mediciones implementadas inicialmente: pérdidas PINA, norma de gradiente global,
ocupación de bases, activaciones, saturación, coeficientes, derivadas físicas,
residual diagnóstico, sensibilidad, tiempo y memoria. Conflictos de gradiente,
rango efectivo y norma de actualización quedan pendientes; no se reportarán como medidos.

Cambios de parámetros requieren versión nueva de manifiesto/paquete. No modificar
umbrales después de observar resultados. Registro científico humano pendiente.


## Addenda de instrumentación y ejecución — iteración heat-burgers-sobol-002-t4x2

La configuración física y los parámetros científicos anteriores permanecen iguales: mismos pools Sobol, semillas, modelo RBF fijo, pérdida, pasos y límites de tiempo. Este paquete añade instrumentación medible, ejecución en dos GPUs y presentación reproducible; no es un candidato SOTA.

| Decisión | Configuración | Qué cuantifica y por qué |
|---|---|---|
| Acelerador | Kaggle `machine_shape=NvidiaTeslaT4`, que reserva dos T4 | Preflight exige dos dispositivos T4 visibles. Una semilla por GPU; cada worker ve un único dispositivo lógico. |
| Scheduler | Máximo 2 semillas simultáneas; orden 101,102,103 | 101 y 102 ocupan GPU0/GPU1; 103 entra al primer slot liberado. Un worker ejecuta Heat y después Burgers. |
| PINA Trainer | `devices=1` en cada proceso | Evita DDP y contención; los dos procesos independientes explotan las dos T4. |
| Curvas | Historial y `run.log` cada 100 pasos; también en hitos de diagnóstico cada 250 | Reduce I/O/W&B sin alterar batches, gradientes ni optimizer steps; cada muestra conserva el step exacto. |
| Checkpoints | Solo checkpoint final de cada problema; NPZ diagnóstico cada 250 pasos | Evita cientos de `.pt` redundantes. El estado final se conserva para evaluación. |
| Muestreo medido | Conteos por dominio, discrepancia centrada CD, distancias NN mínima/mediana/P95/máxima y bounds | Recalculado sobre los pools exactos generados; coordenadas normalizadas para comparar Heat y Burgers. |
| Señales de entrenamiento | Loss total/condiciones, step, wall time, examples seen, throughput, norma global/per-capa de gradiente y de parámetros | Valores observados en callback; guardados cada100 pasos y al hit diagnostico. |
| Autodiferenciación | `u`, `u_x`, `u_t`, `u_xx`, residual PDE; media, RMS, P95 abs y máximo abs | Directo en pool diagnóstico independiente de la cuadrícula de validación. El residual también se guarda normalizado por escala PDE. |
| RBF por capa | Ocupación/bases <0.01/entropía efectiva, norma de coeficientes y base, saturación de tanh | Mide uso de bases y comportamiento de cada capa en los mismos puntos diagnósticos; no atribuye causalidad. |
| Activaciones y sensibilidad | Media/RMS/P95/máximo de activaciones por capa; norma L2 de `du/dtheta` sobre 32 puntos | Señales internas cuantitativas; geometría RBF y puntos se conservan para reproducirlas. |
| Hardware | Modelo GPU, slot físico asignado, memoria pico del proceso y `nvidia-smi` cada10s | La utilización/memoria/potencia/temperatura se guarda en CSV. Telemetría de GPU se identifica a nivel de worker-semilla, que comprende ambos problemas secuenciales. |
| Validación | Malla fija de 129×65; métricas existentes y NPZ por problema/semilla | Incluye error de campo y residual recomputado por autograd; no abre la evaluación final E. |
| Presentación | Plotly para muestreo, pérdidas, señales y campos; HTML ligero con Plotly CDN | Produce visualizaciones visibles en la libreta y archivos en `analysis/`. |
| Análisis posterior | Panel + Plotly; PyVista 0.49.0 opcional para VTK off-screen | La app está en el repo. Heat/Burgers no contienen geometría VTK, así que PyVista no se ejecuta en esta campaña; será aplicable a DFG/NS. |
| W&B | Un run por problema/semilla; modo offline, tabla de señales, métricas y artefactos | El ZIP incluye los datos offline y archivos de artefacto; `fetch` verifica sus hashes y los sincroniza localmente con la clave del `.env`. No se añade ninguna clave al bundle de Kaggle ni al repositorio. |
| Dependencias notebook | PINA, W&B, Plotly 7.1.0, SciPy 1.15.3 y pandas 2.2.3; pins registrados | Versiones y hardware se escriben en cada seed run. PyVista/Panel no aumentan el tiempo de entrenamiento. |
| Salida | CSV cuantitativos, distribuciones descriptivas, Plotly HTML, ZIP y hashes SHA-256 | El ZIP incluye artefactos y registros offline de W&B; el índice SHA-256 cubre sus archivos. Excluye solo rutas de credenciales conocidas; la clave nunca entra al notebook. |

Las filas de resultados son mediciones directas con unidades explícitas en nombre de columna cuando aplica. Un valor ausente permanece ausente; no se rellena como cero. Tres semillas describen dispersión exploratoria, no potencia confirmatoria. Los umbrales siguen sin definir, el gate no acepta ciencia y ninguna afirmación de superioridad se deriva de esta corrida.
