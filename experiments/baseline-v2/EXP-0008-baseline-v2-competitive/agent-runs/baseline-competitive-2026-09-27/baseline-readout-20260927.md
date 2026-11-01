# Baseline readout: cuatro problemas y comparación con literatura

> Snapshot histórico del 2026-09-27. El estado ejecutado y las métricas vigentes están en competitive-baseline-execution.md y competitive-baseline-completion-plan.md.

Fecha de corte: 2026-09-27. `KAGGLE_API_TOKEN` se leyó desde el `.env` local y la autenticación del CLI quedó comprobada; el secreto no se copia a artefactos. Cuota observada: 26.95 GPU-h disponibles hasta 2026-10-03.

Tope autorizado para el cierre: **8 horas de cuota Kaggle en total**, con kernels configurados para **2×T4**; se reservan al menos 18.95 horas de la cuota actual. Kaggle informa que una hora de sesión 2×T4 consume una hora de cuota, no dos ([anuncio de Kaggle](https://www.kaggle.com/product-feedback/361104)). Los límites se aplicarán como suma de tiempo de sesión con GPU habilitada, incluyendo instalación y arranque.

## Resultado principal

Los datos actuales **no alcanzan ni demuestran un baseline competitivo SOTA en los cuatro problemas**. Sirven como baselines exploratorios y para localizar fallos, pero faltan comparadores fuertes ejecutados bajo el mismo contrato. `scientific_accepted=false` permanece sin cambios.

| Problema | Resultado AF disponible | Ancla publicada y comparabilidad | Juicio actual |
|---|---|---|---|
| Heat 1D | RBF + Sobol, n=5 descriptivo combinado de dos bundles: L2 relativo `1.98294e-4 ± 2.38199e-4`, referencia analítica en malla `512×256`. B-spline local n=3: `2.30142e-5 ± 9.52085e-6`, con ~301 s frente a ~191 s de RBF; cohortes distintas. | El tutorial de DeepXDE ofrece un PINN reproducible para la misma ecuación. El número de HyPINO no es isocontrato: usa preentrenamiento supervisado y un operador de 77 M de parámetros. | No hay aún una comparación publicada isocontrato. B-spline es un buen control local, no evidencia de SOTA. |
| Burgers transitorio | RBF + Sobol, n=5: L2 relativo `0.0395498 ± 0.0256492`, referencia Cole–Hopf. La campaña GPU v4 también contiene cinco corridas RAD, que no mejoran el resultado del control en esta configuración. | PINNacle Vanilla publica L2 relativo `1.45e-2 ± 1.59e-3` (n=3), y L-BFGS `1.33e-2 ± 5.30e-5`; difieren el contrato y el presupuesto. El Vanilla aún no se ha reproducido aquí. | La media AF es ~2.73× el error publicado de Vanilla; comparación contextual, no directa. No es competitivo demostrado. |
| Allen–Cahn 1D | RBF + Sobol, n=5 previo: L2 relativo `0.0184323 ± 0.0033758` frente a la referencia DeepXDE `201×101`; sigue provisional. El kernel T4 r3 terminó semillas 104–105, pero su CSV no contiene error de campo; solo permite usar señales físicas/diagnósticas. | Hay métodos publicados con error menor, pero la referencia, la malla, las condiciones y el presupuesto no están igualados. La candidata BDF pasa convergencia numérica preliminar, pero no está aceptada como referencia científica. | No hay reproducción comparable; la corrida T4 nueva no cierra el score de campo. |
| Navier–Stokes DFG 2D-1, Re=20 | RBF + Sobol, cinco semillas GPU (301–305): `Cd=1.075206 ± 0.0945853`, `Cl=-0.0465207 ± 0.0692572` [CORREGIDO 2026-09-30: fórmula de fuerza σ·n validada sobre el campo FEM L3; la cifra anterior usaba la fórmula tangencial con error de signo. Ver local-20260930/README.md] (valor correcto σ·n: Cd 3.9674 ± 0.1796, Cl −0.0260 ± 0.1248), `Δp=0.0776816 ± 0.00553970`. | FEATFLOW publica `Cd=5.579535`, `Cl=0.0106189`, `Δp=0.117520`. Error relativo en Cd: 80.7% (corregido: 28.9%); en Δp: 33.9%. Para Cl, informar error absoluto `0.0571397`; el error relativo es inestable porque la referencia está cerca de cero. No existe aún campo FEM independiente convergido. | No es comparable al valor de referencia en QoIs y no puede llamarse SOTA. Kaggle r4 falló al importar PyVista para visualización (`IPython.core.guarded_eval` no disponible), después del entrenamiento; el manifiesto y los `metrics.json` confirman 303–305 completadas. |

Las cifras publicadas se resumen con sus límites en [literatura Heat/Burgers](literature-heat-burgers.md) y [literatura Allen–Cahn/DFG](literature-ac-dfg.md). Las semillas, no los puntos de una malla, son las unidades para la dispersión reportada. Las cifras exploratorias agregadas entre bundles no equivalen a una confirmación preregistrada.

## Diagnóstico exploratorio para el diseño posterior

| Problema y región | Señal observada | Qué permite decir |
|---|---:|---|
| Burgers, frente / complemento, n=5 | RMSE de campo `0.2161 ± 0.1443` / `0.007408 ± 0.004457` (29×); RMSE de residuo `0.2203 ± 0.1353` / `0.02323 ± 0.00921` (9.5×). | El error se concentra en el frente bajo este contrato. Es una correlación espacial descriptiva, no identifica la causa ni demuestra que muestrear adaptativamente lo resuelva. |
| Allen–Cahn, semillas T4 104–105 | Cambio relativo de energía `−0.8806 ± 0.0027`, cero incrementos observados; RMS del residuo normalizado held-out `0.001560 ± 0.000501`. | La energía decrece en esas dos trayectorias; no hay métrica de campo en el CSV descargado, así que no prueba exactitud frente a la referencia. |
| DFG, global / cerca del cilindro, n=5 | Continuidad RMS `0.03773 ± 0.00501` / `0.11079 ± 0.01649` (2.9×); momentum-x `0.02688 ± 0.00439` / `0.08765 ± 0.01557` (3.3×); momentum-y `0.03192 ± 0.00512` / `0.11097 ± 0.02181` (3.5×). No-slip RMS normalizado `0.02065 ± 0.00694`. | Los residuos son mayores cerca del cilindro; esto localiza una zona difícil, pero no prueba qué mecanismo la causa. Primero hay que descartar errores de QoI, outlet, presión y balance. |

La variación alta de Burgers/Heat entre semillas exige informar cada corrida y curvas error–GPU-s. Estas señales pueden orientar hipótesis futuras; el error de referencia solo se usa para evaluar la señal y no debe entrar en la política de puntos durante el entrenamiento.

## Ejecución remota comprobada

- El CLI autentica usando el token local. `kaggle kernels list --mine` y `kaggle quota` responden; la cuota actual es 26.95 GPU-h.
- Heat/Allen–Cahn r3 y Burgers v4 están `COMPLETE`; los artefactos indican Tesla T4.
- DFG r4 muestra estado de kernel `ERROR`, pero `runs/campaign.json` marca `303`, `304` y `305` como completadas, cada una con 1,200 s de entrenamiento y `metrics.json`. El log identifica el error posterior al entrenamiento: PyVista intenta importar `IPython.core.guarded_eval`, módulo que no está disponible en ese entorno.
- No se reintentaron corridas terminadas ni se envió un bundle nuevo en esta comprobación. No se ejecutaron tests, preflights ni validadores locales.

## Qué falta para un baseline competitivo

1. Ejecutar el comparador DeepXDE de Heat y el PINNacle Vanilla de Burgers con cinco semillas, referencia, malla y límites de GPU declarados de antemano; conservar el B-spline como control local.
2. Cerrar la referencia de Allen–Cahn y registrar métricas de campo, interfaz, energía y residual para el mismo set T4 antes de comparar métodos.
3. En DFG, conservar FEATFLOW como ancla de QoIs, auditar balance/outlet y completar un comparador fuerte con el mismo protocolo. Sin campo FEM convergido, no reportar error de campo.
4. Comparar error frente a pasos y GPU-s, incluir todos los fallos y semillas, y usar un mínimo de cinco pares para cada finalista. Mantener `scientific_accepted=false` hasta revisión humana.

El plan y las propuestas de campaña están en [plan.md](plan.md), [bundles/heat](bundles/heat/README.md) y [bundles/burgers](bundles/burgers/README.md). La política del proyecto exige que el investigador apruebe el pre-registro y el bundle exacto antes de enviar una campaña GPU nueva.
