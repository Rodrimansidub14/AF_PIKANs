# Baseline exploratorio de señales RBF-PIKAN y contraste con frontera SciML

**Fecha:** 2026-09-24  
**Estado:** reporte descriptivo compilado desde corridas existentes; no es una campaña confirmatoria ni un claim SOTA.

## Qué responde

El baseline registra qué señales internas, físicas y de costo se pueden observar al entrenar la RBF-PIKAN, en qué PDE vale la pena explorar cada una, y qué métodos publicados dan contexto para esas hipótesis. Mantiene separadas tres preguntas: qué tan bien resuelve una PDE, qué mecanismos puede inspeccionar la arquitectura, y qué señales podrían guiar una focalización futura.

La RBF fija es la arquitectura de investigación principal porque sus activaciones gaussianas se pueden relacionar con centros y anchos conocidos: para un punto del dominio se puede medir la huella de cada base, la cobertura, la ocupación, la redundancia, la norma de coeficientes y la sensibilidad por zona. La ventaja interpretativa frente a otras arquitecturas es una **hipótesis por probar**, no un hecho establecido ni una afirmación de mejor precisión.

La PIKAN B-spline cúbica queda como control de base explícita: sus soportes locales también son interpretables y permiten comprobar si la geometría fija RBF aporta información distinta. El MLP tanh se usa solo como referencia contextual de la pantalla local. Tiene activaciones, gradientes y sensibilidad; simplemente no expone de forma nativa una base geométrica centro/ancho comparable.

## Evidencia local compilada

El reporte toma EXP-0005 (Heat y Burgers canónico; tres arquitecturas y tres semillas) y EXP-0003 (Allen–Cahn RBF fija; tres semillas). En la pantalla casi igualada en parámetros, el error relativo L2 medio de Heat fue `2.30e-5` para B-spline y `8.88e-5` para RBF; en Burgers fue `5.40e-2` para RBF, `1.16e-1` para MLP y `2.21e-1` para B-spline. Estos valores son descriptivos con n=3, no fijan un ganador general. De hecho, hacen patente que la elección de RBF debe apoyarse en su valor como instrumento para investigar señales y en validaciones posteriores, además de su precisión por PDE.

En Burgers, la auditoría post hoc de tres semillas encontró Spearman de `0.118–0.296` entre residual absoluto y error absoluto. El decil con residual más alto acumuló `2.60–4.84×` el error medio. Esto motiva estudiar localización y bandas del frente, pero no demuestra que una distribución RAD/RAR-D mejore el resultado.

Allen–Cahn registra residual held-out, traza de condiciones, ocupación de bases, derivadas y energía. La energía baja en las cinco muestras temporales por semilla; el error de campo sigue bloqueado porque la referencia BDF todavía requiere cerrar convergencia/regularidad. El residual y la energía no se tratan como sustitutos del error de campo.

Los resultados completos, resumen por semilla y hashes de entradas están en [`var/agent_runs/rbf-signal-baseline-20260924/README.md`](../../var/agent_runs/rbf-signal-baseline-20260924/README.md). El HTML Plotly muestra solo cifras locales comparables dentro del screening; deliberadamente no mezcla puntuaciones de papers en las mismas barras.

## Contraste con publicaciones

La publicación de vRBA de 2026 reporta `8.25e-9` de error relativo L2 para Burgers canónico con vRBA y SSBroyden; el estudio usa Fourier/periodic encodings, adaptive importance sampling, 5.000 pasos Adam y 60.000 SSBroyden. El screening RBF de AF_PIKANs mide `5.40e-2 ± 3.93e-2` en el mismo caso PDE y métrica, pero con otra arquitectura y presupuesto/protocolo. La magnitud del hueco es útil para seleccionar una reproducción; el contraste no es un resultado head-to-head ni una estimación del efecto de la arquitectura.

La misma publicación reporta `1.88e-6` en su Allen–Cahn, pero usa `k=1e-4` y condiciones espaciales periódicas. EXP-0003 usa `k=1e-3` y Dirichlet `u(±1,t)=-1`, y no tiene error de campo habilitado; no hay score comparable todavía. RAMS 2026 reporta una reducción de error relativo L2 de `0.181` a `0.010` mediante movimiento adversarial de puntos en Burgers, con 10 repeticiones. Su red, presupuesto y protocolo también difieren. Free-RBF-KAN (preprint, 2026) es una frontera relevante para explorar centros/formas adaptables y resolución de alta frecuencia, pero el benchmark de Heat de ese trabajo es 2D y distinto al control 1D suave del proyecto.

El clasificador de comparabilidad requiere coincidencia en ecuación, dominio, condiciones iniciales y de frontera, parámetros físicos, métrica, protocolo de evaluación, arquitectura, optimizador, presupuesto y puntos de entrenamiento para etiquetar un head-to-head directo. Si solo coincide la PDE, el score publicado queda como contexto y no se calcula una diferencia de rendimiento.

## Registro de señales por tipo de problema

Se mantienen instrumentadas seis familias generales: geometría de puntos, dinámica de entrenamiento, autodiferenciación, activación/ocupación de bases, resultados físicos y costo. Se adaptan así:

| Problema | Señal que debe priorizarse | Por qué cambia con la PDE |
|---|---|---|
| Heat 1D | decaimiento de modos, energía térmica, escala de `u_xx`, trazas inicial/frontera | difusión suave y pérdida de altas frecuencias permiten medir qué bases capturan modos con el tiempo |
| Burgers | posición/anchura del frente, `u_x`/`u_xx`, residual por banda, disipación viscosa | el error puede concentrarse en gradientes pronunciados y el residual puede representar el error de forma incompleta |
| Allen–Cahn | ubicación/anchura de interfaz, energía y tasa de disipación, balance reacción-difusión | un residual agregado puede ocultar una interfaz incorrecta; la referencia de campo debe estar convergida |
| DFG/NS Re=20 | continuidad, residual de momento por componente, esfuerzo de pared, presión, arrastre/elevación y longitud de estela | la validez depende de conservación y magnitudes integrales, además del error de campo |
| Poisson (alcance pendiente) | balance de flujo, suavidad/singularidad de esquina, traza de frontera | debe definirse la geometría/regularidad antes de escoger una señal prioritaria |

Inspirados por el resultado vRBA, el baseline añade como candidato general la relación señal/ruido de gradiente entre particiones y fases fitting/transición/difusión. No es parte de los resultados actuales; debe instrumentarse en una corrida posterior y su costo debe medirse.

## Qué probar para justificar la selección de RBF

Para convertir la hipótesis de interpretabilidad en evidencia hacen falta: (1) estabilidad entre semillas de huellas y ocupación de bases, (2) capacidad de esas señales para predecir error físico en mallas held-out, (3) prueba de localización frente a B-spline y referencia MLP, y (4) ablación/perturbación de las bases señaladas, verificando que el cambio de campo y residual concuerde con la atribución. Las comparaciones deben mantener PDE, muestreo, presupuesto y parámetros tan pareados como sea posible.

## Siguiente secuencia

1. Mantener RBF fija como control de señales; no cambiar arquitectura en la siguiente intervención de muestreo.
2. En Burgers, preregistrar una sola estrategia frente a Sobol —RAD/RAR-D o una reproducción específica de RAMS/vRBA—, presupuesto y métrica primaria. Medir además residual/error por zona del frente, cobertura/activación RBF, gradiente por partición, error held-out y sobrecosto de diagnóstico/adaptación.
3. Desbloquear el campo de Allen–Cahn mediante una referencia convergida y luego reportar error de interfaz y norma global junto a energía/residual.
4. Preparar DFG/NS con un FEM convergido y QoIs del benchmark antes de entrenar la RBF o elegir señales de focalización.
5. Solo después de seleccionar una señal, correr confirmación con al menos cinco semillas y, cuando proceda, reimplementar el comparador bajo el mismo presupuesto.

## Reproducción de los artefactos

```powershell
& '.\var\agent_runs\sciml-env\Scripts\python.exe' -m src.sciml.rbf_signal_baseline --repo-root . --out var/agent_runs/rbf-signal-baseline-20260924
```

El constructor lee los CSV/JSON de las corridas terminadas bajo `var/agent_runs`, calcula estadísticas a nivel de semilla, registra SHA-256, y escribe README, tablas CSV, crosswalk bibliográfico, registro de señales y visualización Plotly. No lee `evals/`, modifica graders/umbrales ni inicia entrenamiento; la salida declara `scientific_accepted=false`.

## Literatura

- Ouyang et al. (2026), [RAMS: Residual-Based Adversarial-Gradient Moving Sample Method](https://doi.org/10.1002/aidi.202500214).
- Toscano et al. (2026), [A variational framework for residual-based adaptivity in neural PDE solvers and operator learning](https://doi.org/10.1038/s44387-026-00084-4).
- Chiu et al. (2026, preprint), [Free-RBF-KAN](https://arxiv.org/abs/2601.07760).
- Hao et al. (2024), [PINNacle](https://arxiv.org/abs/2306.08827), marco de benchmark de métodos PINN y diversidad de problemas.
- Wu et al. (2023), [Non-adaptive and residual-based adaptive sampling for PINNs](https://doi.org/10.1016/j.cma.2022.115671).

Los registros detallados fueron añadidos a `PIKAN-Vault/PIKAN-Vault/sources/AF_PIKANs_Research_Brief_References_2026-09-23.bib` y `PIKAN-Vault/PIKAN-Vault/papers/Comparadores_RBF_Adaptividad_2026.md`.
