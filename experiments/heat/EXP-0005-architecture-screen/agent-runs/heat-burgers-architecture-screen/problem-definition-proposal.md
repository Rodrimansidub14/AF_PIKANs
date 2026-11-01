# Definición física y contrato — EXP-0005 Heat + Burgers

**Estado:** contratos versionados e implementados; EXP-0005 completó el screening de Heat y Burgers canónico con tres arquitecturas y tres semillas. Evidencia exploratoria, sin aceptación científica ni claim SOTA.

## Heat 1D: conducción en una varilla

Heat representa conducción transitoria en una varilla homogénea, unidimensional y sin generación interna. Se usa el modo fundamental de la ecuación:

\[
u_t = \alpha u_{xx},\quad x\in[0,1],\ t\in[0,1],\quad \alpha=0.1,
\]
\[
u(x,0)=\sin(\pi x),\qquad u(0,t)=u(1,t)=0.
\]

La solución de referencia es exacta:

\[
u(x,t)=\sin(\pi x)\exp(-\alpha\pi^2t).
\]

La interpretación es **conducción sólida**, con los extremos fijados a la temperatura de referencia. Es un caso adimensional de prueba suave; no incluye advección, movimiento de material, convección con un fluido ni geometría transversal. La ecuación de calor en una varilla, sus condiciones en los extremos y la expansión en modos seno se derivan por separación de variables en el material docente de MIT OCW: <https://math.mit.edu/classes/18.085/summer2014/pset/pset8_sol.pdf>.

## Burgers 1D: benchmark PINN con formación de frente

El manifiesto EXP-0002 usa otro caso válido: onda estacionaria exacta de tipo tanh, con \(\nu=0.02\), amplitud \(0.5\), centro \(-0.25\) y fronteras no nulas. Se conserva como evidencia histórica bajo su propio contrato; no se reinterpretará como el benchmark estándar de Raissi.

Para el siguiente screening propongo la definición común del benchmark Burgers 1D de Raissi et al. y PINNacle:

\[
u_t+u u_x-\nu u_{xx}=0,
\quad (x,t)\in[-1,1]\times[0,1],
\quad \nu=0.01/\pi,
\]
\[
u(x,0)=-\sin(\pi x),\qquad u(-1,t)=u(1,t)=0.
\]

Este es un modelo escalar de transporte no lineal con difusión viscosa: sirve de puente conceptual hacia flujo y Navier–Stokes, pero no es un modelo vectorial incomprensible ni contiene presión o geometría de fluido. El problema clásico se atribuye a Raissi et al.; PINNacle (NeurIPS 2024) publica la misma ecuación, dominio, condiciones y viscosidad como su tarea `Burgers1d`:

- Raissi, Perdikaris y Karniadakis, *Physics-informed neural networks*, JCP 2019, DOI: <https://doi.org/10.1016/j.jcp.2018.10.045>.
- Hao et al., *PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs*, NeurIPS 2024, apéndice de definición: <https://papers.nips.cc/paper_files/paper/2024/file/8c63299fb2820ef41cb05e2ff11836f5-Paper-Datasets_and_Benchmarks_Track.pdf>.

Para esa condición inicial la solución de referencia se evalúa mediante Cole–Hopf, con la integral gaussiana aproximada por cuadratura Gauss–Hermite de orden 128. La prueba de referencia compara órdenes 96 y 128 y verifica la ecuación, la condición inicial y las fronteras. **No** se usa la fórmula tanh del shock estacionario como referencia para este nuevo problema. La región del frente, si se reporta, se fija antes del entrenamiento a partir de la definición y se mide además del error global.

## Consecuencias para el diseño experimental

- La campaña anterior se conserva como `EXP-0002`, con su onda estacionaria Burgers y sus resultados ya registrados. No se reescribe ni se renombra.
- El nuevo screening comparte exactamente Heat y cambia la definición de Burgers a la forma PINN estándar; por tanto, sus números de Burgers no son una repetición directa de EXP-0002.
- La nueva referencia se identifica por separado como `burgers_pinnacle_1d` en el evaluador. Así, los checkpoints del Burgers estacionario antiguo y los del benchmark transitorio nuevo conservan contratos distintos.
- Se mantendrán los pools Sobol scrambled fijos y pareados entre arquitecturas sobre el dominio espacio-tiempo. El cambio de muestreo queda para una intervención posterior.
- Si se comparan RBF-PIKAN, PINN-MLP y B-spline PIKAN, deben entrenarse bajo esa misma definición, presupuesto, precisión y pools. Los tres seeds del screening solo describen evidencia exploratoria.
- La viscosidad estándar \(0.01/\pi\) produce un frente más estrecho que \(\nu=0.02\). La verificación local de Cole–Hopf y el smoke CPU de las seis combinaciones arquitectura–problema ya pasaron; el presupuesto no se aumenta tras mirar resultados de validación.
- Heat y Burgers se reportan por separado. El resultado no prueba SOTA ni operator learning.

## Implementación y verificación del contrato

El problema Burgers canónico dispone de un evaluador versionado y una referencia independiente bajo `evals/problems/burgers_pinnacle_1d/`; el checkpoint declara ese identificador, mientras que `burgers` sigue apuntando al shock histórico. Las pruebas verifican la referencia de Cole–Hopf, límites, simetría, convergencia de cuadratura y residual PDE. El smoke PINA local entrenó las tres arquitecturas en Heat y Burgers y obtuvo métricas finitas.

El pre-registro EXP-0005 vuelve a entrenar las tres arquitecturas —RBF-PIKAN fija, MLP PINN y PIKAN B-spline— en ambos problemas con pools Sobol pareados. Esto es necesario porque el Burgers transitorio estándar no coincide con EXP-0002. Tres semillas siguen siendo screening, no evidencia confirmatoria.

## Resultado del screening

EXP-0005 terminó sus 18 runs y sincronizó 18/18 con W&B. En las tres semillas, B-spline obtuvo el menor error relativo L2 de Heat (`2.30e-5 ± 9.49e-6`); RBF obtuvo el menor error relativo L2 de Burgers (`0.0540 ± 0.0393`) y el menor RMSE normalizado del frente (`0.292 ± 0.229`). Estos son descriptivos por problema y muestran dispersión; no establecen una arquitectura ganadora general. El verificador dejó `thresholds_pass=false` porque no se habían fijado umbrales numéricos de aceptación antes del screening.

El [reporte interactivo](analysis/architecture-screen-comparison.html), el [notebook de comparación](../../../notebooks/experiments/heat-burgers-architecture-screen-comparison.ipynb), la [tabla por semilla](analysis/seed-results.csv) y las [señales cuantificadas](analysis/quantified-signals.csv) guardan figuras, mediciones y procedencia. La tabla completa contiene 2.622 observaciones señal-semilla de 354 métricas escalares. La siguiente decisión debe usar esta evidencia para formular y preregistrar una intervención; los contratos Heat y Burgers aquí definidos permanecen congelados.

## Referencias canónicas

- Heat: derivación de conducción en una varilla y modos seno por separación de variables, [MIT OpenCourseWare, 18.085, Problem Set 8 solutions](https://math.mit.edu/classes/18.085/summer2014/pset/pset8_sol.pdf).
- Burgers: formulación y benchmark PINN, Raissi, Perdikaris y Karniadakis, [*Physics-informed neural networks*, Journal of Computational Physics (2019)](https://doi.org/10.1016/j.jcp.2018.10.045).
- Contrato Burgers reproducido por un benchmark contemporáneo: Hao et al., [PINNacle, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/8c63299fb2820ef41cb05e2ff11836f5-Paper-Datasets_and_Benchmarks_Track.pdf).
