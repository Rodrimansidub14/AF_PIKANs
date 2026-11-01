# Decisiones de parámetros — screening Heat + Burgers canónico

**Estado:** exploratorio; tres semillas piloto; no establece SOTA ni acepta un claim.
**Modelo de este paquete:** PINN MLP tanh (58 units) (`mlp_tanh`).

| Decisión | Configuración | Justificación |
|---|---|---|
| Heat | Conducción 1D en varilla homogénea: `u_t=0.1 u_xx`, x,t en [0,1], IC `sin(pi*x)`, extremos a cero. | Modo seno fundamental con solución exacta `sin(pi*x)*exp(-0.1*pi^2*t)`; control suave y analítico. |
| Burgers | `u_t + u*u_x - nu*u_xx=0`, dominio [-1,1]×[0,1], `nu=0.01/pi`, IC `-sin(pi*x)`, fronteras cero. | Benchmark PINN estándar Raissi 2019 / Burgers1d de PINNacle NeurIPS 2024. Es transporte no lineal escalar viscoso, puente hacia NS; no incluye presión ni geometría. |
| Referencia Burgers | Transformación Cole–Hopf; convolución gaussiana por Gauss–Hermite orden128; evaluación independiente en 1025×129. | La variante canónica no tiene la referencia tanh estacionaria de EXP-0002. Orden96 vs128 difirió ≤3.2e-15 en la grilla de refinamiento documentada; IC/fronteras verificadas. |
| Región del frente | `abs(x)<=5*nu` y `t>=1/pi`. | Métrica secundaria definida por el proyecto: frente centrado después del tiempo de empinamiento invíscido, dentro de cinco escalas viscosas. No es una métrica oficial de PINNacle. |
| Geometría y muestreo | Scrambled Sobol fijo; mismos streams/pools, semillas y conteos que EXP-0002. | Aísla arquitectura. Se verificaron arrays idénticos para interior, diagnóstico, fronteras e IC en seeds 101–103. No se compara el algoritmo de muestreo aún. |
| Arquitectura | PINN MLP tanh (58 units); [2, 58, 58, 1]; `3655` parámetros. | Frente a RBF-PIKAN fija de 3.648 parámetros; la MLP tiene 3.655 (diferencia <0.2%). Geometrías B-spline/RBF fijas. |
| Activaciones/base | Hidden tanh; SiLU en camino base de KAN; salida lineal. | Igual función de salida y condiciones físicas para todos los modelos. Heat impone IC y Dirichlet exactamente con `sin(pi*x)+t*x*(1-x)*N`. |
| Escalas físicas | Heat residual 1.1; Burgers residual `2+nu`; amplitud de campo 1. | Escalas explícitas con `U=T=L=1` en Burgers; misma normalización entre modelos. |
| Precisión y optimización | float64, PINA/PyTorch, Adam lr=0.001, sin scheduler ni L-BFGS. | Conserva el protocolo de derivadas segundas del baseline; comparación de una sola variable principal. |
| Presupuesto | Heat: 4.096 interiores, 5.000 pasos/300 s máximo. Burgers: 8.192 interiores, 20.000 pasos/1.200 s máximo. Batch 1.024. | Mismos límites para cada arquitectura; se reportan pasos y tiempo reales y cuál límite terminó el run. Tres semillas son screening. |
| Diagnóstico | Sobol independiente, 256 puntos por snapshot cada250 pasos; sensibilidad `du/dtheta` en32 puntos; métricas cada100 pasos. | Instrumenta muestreo, pérdidas, activaciones/bases, gradientes, derivadas, residual, sensibilidad, hardware y costo sin adaptar puntos. |
| Semillas y cómputo | 101,102,103; dos Tesla T4; máximo dos semillas concurrentes; un proceso por GPU. | Pares por semilla entre arquitecturas; cada proceso entrena Heat y luego Burgers. Timeout exterior 2.100 s por semilla. |
| W&B y artefactos | Un run offline por problema/semilla, grupo compartido `heat-burgers-architecture-screen-EXP-0005`; CSV/JSON/NPZ/Plotly/HTML y notebook reproducible. | La clave W&B solo se usa al sincronizar desde el entorno local; no se incluye en notebook, manifiesto o bundle. |
| Escala visual | Pérdidas, gradientes, residuales y error absoluto positivo en escala log; solución firmada lineal; ceros del error sin color en el mapa log. | Permite leer magnitudes separadas por órdenes sin cambiar los valores registrados ni añadir un epsilon artificial. |
| Estado estadístico | Sin umbrales después de resultados; evaluación final bloqueada. | Informa las tres observaciones y su dispersión descriptiva; no infiere significancia, causalidad ni SOTA. |

**Fuentes:** Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*, JCP 2019, DOI 10.1016/j.jcp.2018.10.045; Hao et al., *PINNacle*, NeurIPS 2024, apéndice Burgers1d; separación de variables de la conducción en una varilla, MIT OCW. Los enlaces y el contrato están en `problem-definition-proposal.md`.
