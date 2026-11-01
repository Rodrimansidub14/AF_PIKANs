# Baseline competitivo: Heat, Burgers, Allen–Cahn y DFG

Actualizado: 2026-09-28. Los cuatro problemas tienen comparadores GPU de cinco semillas. Burgers ahora reproduce la receta PINNacle Adam→L-BFGS con un score FEM casi idéntico al publicado; Heat sigue pendiente de una comparación pareada y Allen–Cahn/DFG conservan brechas de referencia. **El baseline competitivo de los cuatro problemas no está cerrado** y scientific_accepted=false.

- [Ejecución y resultados actuales](competitive-baseline-execution.md): resultados por problema, comparación con literatura, kernels y métricas descargadas.
- [Plan de cierre y brechas restantes](competitive-baseline-completion-plan.md): qué falta antes de llamar competitivo al baseline y antes de diseñar el mecanismo.
- [Scorecard actual](scorecard-current.csv): una fila por método/métrica, comparabilidad y bloqueos explícitos.
- [Plan científico y gates](plan.md).
- [Fuentes Heat/Burgers](literature-heat-burgers.md) · [fuentes Allen–Cahn/DFG](literature-ac-dfg.md).
- [Auditoría de referencias](reference-readiness.md) · [diagnóstico de corridas archivadas](diagnostic-readout.md) y [JSON](diagnostics.json).
- [Protocolo de referencia BDF/FEM](reference-protocol-ac-dfg.md) y [auditoría de gates del repositorio](repo-readiness.md).
- [Estudio de señales](signal-study.md): ganancia predictiva del error held-out sobre el residual; no define todavía una intervención.
- [Borradores y protocolos de kernels](bundles/): los resultados corridos se encuentran bajo kaggle-outputs/ y tienen links directos en el reporte.

La última lectura de Kaggle fue 11:25/30 h usadas, quedan 18:35 h. Se conserva el mínimo solicitado de 12 h semanales; el run Burgers L-BFGS terminó en 32 min 29 s. Se planea solo un screening Heat pareado de 1 h 40 min si la cuota se relee en ≥14 h 10 min después de corregir el bundle. No se corrieron notebooks ni validadores localmente. La referencia BDF de 12.3 s de CPU que aparece en el archivo histórico es candidate-only; no valida el campo Allen–Cahn ni forma parte de los resultados GPU.

El [readout del 2026-09-27](baseline-readout-20260927.md) es un snapshot histórico. No usarlo para los scores actuales.
