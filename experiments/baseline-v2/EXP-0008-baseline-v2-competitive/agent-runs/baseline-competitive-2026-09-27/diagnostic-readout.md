# Diagnóstico cuantitativo de las corridas existentes

Se ejecutó [el script de posproceso](diagnose_archived_runs.py) sobre los CSV ya entregados, sin entrenamiento ni tests. [Salida estructurada](diagnostics.json). Las medias y desviaciones son **entre cinco semillas**; puntos espaciales de una misma predicción no se cuentan como réplicas.

| Problema | Señal descriptiva (media ± DE, n=5) | Decisión de medición siguiente |
|---|---:|---|
| Heat RBF, cohortes EXP-0005+extensión | L2 relativo `1.98294e-4 ± 2.38199e-4`; costo entrenamiento+diagnóstico `190.97 ± 6.36 s`. | [Auditoría estática](heat-cohort-equivalence.md) permite un agregado descriptivo n=5; la dispersión es alta y no convierte el screening en confirmación. B-spline local n=3 mantiene menor L2 (`2.30142e-5`) a mayor costo (~301 s). |
| Burgers Sobol | RMSE de campo en frente / complemento `30.69 ± 14.87`; residual RMS frente / complemento `9.00 ± 2.66`. | Medir si el fallo en el frente persiste al igualar optimizer/capacidad/BC. No inferir todavía que una política de focalización mejore el error. |
| Allen–Cahn | L2 de campo provisional `0.01843 ± 0.00338`; residual normalizado medio `0.00122 ± 0.00084`. | Priorizar una referencia convergida y métricas de interfaz; residual pequeño no reemplaza campo. Las dos magnitudes tienen unidades/normalizaciones distintas y su cociente no es una razón de error física. |
| DFG | RMS cerca del cilindro / global: continuidad `2.94 ± 0.14`, momentum-x `3.26 ± 0.10`, momentum-y `3.46 ± 0.15`; no-deslizamiento RMS normalizado en cilindro `0.02065 ± 0.00694`. | Auditar fuerzas, outlet y FEM; luego separar error de región crítica y QoIs. El ratio de un subconjunto frente al global es descriptivo, no un test de significancia independiente. |

Estos patrones son candidatos de explicación. Para seleccionar una señal del mecanismo habrá que mostrar predicción incremental del error held-out **más allá del residual** entre semillas y problemas; la referencia solo evaluará la señal y nunca alimentará la política durante entrenamiento.
