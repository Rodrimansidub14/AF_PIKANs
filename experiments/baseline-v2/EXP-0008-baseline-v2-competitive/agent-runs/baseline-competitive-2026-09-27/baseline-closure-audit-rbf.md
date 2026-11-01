# Auditoría de cierre — baseline RBF-Sobol

Fecha: 2026-09-29. Revisión de informes, scorecards, manifests y artefactos ya descargados. Solo lectura; sin entrenamiento local/remoto, sin consulta de cuota y sin aceptación científica. Este documento no congela umbrales.

## Lectura ejecutiva

Existe evidencia reutilizable para una comparación interna del control RBF-Sobol en Heat, Burgers y Allen–Cahn. **No se necesita repetir entrenamiento ahora** si una verificación final confirma que ambos brazos comparten inicialización, semillas, pools y contrato, y se entrenan por separado bajo el mismo presupuesto. Reutilizar un checkpoint final solo sirve para comparar scorecards post hoc, no para probar un mecanismo que cambia el entrenamiento. El freeze del baseline y la aceptación de referencias permanecen sujetos a decisión humana. No afirmar SOTA con este conjunto.

> **Actualización 2026-09-29 (auditoría por semilla ejecutada, ver `audit-rbf-20260929/README.md`):** (1) el `2.060803e-6` de Heat es el checkpoint post-L-BFGS; el Adam-only (20k) de las mismas corridas, re-puntuado en CPU, es `2.197255e-4 ± 1.802112e-4` y coincide con la fila `RBF-PIKAN_archived` del scorecard; las cohortes v4 y S2 comparten semillas y pools (hash idéntico), así que la brecha ~81–100× es la etapa L-BFGS, no la cohorte. (2) En Allen–Cahn, el endpoint Adam también se puntuó contra FD8192, no contra ETDRK4; la fila AC de abajo lo dice al revés. Matiz de malla: score-only en malla nativa de la referencia, S3 en 512×256 interpolada. (3) Burgers reproducido en los diez jobs (CSV = run_summary, hashes de checkpoint = evidencia).

## Matriz de elegibilidad

| Problema | Evidencia RBF-Sobol disponible | Estado del control para mecanismo | Bloqueo/acción pendiente |
|---|---|---|---|
| Heat α=0.1 | Analítica; grilla confirmatoria 512×256. Heat v4 RBF n=5: `1.677869e-4 ± 1.338478e-4`; 500 s de Adam por seed, aproximadamente 18k updates, sin L-BFGS. Otra cohorte RBF reporta `2.060803e-6 ± 2.306606e-7` tras 20k Adam y continuación L-BFGS parcial (4 paradas tempranas/1 cap). | `baseline_candidate`, pero solo después de elegir una cohorte y verificar etapa/checkpoint. | La brecha de ~81× y las etapas distintas obligan a fijar código/hash, pools, inicialización, terminación y etapa. El valor 2.06e-6 no queda acreditado como score Adam-only. Ledger Heat v4 `incomplete_validation` se debe a su target de 1M updates; no describe por sí solo la calidad del checkpoint ni el cap fijo de 500 s. Comparar el mecanismo a igual tiempo total con el B0 seleccionado. No agregar los dos grupos. |
| Burgers canónico | EXP-0006, cinco pares Sobol/RAD, seeds 201–205, 20k Adam; Cole–Hopf en grilla común. PINNacle Adam→L-BFGS tiene score adicional con referencia FEM del paper. | `baseline_candidate` para contraste Sobol/RAD; scorecards de entrenamiento concluidos. | Los artefactos contienen scores, pools y hashes. Completar el índice final de hashes/costos y dejar firmado humanamente cualquier umbral. El resultado no es un claim SOTA. La reproducción PINNacle es otro contrato/comparación. |
| Allen–Cahn 1D | RBF-Sobol seeds 401–405: `1.650555e-2 ± 2.435593e-3` contra ETDRK4 candidata en el endpoint anterior a L-BFGS. Después de 240 s/seed de continuación L-BFGS: `9.163726e-3 ± 2.664378e-3` contra FD/RK4 candidata. Diferencia relativa FD8192–ETDRK4 `3.839432e-7`; acuerdo interno ETDRK4 reportado con refinamientos espaciales/temporales. | `screening_only` hasta decisión humana sobre la referencia; baseline RBF reutilizable por etapa. | Elegir explícitamente endpoint Adam o continuación L-BFGS según presupuesto del candidato. Revisar procedencia y formulación de FD/RK4, ETDRK4 y su relación con la referencia BDF/`.mat` del protocolo. No mezclar scores pre y post L-BFGS. |
| DFG Re=20 | FEM Taylor–Hood L3 y score-only de u/v/p en grilla 221×42 con gauge fijo; cohortes separadas. | `screening_only`, fuera del cierre principal. | `Cl` cambia `0.00664849` entre las dos mallas finas, sobre el límite de `0.002`; aclarar evaluador de fuerza. |

## Burgers: resultados pareados del artefacto EXP-0006

| Seed | Sobol L2 relativo | RAD L2 relativo | RAD/Sobol |
|---:|---:|---:|---:|
| 201 | 0.02262770 | 0.30166554 | 13.33 |
| 202 | 0.04387225 | 0.27168738 | 6.19 |
| 203 | 0.02138382 | 0.12500442 | 5.85 |
| 204 | 0.02733542 | 0.15293131 | 5.60 |
| 205 | 0.08252980 | 0.21214355 | 2.57 |

La media geométrica RAD/Sobol es `5.865`; IC t pareado del 95% informado: `2.844–12.098`. RAD tuvo mayor error en 5/5 parejas. Ambos brazos completaron 20,000 pasos. El tiempo registrado promedió aproximadamente 1,024 s para RAD y 1,039 s para Sobol: los resultados son paso igualado, no tiempo exactamente igualado. Para el contraste futuro, contar el tiempo real del mecanismo y usar el cap del control.

Artefactos: `var/agent_runs/burgers-rad-exp006-v4/download-attempts/attempt-01/pikan-4393e173686298ee/analysis/paired-runs.csv`, `paired-primary-summary.csv`, `artifact-hashes.json` y el manifiesto del bundle. El delta práctico de 20% en el preregistro no está firmado; no se interpreta como gate activo.

## Decisión operativa propuesta

1. Seleccionar una sola cohorte B0 RBF-Sobol por problema y contrato; compartir inicialización, semillas y pools con el brazo del mecanismo, pero entrenar cada brazo por separado. Reutilizar checkpoints finales solo para análisis post hoc.
2. En Heat, decidir entre el control fijo de 500 s y la cohorte de 20k Adam más L-BFGS parcial tras confirmar hashes, checkpoint puntuado y correspondencia con el código del mecanismo. El endpoint v4 de 500 s es naturalmente comparable si el candidato recibe el mismo tiempo total.
3. En Burgers, reutilizar EXP-0006 como evidencia Sobol/RAD; mantener PINNacle como comparación externa acotada.
4. En Allen–Cahn, obtener revisión humana de referencia y seleccionar la etapa del checkpoint compatible con el mecanismo. Hasta entonces el score es exploratorio.
5. Preparar propuesta de freeze `eval_integrity` solo después del chequeo final de procedencia, costos y filas por semilla. La aceptación de referencia/claim sigue el gate `scientific_claim`.
6. Abrir nueva corrida solo si no existe un par comparable con inicialización, pools, referencia, contrato y presupuesto compatibles, o si falta el checkpoint puntuado verificable. Cualquier Tier 2 necesita presupuesto autorizado, cuota fresca y reserva mínima verificada.

## Alcance que falta para la verificación final

- Heat: comparar SHA-256 del modelo/código y pools de las cohortes 500 s y 20k Adam + L-BFGS parcial; revisar los cinco `cost.json`, identificar exactamente qué checkpoint puntuó `2.060803e-6`, y vincular una sola identidad/etapa a B0. No presentarlo como Adam-only sin evidencia del snapshot puntuado.
- Burgers: indexar los hashes de los diez checkpoints/pools desde los artefactos descargados y confirmar que el resumen por semilla coincide con los CSV; no requiere recalcular entrenamiento.
- Allen–Cahn: unir hash de cada checkpoint, pools, resumen de entrenamiento pre/post L-BFGS y score ETDRK4/FD por semilla; documentar dictamen humano pendiente de referencia.
- Ningún paso de esta auditoría modifica `evals/`, `registry/claims.jsonl` ni los umbrales protegidos.
