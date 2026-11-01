# EXP-0006 — expediente concreto para firma científica

La ejecución del plan ya está autorizada. Pendiente: decidir congelar el snapshot de validación EXP-0005 y firmar este pre-registro; no se solicita aceptar un resultado futuro.

| Criterio propuesto | Valor | Alcance |
|---|---:|---|
| Razón geométrica emparejada L2_RAD/L2_Sobol | <=0.8 | Mejora práctica del 20%, propuesta a firmar |
| Referencia absoluta 0.8*media piloto | 0.043238121 | Descriptiva; no segundo endpoint |
| front_field_rmse_normalized | <=0.55452116 | Máximo observado entre 3 semillas piloto |
| residual_rmse_abs | <=0.080897356 | Máximo observado entre 3 semillas piloto |
| boundary_max_abs_error | <=0.012985192 | Máximo observado entre 3 semillas piloto |
| initial_max_abs_error | <=0.02727409 | Máximo observado entre 3 semillas piloto |

Se propone revisar estos guardas por semilla, sin compensar un fallo físico con otra semilla mejor. Son envolventes piloto, no tolerancias físicas universales. Reporte principal: cinco razones individuales, media logarítmica y CI t95%; sin aceptación automática. Un intervalo inconcluso no demuestra equivalencia.

Semillas reales 201–205; diez tareas; dos T4; máximo 20000 pasos o 1200 s (control entre operaciones), timeout exterior 2100 s/tarea. Hasta 3.17 h nominales de ejecución, sin cola. No ampliar a 10 pares automáticamente.

Fuente vinculante: AGENTS.md:33 (firmar el pre-registro es humano), context/agent-policy.toml:124 (solo el investigador congela baselines y mueve umbrales). Ni evals ni claims fueron modificados. La propuesta incluye hashes y los tres scorecards en baseline-freeze-proposal.json.
