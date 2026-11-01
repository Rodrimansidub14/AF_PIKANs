# EXP-0006: decisiones de parámetros — propuesta previa a corrida

| Decisión | Valor | Fundamento |
|---|---|---|
| Problema | Burgers canónico, nu=0.01/pi, x en [-1,1], t en [0,1], IC=-sin(pi*x), BC=0 | Contrato y verificador intactos de EXP-0005. |
| Modelo | RBF-PIKAN [2,16,16,1], 3648 parámetros, centros/anchos fijos | Aislar política de muestreo; no modificar capacidad. |
| Entrenamiento | PINA/PyTorch float64, Adam 0.001, batch 1024 | Igual que el baseline. Estado Adam se conserva al adaptar. |
| Interior | 8192 puntos iniciales Sobol, mismo pool por pareja | Conteo fijo; diagnóstico, IC, BC y validación independientes. |
| Frontera e IC | 512 por lado; 1024 en IC | Igual que EXP-0005, no se adaptan. |
| Semillas reales | 201,202,203,204,205 | Nuevas; cinco pares, diez runs, no diez réplicas por brazo. |
| RAD | p_i=0.5/M+0.5*abs(r_i)^2/sum(abs(r)^2) | RAD k=2,c=1 sobre pool finito; 50% masa uniforme limita pérdida de cobertura. Si residual cero, distribución uniforme. |
| Candidatos | 32768 Sobol independientes, stream=10*seed+7 | Cuatro veces el pool activo, soporte reproducible y costo acotado. Propuesta operativa, no hiperparámetro óptimo publicado. |
| Refresco | desde paso 2000; cada 1000; hasta antes del paso 20000 | Warmup 10% del presupuesto; como máximo 18 refrescos. Medir costo, no asumir ganancia. |
| Selección | con reemplazo, RNG=10*seed+8 | Variante discreta explícita de RAD; duplicados codifican probabilidad. Registrar puntos únicos/ESS; no atribuir reproducción exacta de la implementación original. |
| Corrección de importancia | ninguna | RAD cambia el objetivo espacial; no se presenta como estimador insesgado de la integral uniforme. |
| AD de candidatos | chunks de 512, float64, gradientes solo para evaluar residual | No backprop a pesos ni uso de referencia; no cambia el RNG de entrenamiento. |
| Presupuesto | 20000 pasos o 1200 s por brazo, verificado entre operaciones | Igual límite operativo, costo adaptativo incluido; una operación en curso puede sobrepasarlo. Timeout exterior2100s. Si algún brazo no completa pasos, contraste de pasos iguales inconcluso; reportar tiempo y pasos reales. |
| GPU | dos T4, máximo dos tareas concurrentes | Una tarea por GPU; cinco parejas. IDs 201–205 control y 1201–1205 RAD solo para scheduler. Orden intercalado y alternado; slot real registrado. |
| Timeout exterior | 2100 s por tarea; setup <=900 s | Límite nominal de cinco olas=10500 s más setup (~3.17 h), sin cola del proveedor. No promesa de tiempo exacto. |
| Diagnósticos | cada250 pasos, 256 puntos y 32 sensibilidades; métricas cada100 | Conserva instrumentación baseline; RAD añade snapshots, probabilidades, puntos, ESS y costo cada refresh. |
| W&B | offline en Kaggle, sync local posterior | No embebe secretos; un run y artefacto por brazo/problema/semilla. |
| Métrica primaria | log(L2_RAD/L2_Sobol) por semilla; media y CI t95% | Unidad independiente: pareja. n=5 no garantiza potencia ni normalidad; publicar valores individuales. |
| Secundarias | frente, IC/BC, residual, costo total, ejemplos vistos, cobertura | Un error global menor no basta para afirmar mejora física local. |
| Delta práctico propuesto | reducción20%: razón geométrica RAD/Sobol <=0.8; referencia absoluta=0.8*media piloto | Decisión práctica propuesta, no óptimo publicado ni cálculo de potencia. Requiere firma; valores en baseline-freeze-proposal.json. |
| Umbrales | propuesta separada derivada del snapshot EXP-0005; pendientes de congelación humana | No modifica evals ni convierte un screening en baseline aceptado. |
| Cierre | hashes, notebook con tablas/Plotly log, archivo de resultados, reporte de fallos | El test final no se abre; no genera claim accepted. |

Fuentes: [Wu et al., RAD/RAR-D](https://arxiv.org/abs/2207.10289); [vRBA, 2026](https://www.nature.com/articles/s44387-026-00084-4). Los parámetros operativos anteriores son decisiones del proyecto, no una afirmación de óptimo SOTA.
