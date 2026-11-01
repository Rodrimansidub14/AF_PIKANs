# evals/ — el verificador

**Superficie protegida. Ningún agente puede escribir aquí.**

Esta carpeta es el verificador del loop proposer–verifier. Contiene la
definición de qué significa "mejor" en este proyecto. Si un agente pudiera
modificarla, podría hacer pasar cualquier mecanismo cambiando la vara en vez
del resultado; toda la evidencia de la tesis quedaría sin valor.

## Ampliación exploratoria autorizada

El investigador autorizó preparar Burgers, Allen–Cahn y DFG para revisión.
`suites/exploratory_rbf_signals.yaml` declara esa campaña aparte; Poisson queda
pendiente. Hay operadores y meta-evals, no un runner de entrenamiento completo.
Burgers dispone de referencia exacta. Allen–Cahn y DFG bloquean errores de campo
hasta verificar referencias numéricas convergidas. Las suites Tier 0–3 previas
y los baselines congelados no se modifican por esta ampliación.

| Ruta | Responsabilidad |
|---|---|
| `problems/` | Environments: dominio, PDE, BC/IC, solución de referencia, malla de evaluación |
| `graders/` | Scorers numéricos. Exactos, sin etiquetas, sin jueces LLM |
| `suites/` | Agrupación por tier de costo |
| `baselines/` | Puntajes congelados. **De aquí salen los `threshold` de los gates** |
| `meta/` | Meta-evals: verifican que los graders midan lo que dicen medir |

## Por qué no hay AI graders

En post-training de LLMs el grader es el problema difícil: "¿esta respuesta es
buena?" no tiene ground truth, y por eso se recurre a reward models y jueces
LLM. En SciML la señal viene con el problema: el operador de la PDE mide el
residuo sin etiquetas, y los casos canónicos tienen solución analítica.
Sustituir un operador diferencial exacto por un juez aprendido sería cambiar
un verificador exacto por uno ruidoso.

## La amenaza de validez principal

En un PINN el grader y la loss son el mismo operador. Si se evalúa el residuo
sobre los mismos puntos de colocación con los que se entrenó, hay leakage: un
mecanismo de focalización puede "ganar" concentrando puntos donde luego se
mide, sin mejorar la solución.

Por eso cada problema declara una `eval_grid` **densa, fija y held-out**,
generada desde `problem.yaml` y nunca desde el mecanismo. Los mecanismos no
pueden leer la malla de evaluación.

## Tiers

| Tier | Costo | Contenido | Autoridad |
|---|---|---|---|
| 0 | segundos, CPU | smoke: corre, hay artefactos, los graders son sanos (MMS) | agente, libre |
| 1 | minutos, CPU | Heat 1D, presupuesto corto, 2 semillas — gate de regresión | agente, libre |
| 2 | horas, GPU | Burgers / Allen-Cahn, presupuesto completo, >=5 semillas | agente con presupuesto aprobado |
| 3 | caro | Navier-Stokes, evidencia de tesis | gate humano explícito |

El loop agent-first solo es viable si Tier 0/1 absorbe la mayoría de las
iteraciones. La restricción vinculante de este proyecto es GPU-hora, no
throughput de agentes.
