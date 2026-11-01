# EXP-0030, bloques F y G (exploratorio, post hoc)

## F. Incertidumbre entre semillas sobre la malla de referencia

Campos predichos en el paso 20 000 sobre la malla de 4 221 nodos de la referencia FEM. Dispersión = rango intercuartílico entre semillas por nodo; error = |mediana entre semillas − referencia|. Spearman entre el mapa de dispersión y el de error: **solo descripción** (n = 3 o 5 semillas; los nodos no son independientes).

| problema | brazo | n semillas | campo | error del campo mediano (rel. L2) | dispersión mediana (IQR / RMS de la referencia) | Spearman dispersión–error |
|---|---|---|---|---|---|---|
| conv | B | 5 | u | 0.041 | 0.099 | +0.16 |
| conv | B | 5 | v | 0.037 | 0.119 | +0.94 |
| conv | B | 5 | φ | 0.044 | 0.005 | +0.23 |
| conv | E | 5 | u | 0.050 | 0.030 | +0.81 |
| conv | E | 5 | v | 0.029 | 0.032 | +0.88 |
| conv | E | 5 | φ | 0.051 | 0.003 | +0.29 |
| conv | M | 3 | u | 0.166 | 0.151 | +0.48 |
| conv | M | 3 | v | 0.207 | 0.038 | +1.00 |
| conv | M | 3 | φ | 0.043 | 0.004 | +0.16 |
| dire | B | 3 | u | 0.204 | 0.022 | +0.96 |
| dire | B | 3 | v | 0.228 | 0.024 | +1.00 |
| dire | B | 3 | φ | 0.240 | 0.043 | +0.77 |
| dire | E | 3 | u | 0.198 | 0.049 | +0.97 |
| dire | E | 3 | v | 0.220 | 0.057 | +1.00 |
| dire | E | 3 | φ | 0.216 | 0.051 | +0.90 |
| dire | M | 3 | u | 0.183 | 0.027 | +0.67 |
| dire | M | 3 | v | 0.192 | 0.024 | +1.00 |
| dire | M | 3 | φ | 0.211 | 0.056 | +0.81 |

## G. Fichas, proveniencia, dominio y fallos

### Ficha por corrida (de `meta_json` y `run_card`)

| problema | brazo | semilla | hardware | torch | python | s/paso | horas de proceso | commit | ficha de IA responsable |
|---|---|---|---|---|---|---|---|---|---|
| conv | B | 20260728 | T4 | 2.11.0+cu128 | 3.13.15 | 0.432 | 2.40 | ? | no |
| conv | B | 20260729 | H100 | 2.14.1+cu130 | 3.12.10 | 0.213 | 1.18 | ? | no |
| conv | B | 20260730 | H100 | 2.14.1+cu130 | 3.12.10 | 0.207 | 1.15 | ? | no |
| conv | B | 20260731 | A100-80 | 2.14.1+cu130 | 3.12.10 | 0.311 | 1.73 | ? | sí |
| conv | B | 20260732 | A100-80 | 2.14.1+cu130 | 3.12.10 | 0.311 | 1.73 | ? | sí |
| conv | E | 20260728 | T4 | 2.11.0+cu128 | 3.13.15 | 0.456 | 2.53 | ? | no |
| conv | E | 20260729 | H100 | 2.14.1+cu130 | 3.12.10 | 0.222 | 1.24 | ? | no |
| conv | E | 20260730 | H100 | 2.14.1+cu130 | 3.12.10 | 0.231 | 1.28 | ? | no |
| conv | E | 20260731 | A100-80 | 2.14.1+cu130 | 3.12.10 | 0.321 | 1.79 | ? | sí |
| conv | E | 20260732 | A100-80 | 2.14.1+cu130 | 3.12.10 | 0.321 | 1.79 | ? | sí |
| conv | M | 20260728 | A100-80 | 2.14.1+cu130 | 3.12.10 | 0.351 | 1.95 | ? | sí |
| conv | M | 20260729 | A100-80 | 2.14.1+cu130 | 3.12.10 | 0.350 | 1.94 | ? | sí |
| conv | M | 20260730 | A100-80 | 2.14.1+cu130 | 3.12.10 | 0.347 | 1.93 | ? | sí |
| dire | B | 20260728 | T4 | 2.11.0+cu128 | 3.13.15 | 0.326 | 1.81 | ? | no |
| dire | B | 20260729 | H100 | 2.14.1+cu130 | 3.12.10 | 0.171 | 0.95 | ? | no |
| dire | B | 20260730 | H100 | 2.14.1+cu130 | 3.12.10 | 0.163 | 0.90 | ? | no |
| dire | E | 20260728 | T4 | 2.11.0+cu128 | 3.13.15 | 0.349 | 1.94 | ? | no |
| dire | E | 20260729 | H100 | 2.14.1+cu130 | 3.12.10 | 0.187 | 1.04 | ? | no |
| dire | E | 20260730 | H100 | 2.14.1+cu130 | 3.12.10 | 0.172 | 0.95 | ? | no |
| dire | M | 20260728 | A100-80 | 2.14.1+cu130 | 3.12.10 | 0.297 | 1.65 | ? | sí |
| dire | M | 20260729 | A100-40 | 2.14.1+cu130 | 3.12.10 | 0.246 | 1.37 | ? | sí |
| dire | M | 20260730 | A100-40 | 2.14.1+cu130 | 3.12.10 | 0.249 | 1.38 | ? | sí |

`horas de proceso` es el tiempo de pared de un proceso; en Modal corrían 4 procesos por GPU, y en Kaggle uno por GPU. El costo en USD de la ronda de M **no se verificó** (el investigador reportó un gasto ≈ 11 USD mayor que la estimación del agente; probable causa: las A100 resultaron de 80 GB, más caras que la tarifa supuesta; sin confirmar).

### Proveniencia
- **Escribió el agente:** los scripts de este análisis (`analysis/`), las tablas generadas y este texto.
- **Decidió una persona:** qué se corre y en qué hardware, la firma de EXP-0029, la métrica robusta como secundaria (post hoc), el cierre del cómputo y la lectura de resultados.
- **Los datos de entrada** no se modificaron.

### Dominio de validez
Un bimorfo PVDF, dos problemas (converso, directo), receta del artículo (float64, Adam 20 000 pasos, sin L-BFGS, pesos fijos), n ≤ 5 semillas (M: 3), hardware mixto. No se afirma nada fuera de ese dominio.

### Catálogo de fallos
- M converso, semillas 31 y 32: sin memoria (CUDA OOM, A100 de 40 GB con 4 procesos y registro de bloques completos). No existen.
- E converso semilla 32: l2re_u 0.14 → 0.09 → 0.52 en los pasos 10 000, 15 000, 20 000 (salto al final).
- Saltos de pérdida: solo medibles por paso en las corridas de EXP-0029 (A100); B y E semillas 28-30 tienen la pérdida cada 100 pasos.

### Limitaciones de comparabilidad
- **Hardware × brazo:** B y E semilla 28 en T4, semillas 29-30 en H100, semillas 31-32 en A100-80GB; M converso y directo 28 en A100-80GB, M directo 29-30 en A100-40GB. No hay pares M/E con el mismo hardware.
- El pareo por semilla no es exacto en float64 entre dispositivos.
- n distinto por brazo y por problema.
- La semilla 28 de B y E es la del artículo; en el converso 31-32 no hay M.
