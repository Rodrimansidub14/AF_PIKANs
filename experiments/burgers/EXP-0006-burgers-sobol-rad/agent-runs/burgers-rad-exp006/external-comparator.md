# Comparador externo — preparación de viabilidad

Candidato: RGA-KAN, método completo separado del contraste causal Sobol/RAD.
Repositorio oficial consultado: https://github.com/srigas/RGA-KANs
HEAD observado en esta preparación: `bb901937adf75dae4f3a4a1896f0f1274e4b6f05`.
Requirements oficiales: `jaxkan[gpu]==0.2.13`, `jupyterlab==4.4.3`,
`matplotlib==3.10.3`, `seaborn==0.13.2`, `scikit-learn==1.7.0`.
Fuente: https://raw.githubusercontent.com/srigas/RGA-KANs/main/requirements.txt

No instalar JAX dentro del worker PINA; es un entorno separado. Antes de un
comparador científico: fijar commit, inspeccionar notebook Burgers y ecuación,
nu, IC, BC, normalización, precisión, arquitectura, optimizador y muestreo;
exportar checkpoint/campos a evaluación independiente compatible. No copiar
scores publicados. El NS del repositorio no se presume DFG.

Prueba técnica propuesta posterior al gate: una T4, <=120 s de entrenamiento
más instalación/compilación contabilizada aparte, sin seleccionar hiperparámetros
por la métrica final. Registrar VRAM, primer paso/compilación y throughput estable.
Si no es reproducible dentro de cuota, candidato alternativo PirateNet/JMLR2024:
https://www.jmlr.org/papers/v25/24-0313.html

Estado: fuentes y dependencias verificadas; GPU smoke, adaptación y comparación
por cinco semillas NO ejecutadas. Esto no demuestra SOTA ni viabilidad T4.
