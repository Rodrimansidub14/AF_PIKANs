# EXP-0006 — Burgers: Sobol frente a RAD

Estado: preparado, pre-registro pendiente de firma científica. Ejecución del plan autorizada por el investigador en conversación; ningún agente firma el gate científico ni congela el baseline.

Pregunta: ¿la concentración residual observada en EXP-0005 permite reducir el error de campo y del frente bajo el mismo presupuesto?
Hipótesis: la media por pareja de log(L2_RAD/L2_Sobol) es negativa; se refuta una mejora observada si es >=0. Un intervalo que abarque cero permanece inconcluso.
Métrica primaria: field_relative_l2 de evals/checkpoint_validation.py, problem=burgers_pinnacle_1d. Malla y Cole–Hopf existentes, sin cambios. Comparador Sobol concurrente; no sustituirlo por los tres seeds históricos.
Cinco semillas emparejadas 201–205. No usar puntos espaciales como réplicas, no reemplazar fallos, no ajustar parámetros después de observar resultados, no reusar seeds 101–103 como confirmación.
Los brazos difieren únicamente en la actualización del interior. Máximo 20000 pasos/1200 s cada uno; estimación CPU/GPU separada en presupuesto. No atribuir a RAD una comparación a igual número de pasos si un brazo agotó tiempo antes.

Criterios propuestos, NO activos: reducción práctica de20% en razón geométrica RAD/Sobol (<=0.8), con media histórica*0.8 como referencia absoluta reportada, no segundo endpoint; frente, residual e IC/BC dentro de la envolvente máxima medida de EXP-0005. El20% es una elección práctica a firmar, no un óptimo de la literatura ni un cálculo de potencia. Los valores concretos y hashes están en var/agent_runs/burgers-rad-exp006/baseline-freeze-proposal.json. Esa envolvente de tres semillas no representa tolerancia física universal y requiere decisión humana antes de convertirse en gate.
Tamaño: n=5 pares iniciales. Una ampliación a10 se propone únicamente si el intervalo es inconcluso y requiere protocolo nuevo con control de análisis secuencial; no mirar cinco y declarar significancia con diez como si fuera un solo análisis fijo.
Sin aceptación automática; test final cerrado. El verificador actual sigue devolviendo threshold_not_defined mientras no exista baseline congelado. No firmar manifest.approved_by por cuenta del investigador.

Autorización pendiente concreta: congelación del snapshot de validación (sin abrir test final) y firma de criterios/parametrización. Tras ella: bundle con hashes exactos, smoke remoto técnico si se necesita, diez trabajos, descarga/sync W&B, análisis emparejado, decisión humana. No lanzar Allen–Cahn ni DFG por herencia de un resultado de Burgers.

Artefactos preparados: manifiesto, tabla de decisiones, audit.html/localization.csv/hotspot-reproducibility.csv/cost-and-basis.csv, baseline-freeze-proposal.json, notebook reproducible en bundle-v1.
