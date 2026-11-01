# Diagnóstico de señales previo al diseño del mecanismo

**Objetivo:** decidir si una señal disponible durante entrenamiento aporta información sobre el error de solución held-out **después de conocer el residual PDE**, en Heat, Burgers, Allen–Cahn y DFG. Este documento no define todavía una política de focalización ni cambia el control Sobol.

La literatura de muestreo residual ofrece el control natural RAD/RAR-D ([Wu et al.](https://arxiv.org/abs/2207.10289)); [gPINN](https://arxiv.org/abs/2111.02801) motiva estudiar derivadas del residual, pero su coste adicional se cuenta. La ocupación y sensibilidad RBF son hipótesis internas de la tesis, no señales ya validadas por esos papers.

| Cantidad | Entrenamiento puede verla | Uso en este estudio |
|---|---|---|
| Magnitud del residual por componente, IC/BC y gradiente del residual | Sí, con AD y puntos accesibles | Comparador base y predictor físico. En DFG separar continuidad, momentum-x y momentum-y; no reducirlos prematuramente a una suma. |
| Activación/entropía de bases RBF, sensibilidad de salida a centros/coordenadas, norma/conflicto de gradientes por término | Sí, si se instrumenta sin verdad de campo | Predictores candidatos; registrar costo adicional por evaluación. |
| Distancia a pared, cilindro, frente estimado desde `u`, tiempo, velocidad/derivadas | Sí, calculable sin referencia | Covariables de geometría/contexto para evitar que una señal gane solo por localizar la región obvia. |
| Error absoluto de campo, posición/anchura de interfaz, error de QoI frente a FEM/analítica | **No** durante entrenamiento | Solo outcome del análisis; nunca entra en muestreo, loss, early stopping ni selección de checkpoint. |

## Plan de medición

1. Congelar por seed un pool diagnóstico distinto de entrenamiento y del score final. Calcular las señales en checkpoints fijados por **pasos y tiempo**, no seleccionados mirando el error.
2. Una vez cerradas las referencias, etiquetar errores sobre el pool diagnóstico: Heat/Burgers/AC error de campo, frente/interfaz por región; DFG error de `u,p` contra FEM, caudal y QoIs de frontera. Registrar campos sin referencia como `missing`, nunca como cero.
3. Ajustar en datos de desarrollo un predictor residual-only y otro residual + señal candidata + geometría. Comparar por seed el cambio en detección del decil superior de error, calibración y error de predicción; evitar que celdas espaciales de una misma seed se traten como réplicas independientes.
4. Estimar incertidumbre por bootstrap de seeds y dejar fuera una seed completa en cada evaluación. Con solo cinco semillas, cualquier asociación es exploratoria; para afirmar transferencia, repetir leave-one-problem-out y medir si la dirección del efecto se sostiene en al menos dos PDE con referencia confiable.
5. Reportar también costo de instrumentación, robustez entre checkpoints y si la señal añade algo sobre residual + distancia geométrica. Registrar asociaciones negativas. Una señal que solo correlaciona dentro de una corrida no justifica intervención.

**Criterio para pasar al diseño del mecanismo:** señal definida sin fuga de referencia, mejora incremental reproducible fuera de la seed usada para ajustarla, costo compatible con el presupuesto y una hipótesis causal falsable. El resultado RAD negativo de Burgers y los ratios de frente/cilindro en [el diagnóstico actual](diagnostic-readout.md) motivan esta prueba; no predicen su éxito.
