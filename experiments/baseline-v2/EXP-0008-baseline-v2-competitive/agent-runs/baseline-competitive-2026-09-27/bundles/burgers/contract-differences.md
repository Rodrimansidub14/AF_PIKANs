# Diferencias entre PINNacle publicado y este paquete

| Elemento | PINNacle Vanilla publicado | Contrato de paquete | Comparabilidad |
|---|---|---|---|
| PDE e instancia | Burgers Raissi con nu=.01/pi, IC -sin(pi*x) y contorno Dirichlet | Mismo contrato físico, [-1,1]x[0,1] | Reproducible |
| Red y optimizador Vanilla | MLP tanh 5x100, Adam 1e-3, 20k para fila Vanilla | Igual arquitectura/Adam, límite 20k y 900 GPU-s | Receta principal reproducible |
| Batch pipeline | Código/PINA del paper y su loader | Torch-only minibatches con reemplazo desde pool fija común | Adaptación relevante; no es réplica literal |
| Muestreo | Configuración del paper | Sobol AF fija, misma pool bytewise entre brazos; versión SciPy preinstalada registrada | Adaptación común a ambos brazos |
| Métrica/referencia | L2 relativo respecto a FEM discretizada | Port CUDA independiente de Cole–Hopf/Gauss–Hermite 128, grid 1025x129 | Paper es contexto, no score comparable; el port está DRAFT_UNVERIFIED |
| Precisión/muestra/costo | Protocolo propio del artículo, 3 corridas reportadas | float64, cinco seeds 811–815, cap 900 GPU-s por brazo | Diseño AF adaptado |
| L-BFGS | Número publicado distinto pertenece a otra variante | Excluido del brazo Vanilla | Configuraciones separadas |

Fuente: [PINNacle, arXiv:2306.08827](https://arxiv.org/abs/2306.08827). El valor Vanilla (1.45×10^{-2}±1.59×10^{-3}) y el número de la variante L-BFGS (1.33×10^{-2}±5.30×10^{-5}) son anclas contextuales porque usan FEM y configuraciones distintas. No son umbrales del experimento AF.

El scorer final corre solo tras todo el entrenamiento, puntúa endpoints y mantiene en CUDA la malla, Cole–Hopf, autodiferenciación, residuales y métricas; convierte arrays a CPU solo al serializar NPZ. Este port no se ha ejecutado ni validado frente a la referencia del repo, por lo que no hay comparabilidad establecida hasta una futura auditoría autorizada. No se instala Plotly ni paquetes con pip.
