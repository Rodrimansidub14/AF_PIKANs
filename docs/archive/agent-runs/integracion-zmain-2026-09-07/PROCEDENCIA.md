# Procedencia de contenido y figuras

Integración solicitada expresamente sobre `z-main.tex`, 7 de septiembre de 2026. El documento principal está en la raíz; esta carpeta contiene respaldo y evidencia de edición. No se modificaron políticas ni se promovió contenido al vault.

## Figuras integradas

| Archivo bajo figuras/revision2026 | Fuente / objeto recreado | Operación y alcance |
|---|---|---|
| fem_poisson.pdf, fem_convergencia.pdf | Langtangen y Logg, *Solving PDEs in Python: The FEniCS Tutorial I*, DOI 10.1007/978-3-319-52462-7 | Implementación propia de ensamblaje P1 triangular y convergencia con SciPy, trasladada de la revisión previa. No se ejecutó FEniCS. Fuente reproducible: `generar_figuras_previo.py`, función FEM y `validacion-fem-previa.json`. |
| arquitecturas.pdf | Liu et al., KAN, arXiv:2404.19756, sección de arquitectura | Redibujo de la distinción de aristas/nodos MLP–KAN; cambia disposición y tamaño. No replica resultados. |
| bases.pdf | Liu et al., parametrización B-spline; recurrencia Cox–de Boor | Evaluación matemática propia de cúbicas; tercer panel es función compacta ilustrativa, no un resultado de Free-RBF-KAN. |
| perdida.pdf | Raissi et al. 2019, DOI 10.1016/j.jcp.2018.10.045 | Diagrama propio de construcción de residual y condiciones en PINN. |
| tesis-pdf103.pdf | José Daniel González Carrillo 2024, Figura 38, p. impresa 79, PDF103 | Página extraída del PDF adjunto, visualizada con recorte LaTeX `[115 185 90 170]` puntos. Conserva los tres paneles originales de PINN/FEM/error, sin alterar sus valores. |
| tesis-pdf105.pdf | Mismo autor, Figura 40, p. impresa 81, PDF105 | Página extraída del adjunto, recorte LaTeX `[145 255 120 120]` puntos. Conserva comparación de deformación y su escala visual x50. |
| problemas.pdf | Samanta y Chidella 2025, DOI 10.1093/qjmam/hbaf001; DeepXDE 1.12.2; benchmark Featflow DFG Re20 | Onda exacta propia sobre dominio finito, solución BDF/FD del contrato Allen–Cahn y geometría DFG redibujada. No es simulación CFD ni entrenamiento PINN. Código: `generar_figuras.py`. |
| mecanismo.pdf | Diseño del proyecto, apoyado en Rigas 2024/2026 y Southworth et al. 2026 | Arquitectura propuesta; no atribuye la regla conjunta a las fuentes ni presenta evidencia de eficacia. |

Fuentes web directas:

- https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/allen.cahn.html
- https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
- https://arxiv.org/abs/2601.07760v3
- https://repositorio.uvg.edu.gt/entities/publication/75e814c0-864f-4d33-b9df-224f259de11f/full
- https://academic.oup.com/qjmam/article/78/1/hbaf001/8100369

El PDF fuente fue leído de `C:/Users/rodri/Downloads/Tesis_corregida_firmada.pdf`, previa indicación de esa ruta al investigador. El original permanece intacto. La reproducción en el manuscrito responde a la solicitud de incorporar figuras del documento proporcionado; no se afirma una licencia pública adicional. No se trasladaron firmas ni datos de tribunal del antecedente.

## Diferencias documentales que se explicitan

La fuente escribe permitividades negativas y una ecuación con `-epsilon grad(phi)`; combina cortante con un factor 1/2 y describe derivadas normales de tensión/desplazamiento eléctrico como fronteras naturales. El contrato P5 adopta permitividades positivas, cortante ingenieril, tracción y flujo normal. Se documenta como reformulación efectiva consistente, no reproducción exacta ni corrección verificada del código fuente. La consulta web al repositorio vinculado por el preprint no permitió inspeccionar el código; se conserva como contraste posterior sin inventar sus detalles.

La altura total de P5 se fija explícitamente en 0.001 m. La reducción 2D se trata como material efectivo: no se infiere un tensor 3D ni un certificado experimental.

El nombre bibliográfico de la tesis es José Daniel González Carrillo. La hoja del proyecto utiliza **Daniel Gonzales**, tal como indicó el investigador, y **Gabriel Barrientos** como supervisor. Los examinadores no designados quedan sin nombre.

## Estado de evidencia y preprints

La literatura reciente incluye Rigas 2026, RA-PIKAN, AB-PINNs, Free-RBF-KAN y KAN multinivel. No se descartan ideas por estado editorial. RA-PIKAN se cita con la limitación de resumen/ficha pública; Free-RBF-KAN por su ficha v3. Las descripciones de algoritmos no accesibles no se presentan como reproducciones implementadas.

Los cálculos numéricos realizados aquí son verificaciones e ilustraciones clásicas. No se entrenó el mecanismo, no se ejecutó una campaña de P4/P5 y no se aprobó un claim de superioridad. El espacio de problemas y la primera especificación del espacio de soluciones sí están escritos en las secciones integradas.

## Reproducción

`compilar.ps1` compila el documento raíz y ejecuta la verificación. `generar_figuras.py` reproduce las tres figuras nuevas y guarda la solución ilustrativa Allen–Cahn en NPZ. `generar_figuras_previo.py` conserva el generador de las recreaciones reutilizadas; escribe sus salidas en una carpeta `figuras` junto al script, que pueden compararse con las integradas. Las páginas fuente se extraen con `preparar.py`; no hace falta volver a ejecutarlo para compilar el documento.

`integrar.py` registra la operación inicial de integración de secciones. No se debe volver a ejecutar sobre una edición posterior, porque sobrescribiría el texto revisado. El respaldo previo está en `originales/`.
