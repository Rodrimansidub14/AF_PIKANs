---
title: "Fuentes oficiales para la pila de visualización SciML"
year: 2026
venue: "Documentación oficial consultada el 23 de septiembre de 2026"
layer: APP
status: done
source_role: literature_scout
tags: #app #visualization #pyvista #plotly #panel #scientific-computing
---

# Fuentes Oficiales para la Pila de Visualización SciML

**Consulta de documentación oficial · 23 de septiembre de 2026**

> Registro bibliográfico de herramientas de visualización que se aplican o se consideran para las campañas SciML de AF_PIKANs.

## Links

### Papers relacionados
- [[Takamoto_2022_PDEBench]]
- [[Wei_2024_PDENNEval]]

### Dominios
- [[Fluidos_y_Mecánica_Estructural]]

### Conceptos
- [[Adaptive_Collocation]]

## Fuentes oficiales

1. PyVista, guía de visualización y configuración off-screen: https://docs.pyvista.org/user-guide/ y https://docs.pyvista.org/user-guide/configuration
2. Plotly.py, gráficos científicos e interacción en notebooks: https://plotly.com/python/ y https://plotly.com/python/getting-started/
3. Panel, creación de aplicaciones y dashboards en Python: https://panel.holoviz.org/getting_started/
4. VisPy, visualización científica interactiva 2D/3D: https://vispy.org/
5. Taichi, GUI y visualización de datos de simulación: https://docs.taichi-lang.org/docs/gui_system
6. Makie.jl, ecosistema de visualización interactiva para Julia: https://docs.makie.org/dev/ y https://doi.org/10.21105/joss.03349
7. napari, visualización de arreglos de imágenes n-dimensionales: https://napari.org/stable/getting_started/viewer.html
8. W3C WebGPU, API para gráficos y cómputo en GPU desde la web; documento Candidate Recommendation Draft del 15 de septiembre de 2026: https://www.w3.org/TR/webgpu/

## BibTeX de herramientas

    @manual{pyvista_docs,
      title = {PyVista Documentation},
      organization = {PyVista Developers},
      url = {https://docs.pyvista.org/user-guide/},
      note = {Consultado 2026-09-23}
    }

    @manual{plotly_python_docs,
      title = {Plotly Python Documentation},
      organization = {Plotly},
      url = {https://plotly.com/python/},
      note = {Consultado 2026-09-23}
    }

    @manual{panel_docs,
      title = {Panel Documentation},
      organization = {HoloViz},
      url = {https://panel.holoviz.org/getting_started/},
      note = {Consultado 2026-09-23}
    }

    @manual{vispy_docs,
      title = {VisPy Documentation},
      organization = {VisPy Developers},
      url = {https://vispy.org/},
      note = {Consultado 2026-09-23}
    }

    @manual{taichi_docs,
      title = {Taichi Documentation},
      organization = {Taichi Developers},
      url = {https://docs.taichi-lang.org/docs/gui_system},
      note = {Consultado 2026-09-23}
    }

    @article{danisch2021makie,
      title = {Makie.jl: Flexible high-performance data visualization for Julia},
      author = {Danisch, Simon and Krumbiegel, Julius},
      journal = {Journal of Open Source Software},
      volume = {6},
      number = {65},
      pages = {3349},
      year = {2021},
      doi = {10.21105/joss.03349}
    }

    @manual{napari_docs,
      title = {napari Documentation},
      organization = {napari contributors},
      url = {https://napari.org/stable/},
      note = {Consultado 2026-09-23}
    }

    @misc{w3c2026webgpu,
      title = {WebGPU},
      organization = {World Wide Web Consortium},
      howpublished = {W3C Candidate Recommendation Draft},
      year = {2026},
      url = {https://www.w3.org/TR/webgpu/},
      note = {Documento publicado el 15 de septiembre de 2026; consultado 2026-09-23}
    }

## Fuentes de ejecución y documentación citadas en el brief

9. PINA, documentación oficial del framework SciML basado en PyTorch: https://pina-org.github.io/PINA/ (consulta 2026-09-23).
10. Weights & Biases, documentación de experiment tracking y artefactos: https://docs.wandb.ai/ (consulta 2026-09-23).
11. MIT OpenCourseWare, 18.085, *Problem Set 8 solutions*: derivación/solución de conducción 1D usada para contextualizar Heat de varilla: https://math.mit.edu/classes/18.085/summer2014/pset/pset8_sol.pdf (consulta 2026-09-23).

    @manual{pina_docs,
      title = {PINA Documentation},
      organization = {PINA Developers},
      url = {https://pina-org.github.io/PINA/},
      note = {Consultado 2026-09-23}
    }

    @manual{wandb_docs,
      title = {Weights & Biases Documentation},
      organization = {Weights & Biases},
      url = {https://docs.wandb.ai/},
      note = {Consultado 2026-09-23}
    }

    @misc{mit_ocw_18085_pset8,
      title = {18.085 Problem Set 8 Solutions},
      organization = {MIT OpenCourseWare},
      url = {https://math.mit.edu/classes/18.085/summer2014/pset/pset8_sol.pdf},
      note = {Consultado 2026-09-23; fuente docente para la solución Heat 1D}
    }
