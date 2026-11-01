# Parámetros de la receta de Daniel (PINN piezoeléctrica indirecta)

Este documento conserva solo los **parámetros** necesarios para reproducir la receta, no su código. Los valores se leyeron de `PINNs_piezoelectricity` en el commit `e7856f1` (2026-10-02), de González Carrillo.

## Procedencia y licencia

- Repositorio original: https://github.com/Daniel14gonc/PINNs_piezoelectricity
- Copia fijada (fork): https://github.com/Rodrimansidub14/PINNs_piezoelectricity, mismo commit `e7856f1`.
- Licencia: el repositorio original no trae archivo `LICENSE` (GitHub no detecta licencia). Su `pyproject.toml` declara MIT. El código no se redistribuye aquí, solo se enlaza al fork. Los valores numéricos de abajo son datos, no código.

## Cómo reproducir las corridas

`daniel_arm.py` (en esta carpeta) importa el código de Daniel sin modificarlo desde un clon:

```bash
git clone https://github.com/Rodrimansidub14/PINNs_piezoelectricity
git -C PINNs_piezoelectricity checkout e7856f1
python daniel_arm.py --daniel-root PINNs_piezoelectricity --arm mlp --seed 20260728 --epochs 20000 --out DIR --fem fem.npz
```

Las corridas de EXP-0009 que usaron esta receta guardaron antes una copia del código en `results/daniel-*/payload/daniel/`. Esas copias se retiraron del repositorio. Los resultados (`results/*/results/*/run.json`) se conservan.

## Geometría y escalas (`config.py`)

| Parámetro | Valor |
|---|---|
| Ancho de la viga | 100e-3 m |
| Alto total | 1e-3 m (capas de igual espesor, interfaz en H/2) |
| Voltaje de referencia | 100 V |
| Fuerza de referencia (efecto directo) | 0.1 N (1 N solo como verificación de linealidad) |
| Signo de polarización canónico | -1.0 |
| `U_c`, `SIGMA_C`, `D_c` | 1e-9, 1e-6, 8.854e-7 |
| `L0`, `SIGMA_0`, `PHI_0`, `D0` | 1e-3 m, 1e9 Pa, 1 V, 1e-3 C/m² |

## Material PVDF (`materials.py`)

| Parámetro | Valor |
|---|---|
| Módulo elástico E | 2.0e9 N/m² |
| Coeficiente de Poisson ν | 0.29 |
| Módulo de corte G | 0.775e9 N/m² |
| d31, d33 | 2.2e-11, -3.0e-11 C/N |
| Permitividad relativa | 12 |
| Permitividad del vacío | 8.854187817620e-12 F/m |

## Receta de entrenamiento (cabecera de `daniel_arm.py`, leída del repositorio)

- Perceptrón propio de Daniel: 3 capas de 50 neuronas con SiLU, tronco compartido enriquecido en fase. Alternativa: tronco RBF-PIKAN en `model.net`.
- Muestreo por capas con 2 048 puntos interiores por capa, puntos de interfaz y de frontera, remuestreo cada 200 épocas.
- Escalas por campo, enrutamiento de gradiente detenido en la constitutiva mecánica y levantamiento duro del voltaje.
- 20 000 épocas de Adam con tasa de aprendizaje 1e-3, doble precisión, sin L-BFGS.
- Evaluación contra la rejilla FEM con signo de polarización -1, común al perceptrón y a la RBF.

## Cómo se configuraron las corridas

Todas las corridas con la receta de Daniel usaron el commit `e7856f1`, 20 000 épocas de Adam en doble precisión y dos GPU Tesla T4 por sesión de Kaggle. Cada una guarda su configuración y su commit de origen en su `run.json`. Los brazos se evalúan con el mismo evaluador contra la rejilla FEM.

| Brazo | Arquitectura | Parámetros | Semillas | Sesión (carpeta en `results/`) |
|---|---|---|---|---|
| `mlp` | Perceptrón propio de Daniel | 5 758 | 20260728 | `daniel-recipe-20261003` |
| | | | 20260729, 20260730 | `daniel-seeds-20261004` |
| | | | 20260731, 20260732 | `daniel-seeds2-20261004` |
| `rbf` | Tronco RBF-PIKAN de ancho 16 en lugar de `model.net` | 5 376 | 20260728 | `daniel-recipe-20261003` |
| | | | 20260729, 20260730 | `daniel-seeds-20261004` |
| | | | 20260731, 20260732 | `daniel-seeds2-20261004` |

Los tiempos van de unos 3 900 a 4 100 s para el perceptrón y de unos 9 200 a 9 700 s para la RBF. Las corridas se marcaron `DRAFT_UNVERIFIED` y no aceptan por sí solas ningún resultado (`scientific_accepted` y `reference_accepted` en falso).
