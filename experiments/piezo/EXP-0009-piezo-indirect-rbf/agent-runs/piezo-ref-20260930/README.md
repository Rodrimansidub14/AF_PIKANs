# Referencia piezo: PINN de Daniel González (inspección, 2026-09-30)

Clon de solo lectura de https://github.com/Daniel14gonc/PINNs_piezoelectricity (commit `b90b19c`, 2026-06-17) en `daniel-ref/`. **No se modificó ni se integra a `src/`.** Nuestra arquitectura sigue siendo RBF-PIKAN; esto es solo referencia y comparador externo. Todo se ejecutó en CPU, sin Kaggle. Exploratorio; nada aquí acepta referencias ni claims.

Scripts propios (en esta carpeta): `check_daniel.py`, `fem_grid.py`, `eval_paper_weights.py`. Resultados: `daniel-checks.json`, `paper-weights-vs-fem.json`, `paper-weights-direct-scale.json`, `fem-grid-*.npz`. scikit-fem 12.0.2 se tomó de `../teoria-visual-v2/vendor` (ya estaba en el repo; no se instaló nada). Python 3.12 para el FEM y 3.13 para torch.

## Qué trae el repo

- Dos formulaciones 2D de una viga bimorfa de PVDF (L=0,1 m, H=1 mm, apoyada a la izquierda): **indirecta** (100 V arriba, 0 abajo, se predice la deformación) y **directa** (fuerza en el extremo, se predice el voltaje).
- PINN formulación mixta: la red da 8 salidas (u, v, φ, σx, σz, τxz, Dx, Dy), MLP tanh con Glorot normal y forma de pirámide (100, 250) o uniforme; condiciones de borde duras; Adam y después L-BFGS; pesos adaptativos entre PDE y borde. Float64.
- Pesos del paper committeados (`models/*/model_PINN_*_paper_3.pt`).
- **Solucionador FEM propio** (`fem.py`, scikit-fem, P2 triangular, forma e-stress-charge con polarización opuesta en las dos capas) añadido en la revisión del paper. El `FEM.csv` del paper original (exportación externa) **no está en el repo** (gitignorado).
- Licencia: `pyproject.toml` declara MIT, pero **no hay archivo LICENSE**. (Corrijo lo que dije antes: la página de GitHub no la mostraba, el metadato sí la declara.) Confirmar con Daniel antes de reutilizar código.
- `requirements.txt` no lista scikit-fem aunque `fem.py` lo necesita.

## Verificaciones hechas

1. **Coeficientes de tu metodología P5 = los del repo.** C11 2.1836e9, C12 0.6332e9, G 0.775e9, e21 0.02904, e22 0.05157 y κ 1.0625e-10/1.0406e-10 coinciden con `materials.py` dentro de 0,05 %.
   **Discrepancia de signo a revisar:** en el repo e31 = +0.02904 y e33 = −0.05158 (signos opuestos, como corresponde a d31 > 0 y d33 < 0 del PVDF). Tu tabla `tab:pvdf` lista e21 y e22 **con el mismo signo** (−0.02904, −0.05157 arriba; + abajo). Hay que contrastarlo con las tablas 4–5 de la tesis de González Carrillo (PDF que tienes tú). No lo cambié en `ProyectoGrad`.
2. **Su FEM converge y pasa dos chequeos de mano** (mallas 100×4, 200×8, 400×16):
   - Indirecto: flecha en la punta 3.2875e-5 → 3.2892e-5 → 3.2896e-5 m (cambio 0.06 % entre las dos finas). Estimación de bimorfo (curvatura 3·Δε/2H con ε = d31·E) da 3.3e-5 m.
   - Directo con 1 N/m: flecha 1.98e-3 m (viga de Euler–Bernoulli da 2.0e-3) y voltaje generado de −61 V (estimación del propio README: ~60 V).
   Tiempo: 0.4 s a 3 s por solución en CPU.
3. **Sus pesos del paper contra su propio FEM** (201×21 puntos, CPU, float64):

| Formulación | Campo | L2 relativo | Nota |
|---|---|---|---|
| Indirecta, `poling_sign=−1` | u | 0.195 | reproduce exactamente lo que dice su README (~0.19 u/v, ~1e-3 φ) |
| | v | 0.193 | |
| | φ | 1.0e-3 | |
| Indirecta, `poling_sign=+1` (default de su FEM) | u, v | 1.81 | el PINN dobla en sentido contrario |
| Directa (FEM a 0.1 N) | u / v / φ | 0.88 / 0.89 / 0.78 | |

   - Indirecta: el modelo solo coincide con el FEM si se invierte el signo de polarización; el default de su solucionador da el sentido opuesto. Es una convención, no un error de la física, pero hay que fijarla antes de comparar.
   - Directa: la forma del desplazamiento coincide (correlación 0.997 en v) pero la amplitud es ~0.11 veces la del FEM a 0.1 N (la fuerza con que entrenaron no la pude determinar), y el potencial φ no coincide (correlación 0.50). Su README marca explícitamente esta discrepancia del efecto directo como pendiente. **La formulación indirecta es el comparador utilizable; la directa no lo es todavía.**

## Qué significa para piezo con RBF-PIKAN

- **Comparador creíble**: el brazo de referencia sería su PINN indirecta (MLP 100-250, 8 salidas, ~0.19 L2 en u y v). Es un error alto, con margen para medir mejoras.
- **Referencia de campo**: su solucionador FEM sirve para generar la vara, pero lo escribió el mismo grupo que la PINN y sus chequeos son de mano, no de tercero. Para mantener tu criterio de referencia de terceros haría falta contrastarlo con COMSOL o con la solución analítica de bimorfo (se puede hacer sin GPU).
- **Formulación**: usa salidas mixtas de primer orden (las tensiones y el desplazamiento eléctrico como salidas). Nuestra RBF-PIKAN tendría que adoptar la misma parametrización de salidas para comparar de forma justa, o compararse con su "caso A estándar" (hay un `indirect/standard.py`). Decisión de diseño pendiente.
- **Pendiente antes de usar**: licencia (archivo LICENSE), permiso de Daniel, y la convención de polarización y de signos de e.
