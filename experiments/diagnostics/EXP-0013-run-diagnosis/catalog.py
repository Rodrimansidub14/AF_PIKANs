#!/usr/bin/env python3
"""EXP-0013 — genera results/signals-catalog.md: definición, tipo (señal/etiqueta), ¿usa referencia?, y fuente de cada columna calculada."""
from __future__ import annotations

import csv
import re

from common import RESULTS

RULES = [  # (regex, definición, usa_referencia)
    (r"^res/(all|pde|momentum_x|momentum_y|continuity|constitutive_\w+|equilibrium_\w+|gauss)/rms$", "RMS del residuo de la PDE (por punto, escalado por la escala del contrato) sobre la malla del problema", "no"),
    (r"/(p50|p95|p99|max)$", "percentil / máximo de |residuo|", "no"),
    (r"/gini$", "coeficiente de Gini de |residuo| (0 = uniforme, 1 = concentrado en un punto); en `err/` es el de |error|", "no (res/) / sí (err/)"),
    (r"/top1_mass$|/top5_mass$", "fracción de la masa Σ r² (o Σ e²) contenida en el 1 % / 5 % de puntos con mayor valor", "no (res/) / sí (err/)"),
    (r"/entropy_norm$", "entropía de Shannon normalizada de r² (o e²) sobre los puntos (1 = uniforme)", "no (res/) / sí (err/)"),
    (r"/kurtosis$", "curtosis (exceso) de |residuo|", "no"),
    (r"^res/t_quartile\d_mass$", "fracción de la masa del residuo en cada cuartil de tiempo (problemas 1D-tiempo)", "no"),
    (r"^res/(x|t)_centroid$", "centroide de la masa del residuo en x o t", "no"),
    (r"^res/n_points$", "número de puntos de la malla", "no"),
    (r"^struct/", "estructura de la solución del propio modelo: máx. y p95 de |∇u|, máx. |Δu|, anchura de estructura 2·max|u|/max|∂u/∂xi| por dimensión", "no"),
    (r"^align/spearman_res_(gradnorm|laplacian)$", "correlación de Spearman entre |residuo| y |∇u| (o |Δu|) del propio modelo", "no"),
    (r"^align/top5res_in_top5grad$", "fracción del 5 % superior de |residuo| que cae en el 5 % superior de |∇u|", "no"),
    (r"^basis/l\d/occ_(mean|min)$", "ocupación de bases RBF de la capa l: media y mínima de la activación media por base", "no"),
    (r"^basis/l\d/dead_fraction$", "fracción de pares (arista, base) cuya activación máxima sobre la malla es < 0.01 (bases muertas)", "no"),
    (r"^basis/l\d/eff_bases$", "número efectivo de bases exp(H) de la ocupación", "no"),
    (r"^basis/l\d/(coef_norm|small_edge_fraction|coef_basis_gini|tanh_saturation)$", "norma de coeficientes; fracción de aristas con norma < 1e-3·máx; Gini de la norma por base; fracción de activaciones tanh saturadas (|h|>0.99)", "no"),
    (r"^scale/(spacing|width)_phys_", "espaciado y ancho de las bases RBF de la capa 0 en unidades físicas por dimensión de entrada", "no"),
    (r"^scale/ratio_struct_to_(spacing|width)_", "anchura de estructura del modelo dividida por el espaciado/ancho físico de bases de la capa 0 (< 1 = estructura más fina que la base)", "no"),
    (r"^jac/", "espectro del jacobiano ∂u/∂θ en 256 puntos fijos: rango efectivo exp(H(σ/Σσ)), por parámetro, número de condición, σmax", "no"),
    (r"^param/", "cuenta, norma, máximo y fracción pequeña (<1e-3) de los parámetros", "no"),
    (r"^gradterm/norm_", "norma del gradiente ∇θ de ½·media(r_c²) por componente residual c (1 024 puntos)", "no"),
    (r"^gradterm/cos_|^gradterm/norm_ratio", "coseno entre gradientes de componentes residuales; razón máx/mín de normas", "no"),
    (r"^heat/", "energía ∫u²dx, tasa de decaimiento ajustada y su razón con el valor de la PDE (−2απ²), |u|máx (propiedades de la PDE, no de la solución exacta)", "no"),
    (r"^burgers/", "antisimetría u(−x,t)+u(x,t), ux máx tardío, anchura mínima del frente, posición media del frente", "no"),
    (r"^ac/", "sobrepaso de |u|>1, número de cruces por cero, energía de Ginzburg–Landau y fracción de pasos con aumento de energía", "no"),
    (r"^dfg/", "caudal en entrada, salida y x=1.1, y desbalance relativo salida−entrada", "no"),
    (r"^piezo/", "magnitudes del modelo (RMS de deformación, u, v, σxx, flecha en la punta)", "no"),
    (r"^train/", "señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos)", "no"),
    (r"^err/", "ETIQUETA: error relativo L2 por componente, plano y normalizado, y concentración del error", "sí"),
    (r"^region/", "ETIQUETA: masa y concentración del error y del residuo en regiones definidas con la referencia (steep5, interfaz, cerca del cilindro, capas, …)", "sí"),
    (r"^audit/", "AUDITORÍA: alineación residuo–error (Spearman, solape de 5 % superiores, masa de error en el 5 % superior del residuo)", "sí"),
]


def describe(k: str):
    for rx, d, ref in RULES:
        if re.search(rx, k):
            return d, ref
    return "(sin descripción registrada)", "?"


def main():
    keys = []
    for fn in ("signals_wide.csv", "labels_wide.csv"):
        p = RESULTS / fn
        if p.exists():
            head = next(csv.reader(open(p, encoding="utf-8")))
            keys += [(fn, c) for c in head if "/" in c and not c.startswith("train/meta")]
    lines = ["# Catálogo de señales y etiquetas (EXP-0013)", "",
             "Generado automáticamente por `catalog.py` a partir de las columnas calculadas. **Regla:** las señales (`res/`, `struct/`, `align/`, `basis/`, `scale/`, `jac/`, `param/`, `gradterm/`, problema, `train/`) no usan la referencia; las etiquetas (`err/`, `region/`) y la auditoría (`audit/`) sí, y solo sirven como variable a explicar.", "",
             "Notas: (1) los checkpoints float32 se promueven a float64 para calcular señales; (2) las señales de bases solo existen para modelos RBF; (3) `gradterm/` solo existe con ≥ 2 componentes residuales; (4) la anchura de estructura en la dirección delgada de piezo no es físicamente interpretable (ver ficha).", "",
             "| Columna | Origen | Definición | ¿Usa referencia? |", "|---|---|---|---|"]
    seen = set()
    for fn, k in keys:
        base = re.sub(r"basis/l\d/", "basis/lN/", k)
        base = re.sub(r"gradterm/(norm|cos)_.*", r"gradterm/\1_*", base)
        base = re.sub(r"res/(momentum_x|momentum_y|continuity|constitutive_\w+|equilibrium_\w+|gauss|pde)/", "res/<componente>/", base)
        base = re.sub(r"region/[A-Za-z0-9_]+/", "region/<región>/", base)
        base = re.sub(r"scale/(spacing_phys|width_phys|ratio_struct_to_spacing|ratio_struct_to_width)_\w+", r"scale/\1_<dim>", base)
        base = re.sub(r"struct/width_\w+", "struct/width_<dim>", base)
        base = re.sub(r"err/(u|v|p|phi)_rel_l2", "err/<comp>_rel_l2", base)
        if base in seen:
            continue
        seen.add(base)
        d, ref = describe(k)
        lines.append(f"| `{base}` | {fn.split('_')[0]} | {d} | {ref} |")
    (RESULTS / "signals-catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(len(seen), "entradas en el catálogo")


if __name__ == "__main__":
    main()
