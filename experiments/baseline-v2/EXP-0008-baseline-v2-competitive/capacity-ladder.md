# Escalera de capacidad (ancho, bases, hidden units) — Burgers y Allen–Cahn

Estado: borrador de EXP-0008. Punto 3 del análisis. Se ejecuta **con la receta
v2 ya elegida en la etapa A1**, nunca con la receta v1: variar capacidad bajo
Adam constante mezclaría dos brechas (optimización y capacidad).

## 1. Por qué existe

La comparación con la literatura no es de igual capacidad:

| Modelo | Parámetros |
|---|---:|
| RBF-PIKAN v1 (R0), MLP tanh 58 (control v1) | 3 648 / 3 655 |
| MLP PINNacle 5×100 (reproducido: CH 5.66e-4 tras Adam→L-BFGS) | 40 801 |

Con 3.6k parámetros, un ancho gaussiano fijo de 0.21 y solo 16 unidades por
capa, la base no puede resolver por sí sola el frente de Burgers (espesor del
orden de `ν = 0.0032`). La igualdad de parámetros con el MLP fue una decisión
metodológica válida para *aislar señales*, pero no es el protocolo de la
literatura. Esta escalera mide cuánta de la brecha residual se debe a
capacidad, y cuál de los tres ejes (ancho, número de bases, hidden) la mueve.

## 2. Definiciones

Modelo `RBF-PIKAN[2,H,H,1]` con `G` bases gaussianas por arista, centros fijos
equiespaciados en `[−1.05, 1.05]`, espaciado `s = 2.1/(G−1)` y ancho
`σ = κ·s`. En v1, `G = 11`, `κ = 1`, `σ = 0.21` (= `s`).

Parámetros: `P(H,G) = (H² + 3H)·(G + 1)` (aristas × (coeficientes + base SiLU)).
MLP `[2,w,w,1]`: `P = w² + 5w + 1`.

## 3. Rungs

| Rung | H | G | κ (σ) | Parámetros | Eje que cambia respecto de R0 |
|---|---:|---:|---|---:|---|
| **R0** | 16 | 11 | 1 (0.210) | 3 648 | — (control v1) |
| R1 | 16 | 21 | 1 (0.105) | 6 688 | resolución de la base |
| R2 | 32 | 11 | 1 (0.210) | 13 440 | hidden |
| R3 | 32 | 21 | 1 (0.105) | 24 640 | hidden + base |
| R4 | 64 | 11 | 1 (0.210) | 51 456 | hidden (≈ tamaño PINNacle) |
| R5 | 32 | 21 | 2 (0.210) | 24 640 | **ancho** aislado (R5 vs R3) |

Lecturas pareadas dentro de la etapa A2 (mismo problema, receta y semillas
dev): R1−R0 (bases), R2−R0 (hidden), R3−R1 y R3−R2 (interacción), R5−R3
(ancho), R4−R2 (escala de hidden). Una rejilla completa 3×3×2 costaría ~3×;
se omite (se declara), a cambio de que la interacción solo se lee en R3.

**Sin rungs con κ < 1**: con `σ < s` las gaussianas dejan huecos entre centros
(la suma no cubre el intervalo), lo que confunde ancho con cobertura.

## 4. Controles de arquitectura bajo la misma receta v2

| Control | Parámetros | Papel |
|---|---:|---|
| MLP tanh `w=58` | 3 655 | par exacto de R0 (v1) |
| MLP tanh `w=154` | 24 487 | par de R3/R5 (Allen–Cahn; solo si el rung elegido es R2/R3/R5) |
| MLP tanh 5×100 (forma PINNacle) | 40 801 | par de la literatura y de R4 |
| PIKAN B-spline cúbica | igual que R0 | control de base explícita; en v2 solo en Heat |

Todos con Xavier-uniform, la misma receta AF-R2, los mismos pools y semillas.
El MLP 5×100 no repite la receta PINNacle (esa ya está reproducida como
baseline B2): aquí mide "MLP grande + receta AF-R2" para separar *receta* de
*arquitectura* en la comparación.

## 5. Diseño de la etapa A2 (semillas dev)

| Problema | Rungs | Semillas dev | Corridas |
|---|---|---|---:|
| Burgers | R1–R5 (R0 viene de A1) | 611, 612 | 10 |
| Allen–Cahn | R0–R5 | 621, 622 | 12 (incluye chequeo de transferencia de receta en R0) |

= 22 corridas. **Recorte previo declarado:** si WP-B0 mide para R4 o R5 más de
2 400 s por corrida, ese rung se excluye del screening y se registra en
`proposals.jsonl` con el motivo (no se ajusta a mitad de tramo).

### Métricas de selección (V, sin abrir E)

1. Primaria: media geométrica del L2 relativo en la grilla **V** tras
   L-BFGS, por problema.
2. Guardas: RMSE en el frente (Burgers) / en snapshots t∈{0.25,0.5,0.75,1}
   (AC), tiempo por corrida, 100 % de semillas finitas.
3. Diagnósticos (no deciden): ocupación de bases, número efectivo de bases,
   saturación de `tanh`, sensibilidad `‖du/dθ‖`, residual regional.

### Regla de elección del rung `R*` (fijada antes de ver datos)

- `R*` = rung con menor métrica primaria **común a ambos problemas** (media
  de las razones `L2(R)/L2(R0)` en Burgers y AC; si un problema no es finito
  para algún rung, ese rung no es elegible).
- Se exige `L2(R*)/L2(R0) ≤ 0.67` (≥1.5×) en ambos problemas; si no, `R* = R0`
  y H2 queda falsada en esa escala.
- Si el mejor rung de cada problema es distinto, se declaran **dos**
  configuraciones finales (una por problema), y el costo de confirmación
  crece solo en los brazos afectados (ver `work-plan.md`).
- Empate (<10 %) → menor costo; si hay empate de costo → menor número de
  parámetros.

## 6. Qué decide cada resultado

| Resultado | Lectura | Siguiente paso |
|---|---|---|
| R0 cumple G2 tras v2 | la brecha era de receta, no de capacidad | congelar R0; la escalera se reporta pero no cambia la confirmación |
| R0 no cumple G2; un rung sí | la capacidad importa | confirmar `R*`; documentar parámetros/costo junto al error |
| R1 o R5 mejoran ≥1.5× | la resolución/ancho de la base es limitante | gatilla la hipótesis de **bases aprendibles** (§7) |
| Solo R2/R4 (hidden) mejoran | la base fija no es el límite; lo es la profundidad efectiva | no se justifica Free-RBF-KAN por ahora |
| Ningún rung mejora | ni receta ni capacidad explican; revisar evaluador/PDE | detener y volver a diagnóstico antes de gastar T3 |

## 7. Bases aprendibles: hipótesis separada (fuera de este preregistro)

Se preregistra como experimento propio (id reservado EXP-0010) **solo si**
R1 o R5 mejoran ≥1.5× sobre R0/R3 en al menos un problema (indicio de que la
geometría de la base es un factor).

Diseño previsto (para que el gatillo sea concreto, no aprobación):

- Mismo modelo y receta AF-R2; centros `c_g` y `log σ_g` entrenables, iniciados
  en la rejilla fija (R0 o `R*`), con lr ×0.1 respecto del resto,
  centros acotados a `[−1.2, 1.2]`, `σ_g ≥ 0.25·s`.
- Emparejado con la versión fija (mismos pools, mismas semillas, mismo
  presupuesto de pasos); se cuentan los `2·G·(aristas)` parámetros extra en el
  reporte de costo.
- Comparador bibliográfico: Free-RBF-KAN (preprint 2026). No se transfiere su
  resultado; su configuración publicada no coincide con estas PDE.
- Señales nuevas: deriva de centros y anchos, solapamiento entre bases,
  número efectivo de bases en el tiempo, cobertura del frente.
