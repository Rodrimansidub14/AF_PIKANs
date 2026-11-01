---
experiment_id: EXP-0029
decision: BORRADOR_SIN_FIRMAR   # propuesta del agente: not_supported (con confusión de hardware declarada)
claims: []
decided_by: pendiente (Rodrigo Mansilla)
date: pendiente
---

# Borrador de decisión, EXP-0029 (mecanismo final M en piezo)

**Estado: borrador del agente, sin firmar.** Para que sea decisión, copiar a `decision.md` (mismo directorio), cambiar `decision` y `decided_by`/`date`, y firmar. Si no se está de acuerdo, editar libremente: nada aquí es vinculante.

## Qué se firmó antes de correr
`EXP-0029_spec.md` (firma en el chat del 2026-10-26: «firmo N, no hacen falta pruebas, córrelo») fijó la hipótesis H-FINAL y cinco predicciones en `DISENO_FINAL.md` §3. Presupuesto de datos: converso y directo, semillas 20260728 a 20260732, Adam 20 000, float64, sin L-BFGS.

## Qué se corrió
10 de las 12 corridas previstas del bloque final (Modal A100): M converso 28 a 30, M directo 28 a 30, B y E converso 31 y 32. **M converso 31 y 32 no corrieron** (OOM en A100 de 40 GB). B y E directo 31 y 32 no se corrieron. Cómputo cerrado por decisión del investigador.

## Resultado frente a cada predicción (`results/EXP0029_vs_predicciones.md`)

| # | Predicción firmada | Resultado |
|---|---|---|
| 1 | Converso: u(M)/u(E) en [0.85, 1.15] y u(M)/u(B) ≤ 0.5 | **No se cumple.** M/E = 6.7, 2.2, 5.0; M/B mediana 0.92. M se comporta como B |
| 2 | Directo: u(M)/u(E) en [0.85, 1.15] y ninguna semilla > 1.5 | **Formalmente falsada** (2 de 3 fuera del intervalo, una a cada lado); sí cumple el tope de 1.5. M ≤ E en 2 de 3; medias M 0.180, E 0.193 |
| 3 | Converso: φ(M)/φ(E) ≤ 1.20 | **Se cumple** (0.78, 0.83, 0.86) |
| 4 | Directo: saltos de M ≤ saltos de E + 3 | **No evaluable** (las corridas viejas de E registran la pérdida cada 100 pasos) |
| 5 | Costo por paso M ≤ 1.10 × E | **Se cumple por poco** (1.09 en A100, semillas 31 y 32) |

## Decisión propuesta
**H-FINAL no queda respaldada en el converso, y queda formalmente falsada en el directo** según los criterios firmados. Propongo registrar `not_supported` y no `rejected`, por una razón de método que la caracterización posterior hizo visible:

- M corrió solo en A100. E de las semillas 28 a 30 corrió en T4 y H100. No hay un solo par M/E con el mismo hardware (`EXP-0030/results/A2_mismo_hardware.md`). Las únicas semillas de E en A100 (31 y 32) tampoco muestran la ventaja de E sobre B.
- Por tanto el criterio firmado no se cumple, pero **no se puede atribuir la falla a las patas** de M: puede ser hardware, semillas o las patas.

## Lo que sí se aprendió (descriptivo, post hoc)
- P4 (guarda de saltos) dañó la pérdida directa 2 a 3 veces; en M quedó solo observando.
- P2 (balance por grupos) está mal planteada en el converso: la parte eléctrica pesa ≈ 0 en la traza.
- P5 fijo (arranque denso en 250 y 500) no ayudó; queda la idea M2 (P5 disparado por cambio de cuotas), **no corrida**.
- Con n = 5 en B y E converso, las medias del último paso son iguales (0.152 y 0.158); la ventaja de E está en salir antes del llano y en la cola. Métrica robusta = post hoc.

## Límites que deben acompañar cualquier cita
n = 3 a 5 por brazo, hardware mixto (T4, H100, A100) y entornos distintos (torch 2.11 y 2.14.1), sin p-valores, un problema (piezo) con dos objetivos, M converso con 3 semillas.

## Claims
Ninguno aceptado aquí. Candidatos para `registry/claims.jsonl`: ver `docs/plans/2026-10-26-claims-candidatos.md`.
