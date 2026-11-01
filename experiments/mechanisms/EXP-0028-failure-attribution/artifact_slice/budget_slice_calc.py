"""EXP-0028 rebanada con artefactos: calculo del presupuesto de budget_slice.md. Parametros = tiempos MEDIDOS en T4 (fuente en el comentario de cada uno; los mismos de budget_1e_calc.py / budget_1d.md);
la unica cifra NO medida es el sobrecosto de las instantaneas (SNAP_S, estimado; el primer kernel lo mide). Despacho: FIFO de 2 ranuras como kaggle_block_slice.py. Cuota = horas de SESION (pared del kernel).
Uso: python budget_slice_calc.py     (imprime las tablas; no escribe nada)"""
import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_block_slice as kb  # noqa: E402

OVERHEAD = {"ns": 40.0, "dfg": 40.0, "ac": 25.0, "kpp": 25.0, "burgers": 25.0}   # arranque, calentamiento de 50 pasos, pools, puntuacion (como 1e)
SNAP_S = {"ns": 40.0, "dfg": 80.0, "ac": 25.0, "kpp": 25.0, "burgers": 50.0}      # ESTIMADO (no medido): ~4 instantaneas completas + ~10 livianas + W&B + historia por termino/gradientes
PREAMBLE_H = {"ns": 0.07, "dfg": 0.07, "ac": 0.03, "kpp": 0.03, "burgers": 0.04}  # prueba --check (3 corridas cortas + calentamiento) en GPU
M = {"ns": 10000, "dfg": 6000, "ac": 10000, "kpp": 8000, "burgers": 10000}
LBFGS = 500


def ns(arm):
    # 1c-t1 (T4): B 0.0535 s/paso, F 0.0555, J/E/I 0.0561; recalculo 5.4 s (en M/1000 + 1 ocasiones); L-BFGS 0.0776 s/it.
    step = {"B": 0.0535, "F": 0.0555}.get(arm, 0.0561)
    pre = (M["ns"] // 1000 + 1) * 5.4 if arm in "JEI" else 0.0
    return M["ns"] * step + pre + LBFGS * 0.0776


def dfg(arm):
    # 1c-dfg (T4, 2000 pasos): B 154.3, F 160.1, J 177.3 (3 recalculos = 19.0 s), E 172.8 (18.7), I 179.2 (20.8); L-BFGS 0.123 s/it.
    adam = {"B": (154.3, 0.0), "F": (160.1, 0.0), "J": (177.3, 19.03 / 3), "E": (172.8, 18.70 / 3), "I": (179.2, 20.84 / 3)}[arm]
    step = (adam[0] - 2 * adam[1]) / 2000
    return M["dfg"] * step + (M["dfg"] // 1000 + 1) * adam[1] + LBFGS * 0.123


def ac(arm):
    # fase 1/1d medido: B 0.0196, F 0.0203, J ~0.0212, E 0.0213 (incluye el recalculo amortizado), I ~0.0213; L-BFGS 500 it. ~ 15 s (1.5 s por 50 it.)
    return M["ac"] * {"B": 0.0196, "F": 0.0203, "J": 0.0212, "E": 0.0213, "I": 0.0213}[arm] + 15


def kpp(arm):
    # fase 1: B 0.0204, F 0.0212, E 0.0223, J 0.0221, I 0.0225
    return M["kpp"] * {"B": 0.0204, "F": 0.0212, "J": 0.0221, "E": 0.0223, "I": 0.0225}[arm] + 15


def burgers(arm):
    # budget_1d: B 0.0782 s/paso a 16 384 pts; recalculo J 3.26 s, I 4.9 s, E 3.5 s (fase 1) x (M/1000 + 1); L-BFGS 500 it. float32 ~ 50 s (estimado: 0.1 s/it.)
    pre = {"J": 3.26, "E": 3.5, "I": 4.9}.get(arm, 0.0) * (M["burgers"] // 1000 + 1)
    return M["burgers"] * 0.0782 * (1.03 if arm == "F" else 1.0) + pre + 50


EST = {"ns": ns, "dfg": dfg, "ac": ac, "kpp": kpp, "burgers": burgers}


def est(p, a):
    return EST[p](a) + OVERHEAD[p] + SNAP_S[p]


def simulate(jobs, n_slots=2):
    free = [0.0] * n_slots
    heapq.heapify(free)
    end = 0.0
    for p, a, s in jobs:
        t = heapq.heappop(free)
        e = t + est(p, a)
        heapq.heappush(free, e)
        end = max(end, e)
    return end


def main():
    print("estimado por corrida (s), incluye arranque y sobrecosto de instantaneas:")
    for p in EST:
        print(" ", p, "M =", M[p], {a: round(est(p, a)) for a in kb.ARMS})
    tot_h, tot_gpu = 0.0, 0.0
    cum = 0.0
    print("\nbloque  prio  corridas  GPU-s    sesion_h (pared simulada + prueba)  margen al kill 3.9 h  acumulado")
    for i, b in enumerate(kb.PLAN, 1):
        jobs = kb.make_jobs(b)
        gpu = sum(est(p, a) for p, a, _ in jobs)
        pre = sum(PREAMBLE_H[p] for p, _, _ in kb.PLAN[b])
        h = simulate(jobs) / 3600 + pre
        tot_h += h
        tot_gpu += gpu
        cum += h
        print(f"{b:6s}  {i}     {len(jobs):3d}      {gpu:7.0f}  {h:5.2f}                  margen {3.9 - h:4.2f} h    {cum:5.2f}")
    print(f"\nTOTAL: {tot_gpu:.0f} GPU-s = {tot_gpu / 3600:.1f} GPU-h (dos T4 ocupadas); {tot_h:.2f} h de sesion (cuota)")
    for p in EST:
        jobs = [j for j in sum((kb.make_jobs(b) for b in kb.PLAN), []) if j[0] == p]
        print(f"  {p:8s} {sum(est(p, a) for _, a, _ in jobs) / 3600:5.2f} GPU-h")


if __name__ == "__main__":
    main()
