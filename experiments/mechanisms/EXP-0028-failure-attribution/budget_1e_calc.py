"""EXP-0028 Fase 1e: calculo del presupuesto de budget_1e.md. Parametros = tiempos MEDIDOS en T4 (fuente en el comentario de cada uno); el despacho se simula con el mismo orden FIFO de 2 ranuras
que kaggle_block_1e.py. La cuota de Kaggle se cuenta en horas de SESION (pared del kernel, T4x2 = 1 h por hora de pared); «GPU-s» = suma de segundos de corrida (las dos T4 ocupadas).
Uso: python budget_1e_calc.py            (imprime las tablas; no escribe nada)"""
import heapq

import kaggle_block_1e as kb

OVERHEAD = 40.0       # s por proceso: arranque, calentamiento (50 pasos), pools, puntuacion (1c: total_seconds excluye todo esto)
PREAMBLE_H = 0.07     # prueba B == ID en GPU (solo bloques ns/dfg), ~4 min
PRIORITY = {"ns1": 1, "ns2": 1, "dfg1": 1, "dfg2": 1, "dfg3": 1, "acf": 2, "bu1": 3, "bu2": 3, "bu3": 3}


def ns(arm):
    # 1c-t1 (T4, 2000 pasos): B 0.0535 s/paso, F 0.0555, J/E/I 0.0561 (+ recalculo 5.4 s cada 1000 pasos: 20 en Adam); L-BFGS 38.8 s/500 it. = 0.0776 s/it.; timing_ns.json: Adam 0.0518, L-BFGS 0.0742
    step = {"B": 0.0535, "F": 0.0555}.get(arm, 0.0561)
    pre = 20 * 5.4 + 5.4 if arm in "JEI" else 0.0
    return 20000 * step + pre + 500 * 0.0776 + OVERHEAD


def dfg(arm):
    # 1c-dfg (T4, 2000 pasos): B 154.3 s, F 160.1 s, J 177.3 s (3 recalculos = 19.0 s), E 172.8 s (18.7 s), I 179.2 s (20.8 s); L-BFGS 61.5 s/500 it. = 0.123 s/it. x 4 000 it.
    adam = {"B": (154.3, 0.0), "F": (160.1, 0.0), "J": (177.3, 19.03 / 3), "E": (172.8, 18.70 / 3), "I": (179.2, 20.84 / 3)}[arm]
    step = (adam[0] - 2 * adam[1]) / 2000
    pre = 21 * adam[1]
    return 20000 * step + pre + 4000 * 0.123 + OVERHEAD


def ac(arm):
    # 1d medido (T4): E 0.0218 s/paso (872 s de Adam, recalculo incluido), total 884 s; J, I como E (fase 1: J 0.0212, I 0.0213, E 0.0214); L-BFGS por meseta, 26-104 s medidos -> 60 s
    return 40000 * 0.0218 + 60 + 25 if arm in "JI" else 40000 * 0.0202 + 104 + 6


def kpp(arm):
    # fase 1: J 0.0221, I 0.0225 (E 0.0223 / B 0.0204 con AC: factor 1d/fase1 = 1.02); L-BFGS ~110 s (esperado, peor caso 300)
    return 40000 * {"J": 0.0221, "I": 0.0225, "B": 0.0204}[arm] * 1.02 + 110 + 25


def burgers(arm):
    # budget_1d: B 0.0782 s/paso a 16 384 pts; recalculo J 3.26 s, I 4.9 s (fase 1) x 24; L-BFGS en el tope de 1 800 s (peor caso, es lo que supone budget_1d) + recalculo f64 <= 35 s
    pre = {"J": 3.26, "I": 4.9, "B": 0.0}[arm] * 24
    return 24000 * 0.0782 + pre + 1800 + (35 if arm in "JI" else 0) + 25


EST = {"ns": ns, "dfg": dfg, "ac": ac, "kpp": kpp, "burgers": burgers}


def simulate(jobs, n_slots=2):
    free = [0.0] * n_slots
    heapq.heapify(free)
    end = 0.0
    for p, a, s in jobs:
        t = heapq.heappop(free)
        e = t + EST[p](a)
        heapq.heappush(free, e)
        end = max(end, e)
    return end


def main():
    print("estimado por corrida (s):")
    for p in EST:
        print(" ", p, {a: round(EST[p](a)) for a in (("B", "F", "J", "E", "I") if p in ("ns", "dfg") else ("B", "J", "I"))})
    tot_h, tot_gpu = 0.0, 0.0
    cum = {1: 0.0, 2: 0.0, 3: 0.0}
    print("\nbloque  prio  corridas  GPU-s    sesion_h  (pared simulada + preambulo)  margen al kill 3.9 h")
    for b in kb.PLAN:
        jobs = kb.make_jobs(b)
        gpu = sum(EST[p](a) for p, a, _ in jobs)
        pre = PREAMBLE_H if any(p in ("ns", "dfg") for p, _, _ in jobs) else 0.0
        h = simulate(jobs) / 3600 + pre
        tot_h += h
        tot_gpu += gpu
        cum[PRIORITY[b]] += h
        print(f"{b:6s}  {PRIORITY[b]}     {len(jobs):3d}      {gpu:7.0f}  {h:5.2f}     margen {3.9 - h:4.2f} h")
    print(f"\nTOTAL: {tot_gpu:.0f} GPU-s = {tot_gpu / 3600:.1f} GPU-h (dos T4 ocupadas); {tot_h:.2f} h de sesion (cuota)")
    print("por prioridad (h de sesion):", {k: round(v, 2) for k, v in cum.items()}, "acumulado:", [round(sum(list(cum.values())[:i]), 2) for i in (1, 2, 3)])
    print("contingencia B en ac/kpp/burgers (X28_WITH_B=1): +", {p: round(sum(EST[p]("B") for _ in range(5)) / 3600 / 2, 2) for p in ("kpp", "burgers", "ac")}, "h de sesion aprox.")


if __name__ == "__main__":
    main()
