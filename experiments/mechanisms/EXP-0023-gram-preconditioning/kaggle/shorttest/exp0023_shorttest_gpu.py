"""EXP-0023, prueba previa 7b repetida tras la Enmienda 1 (P normalizado) y 7a de nuevo. Prueba corta; NO lanza el bloque.

Cuatro procesos en paralelo: Burgers y Allen-Cahn en GPU (T4 0 y 1) y los mismos en CPU. Semilla 1.
Prueba 7a (estado compartido, float64, N = 512): E, I y J recalculan sus bloques sobre el mismo estado en CPU y GPU; criterio: diferencia relativa maxima de P, c' y salida < 1e-6.
Prueba 7b (Enmienda 1, criterio fijado antes de correr): cinco brazos B/J/E/I/F, dos problemas, CPU y GPU, 100 pasos de Adam (recalculo cada 50: pasos 0 y 50, N = 512; despues
  20 iteraciones de L-BFGS solo como dato). Pasa si la perdida es finita y, en el paso 100, <= 2x la perdida del brazo B (del mismo dispositivo). No hay segunda enmienda.
El presupuesto (prueba 6) ya se midio en la corrida anterior y no se repite. Ninguna parte puntua hipotesis. Variables de entorno (solo para probar en CPU): SHORTTEST_TINY=1, BUNDLE_DIR, OUT_DIR.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

TINY = os.environ.get("SHORTTEST_TINY") == "1"
bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("pre_trainer.py")).parent)
out = Path(os.environ.get("OUT_DIR", "/kaggle/working/exp0023_shorttest"))
out.mkdir(parents=True, exist_ok=True)
ADAM_B, ITERS_B, EVERY_B, N_B = (10, 5, 5, 64) if TINY else (100, 20, 50, 512)
STEPS_BUDGET, REPS, N_BUDGET = (5, 1, 64) if TINY else (200, 3, 2048)
TOL_STATE = 1e-6


def worker(problem, device, tag):
    """Un proceso: calentamiento, (GPU) presupuesto, estado compartido float64 y corridas cortas."""
    import torch
    sys.path.insert(0, str(bundle))
    os.environ["BUNDLE_DIR"] = str(bundle)
    import pre_trainer as pt
    import measure_budget as mb
    dev = torch.device(device)
    res = dict(problem=problem, device=device)
    pt.t1.warmup_t1(problem, device, "float32", 10 if TINY else 50)
    cfg = pt.PROBLEMS[problem]
    for arm in ("E", "I", "J"):                                           # calienta los nucleos de los bloques en ese dispositivo
        prob = pt.pc.make_problem(cfg["name"], str(cfg["ref"]))
        m = prob.build("rbf", cfg["width"], 11, 2).to(device=dev, dtype=torch.float32)
        P = prob.pools(2, dev, torch.float32)
        pre = pt.gp.GramPreconditioner(arm, n_points=64)
        pre.attach(m)
        pre.refresh(m, prob, P["residual"])
    is_gpu = tag.endswith("-gpu")
    # prueba 7a: estado compartido en float64
    st = out / f"state-{problem}.pt"
    if not is_gpu:
        prob = pt.pc.make_problem(cfg["name"], str(cfg["ref"]))
        m = prob.build("rbf", cfg["width"], 11, 1).to(dtype=torch.float64)
        P = prob.pools(1, torch.device("cpu"), torch.float64)
        pt.adam_with_pre(prob, m, P, 5 if TINY else 30, None)
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, st)
    else:
        while not st.exists():
            time.sleep(2)
        time.sleep(2)
    saved = torch.load(st, map_location="cpu")
    shared = {}
    for arm in ("E", "I", "J"):
        prob = pt.pc.make_problem(cfg["name"], str(cfg["ref"]))
        m = prob.build("rbf", cfg["width"], 11, 1).to(dtype=torch.float64)
        m.load_state_dict(saved)
        m = m.to(dev)
        P = prob.pools(1, dev, torch.float64)
        pre = pt.gp.GramPreconditioner(arm, n_points=64 if TINY else 512)
        pre.attach(m)
        pre.refresh(m, prob, P["residual"])
        x = prob.pools(1, dev, torch.float64)["residual"][:256]
        with torch.no_grad():
            y = m(x).cpu()
        shared[arm] = dict(P=[l.pre_P.detach().cpu() for l in m.layers], cp=[l.coefficients.detach().cpu() for l in m.layers], y=y)
    torch.save(shared, out / f"shared-{tag}.pt")
    # prueba 7b: corridas cortas
    runs = {}
    for arm in ("B", "J", "E", "I", "F"):
        runs[arm] = pt.run_one(problem, arm, 1, ADAM_B, ITERS_B, device, "float32", pre_kw=dict(every=EVERY_B, n_points=N_B))
        print(problem, device, arm, runs[arm].get("primary_adam"), runs[arm].get("primary_final"), runs[arm].get("total_seconds"), flush=True)
    res["runs"] = runs
    (out / f"{tag}.json").write_text(json.dumps(res))


def launch(problem, device, slot, tag):
    env = dict(os.environ, BUNDLE_DIR=str(bundle))
    if device == "cpu" or TINY:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = "2"
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(slot)
    with open(out / f"{tag}.log", "w") as log:
        return subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", problem, "cpu" if (device == "cpu" or TINY) else "cuda:0", tag],
                              env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(bundle)).returncode


def rel(a, b):
    import torch
    return float(torch.linalg.norm((a.double() - b.double()).reshape(-1)) / torch.linalg.norm(b.double().reshape(-1)).clamp_min(1e-300))


def report():
    import torch
    rc, rep, ok = {}, {}, True
    for problem in ("burgers", "ac"):
        g = out / f"{problem}-gpu.json"
        c = out / f"{problem}-cpu.json"
        entry = {}
        if not (g.exists() and c.exists()):
            rep[problem] = dict(error="faltan resultados", gpu=g.exists(), cpu=c.exists())
            ok = False
            continue
        G, C = json.loads(g.read_text()), json.loads(c.read_text())
        sg, sc = torch.load(out / f"shared-{problem}-gpu.pt"), torch.load(out / f"shared-{problem}-cpu.pt")
        shared = {}
        for arm in sg:
            d = max(max(rel(a, b) for a, b in zip(sg[arm]["P"], sc[arm]["P"])), max(rel(a, b) for a, b in zip(sg[arm]["cp"], sc[arm]["cp"])), rel(sg[arm]["y"], sc[arm]["y"]))
            shared[arm] = dict(max_rel_diff=d, within_tolerance=bool(d < TOL_STATE))
            ok &= shared[arm]["within_tolerance"]
        entry["test7a_shared_state_float64"] = shared
        runs = {}
        for dev_name, R in (("gpu", G["runs"]), ("cpu", C["runs"])):
            lb = R["B"]["hist"][-1]["loss"] if R["B"].get("hist") else None
            for arm in ("B", "J", "E", "I", "F"):
                r = R[arm]
                l100 = r["hist"][-1]["loss"] if r.get("hist") else None
                fin = bool(l100 is not None and l100 == l100 and l100 != float("inf") and r.get("adam_status") == "adam_complete")
                runs[f"{arm}-{dev_name}"] = dict(adam_status=r.get("adam_status"), loss_step100=l100, ratio_to_B=(l100 / lb if fin and lb else None),
                                                 primary_adam=r.get("primary_adam"), primary_final=r.get("primary_final"), lbfgs_status=r.get("lbfgs_status"),
                                                 pre_seconds=r.get("pre_seconds"), passes=bool(fin and lb and l100 <= 2 * lb))
                ok &= runs[f"{arm}-{dev_name}"]["passes"]
        entry["test7b_short_runs_float32"] = runs
        rep[problem] = entry
    rep["all_within_tolerance"] = bool(ok)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4])
        sys.exit(0)
    t0 = time.time()
    jobs = [("burgers", "gpu", 0), ("ac", "gpu", 1), ("burgers", "cpu", 0), ("ac", "cpu", 0)]
    codes = {}

    def go(p, d, s):
        codes[f"{p}-{d}"] = launch("burgers" if p == "burgers" else "ac", d, s, f"{p}-{d}")

    ths = [threading.Thread(target=go, args=j) for j in jobs]
    [t.start() for t in ths]
    [t.join() for t in ths]
    print("return codes", codes, "seconds", time.time() - t0, flush=True)
    if TINY:   # en TINY el "GPU" tambien corre en CPU: la medicion de presupuesto no ocurre
        pass
    report()
    (out / "session_seconds.txt").write_text(str(time.time() - t0))
