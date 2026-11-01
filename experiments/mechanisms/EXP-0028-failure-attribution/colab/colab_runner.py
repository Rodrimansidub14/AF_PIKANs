#!/usr/bin/env python
"""EXP-0028, despachador portatil para Google Colab (A100, pago por uso). NO tiene rutas de Kaggle ni de Colab: todo entra por argumentos.
Equivalente a kaggle_block*.py (mismo planificador de ranuras con tope de despacho y tope de kill) pero generico:

  - lee un manifiesto JSON de trabajos (problema, brazo, semilla, receta, bloque, prioridad); la receta fija el ejecutor (run_piezo.py, run_phase1d.py, run_phase1e.py, run_slice.py);
  - 1 GPU (o N: --gpus 0,1; o varios procesos por GPU: --jobs-per-gpu); en CPU solo para humo (--device cpu --slots N);
  - REANUDABLE: un trabajo esta hecho si sus archivos de resultado ya estan en --out/<grupo>/ (se salta); los fallidos se reintentan en la sesion siguiente hasta --max-attempts;
  - cada trabajo corre en una carpeta local de trabajo (--scratch) y, SOLO si termina bien, se copia a --out (p. ej. Google Drive montado); el JSON se copia al final (es la marca de "hecho");
  - W&B siempre OFFLINE (WANDB_MODE=offline) y la clave se ELIMINA del entorno de los procesos hijos; este archivo no lee ni escribe claves ni .env. Se sincroniza despues, en local:
    wandb_tools/sync_wandb.py <carpeta>;
  - topes: --cutoff-hours (no se despacha trabajo nuevo), --hard-cap-hours (se matan los trabajos en curso), --max-compute-units con --units-per-hour (tope de gasto), --max-priority
    (parar tras la prioridad 1 = 3 semillas), archivo STOP en --out (parada ordenada), y --fit-check (no despachar un trabajo que, segun projection.json, no cabe antes del kill);
  - ledger: --out/ledger.jsonl (una linea por evento: session_start, precheck, start, end, killed, skipped, session_end) y --out/progress.json.

Subcomandos:
  run      despacha el manifiesto             python colab_runner.py run --manifest M --bundle B --out O [--gpus 0] [--max-priority 1] [--cutoff-hours 11] [--hard-cap-hours 11.5]
  probe    prueba de tiempo corta en la GPU    python colab_runner.py probe --manifest M --bundle B --out O --steps 300 [--units-per-hour R]   (escribe timing_probe.json y projection.json)
  project  proyeccion de costo desde la prueba python colab_runner.py project --manifest M --out O [--units-per-hour R] [--slots N] [--throughput-gain G]
  status   hechos / pendientes por prioridad   python colab_runner.py status --manifest M --out O
"""
import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SELF = Path(__file__).resolve()
SCALAR = ("ac", "kpp", "burgers")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------- recetas
def _selftest_cmd(job, bundle, wd, dev, extra):
    return [sys.executable, str(SELF), "_selftest_job", "--out", str(wd), "--id", job["id"], "--seconds", str(job.get("seconds", 1)), "--fail", str(int(bool(job.get("fail", False))))]


def _script_cmd(script, argv):
    def f(job, bundle, wd, dev, extra):
        base = [sys.executable, str(Path(bundle) / script)] + argv(job) + ["--out", str(wd), "--device", dev]
        return base + list(extra)
    return f


def _p1e_argv(job):
    a = ["--problem", job["problem"], "--arms", job["arm"], "--seeds", str(job["seed"])]
    return a + (["--include-b"] if job.get("include_b") else [])


RECIPES = {
    # script, comando, archivos esperados, banderas por defecto (produccion), banderas de la prueba de tiempo, prueba previa B == ID (si existe)
    "piezo": dict(
        cmd=_script_cmd("run_piezo.py", lambda j: ["--problems", j["problem"], "--arms", j["arm"], "--seeds", str(j["seed"])]),
        expected=lambda j: [f"piezo-{j['problem']}-{j['arm']}-seed{j['seed']}.json"],
        extra=["--warmup-steps", "50"],
        probe=lambda n, ev: ["--adam", str(n), "--eval-every", "1000000000", "--pre-every", str(ev), "--no-artifacts", "--warmup-steps", "20"],
        precheck=lambda j: dict(key="piezo", script="run_piezo.py", argv=["--check-id", "--problems", "converse,direct", "--adam", "12", "--smoke-pool", "32", "--warmup-steps", "50"],
                                result="check_B_ID_piezo.json")),
    "phase1d": dict(
        cmd=_script_cmd("run_phase1d.py", lambda j: ["--problems", j["problem"], "--arms", j["arm"], "--seeds", str(j["seed"])]),
        expected=lambda j: [f"{j['problem']}-{j['arm']}-seed{j['seed']}.json"],
        extra=["--warmup-steps", "50"],
        probe=lambda n, ev: ["--adam", str(n), "--lbfgs-iters", "5", "--lbfgs-seconds", "5", "--pre-every", str(ev), "--warmup-steps", "20"],
        precheck=None),
    "phase1e": dict(
        cmd=_script_cmd("run_phase1e.py", _p1e_argv),
        expected=lambda j: [f"{j['problem']}-{j['arm']}-seed{j['seed']}.json"],
        extra=["--warmup-steps", "50"],
        probe=lambda n, ev: ["--adam", str(n), "--lbfgs-iters", "5", "--pre-every", str(ev), "--warmup-steps", "20"],
        probe_scalar=["--lbfgs-seconds", "5"],
        precheck=lambda j: (dict(key=f"p1e-{j['problem']}", script="run_phase1e.py", argv=["--problem", j["problem"], "--check-id", "--adam", "20", "--smoke-pool", "512", "--warmup-steps", "50"],
                                 result=f"check_B_ID_{j['problem']}.json") if j["problem"] in ("ns", "dfg") else None)),
    "slice": dict(
        cmd=_script_cmd("run_slice.py", lambda j: ["--problem", j["problem"], "--arms", j["arm"], "--seeds", str(j["seed"])]),
        expected=lambda j: [f"{j['problem']}-{j['arm']}-seed{j['seed']}.json", f"{j['problem']}-{j['arm']}-seed{j['seed']}-artifacts.npz"],
        extra=["--warmup-steps", "50"],
        probe=lambda n, ev: ["--adam", str(n), "--lbfgs-iters", "5", "--pre-every", str(ev), "--light-every", "1000000000", "--warmup-steps", "20"],
        precheck=lambda j: dict(key=f"slice-{j['problem']}", script="run_slice.py", argv=["--problem", j["problem"], "--check", "--adam", "20", "--smoke-pool", "512", "--warmup-steps", "50"],
                                result=f"check_slice_{j['problem']}.json")),
    "selftest": dict(cmd=_selftest_cmd, expected=lambda j: [f"selftest-{j['id']}.json"], extra=[], probe=None, precheck=None),
}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------- utilidades
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_manifest(path):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [j["id"] for j in m["jobs"]]
    if len(set(ids)) != len(ids):
        sys.exit("manifiesto con ids repetidos")
    for j in m["jobs"]:
        if j["recipe"] not in RECIPES:
            sys.exit(f"receta desconocida {j['recipe']!r} en {j['id']}")
        j.setdefault("group", j["recipe"])
        j.setdefault("priority", 1)
    return m


def ordered(jobs):
    """Orden de despacho = prioridad, luego el orden del archivo (el manifiesto ya viene ordenado por semilla completa)."""
    return [j for _, j in sorted(enumerate(jobs), key=lambda t: (t[1]["priority"], t[0]))]


def expected_files(job):
    return RECIPES[job["recipe"]]["expected"](job)


def is_done(job, out):
    d = Path(out) / job["group"]
    return all((d / f).exists() for f in expected_files(job))


def child_env(base_dir, dev_is_cpu, gpu):
    env = dict(os.environ, BUNDLE_DIR=str(base_dir), WANDB_MODE="offline", WANDB_SILENT="true", PYTHONUNBUFFERED="1")
    for k in list(env):                                   # el hijo nunca recibe una clave de W&B (ni de nada parecido)
        if k.upper() in ("WANDB_API_KEY", "WANDB_IDENTITY_TOKEN_FILE") or k.upper().endswith("_API_KEY"):
            env.pop(k)
    if dev_is_cpu:
        env["OMP_NUM_THREADS"] = "2"
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def append_ledger(out, rec):
    rec = dict(rec, ts=now())
    with open(Path(out) / "ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def read_ledger(out):
    p = Path(out) / "ledger.jsonl"
    if not p.exists():
        return []
    recs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    return recs


def attempts(out):
    n = {}
    for r in read_ledger(out):
        if r.get("event") in ("end", "killed") and r.get("status") in ("failed", "killed_hard_cap", "interrupted"):
            n[r["job_id"]] = n.get(r["job_id"], 0) + 1
    return n


def gpu_name():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip().splitlines() if r.returncode == 0 else []
    except Exception:  # noqa: BLE001
        return []


def kill_tree(proc):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        proc.kill()


def publish(workdir, dest, job):
    """Copia lo producido a --out/<grupo>/: primero todo lo demas, al final los archivos esperados (el JSON es la marca de hecho)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    exp = set(expected_files(job))
    wd = Path(workdir)
    for p in sorted(wd.rglob("*")):
        rel = p.relative_to(wd)
        if p.is_dir() or rel.name in exp and len(rel.parts) == 1 or rel.name == "job.log":
            continue
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest / rel)
    for name in sorted(exp, key=lambda n: n.endswith(".json")):          # npz antes que json
        shutil.copy2(wd / name, dest / name)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------- run
def run_prechecks(jobs, a, out, bundle, dev, env_for):
    """Prueba B == ID (u otra) por problema antes de despachar, como los kernels de Kaggle: si falla, no se despachan los trabajos de ese problema. Devuelve (jobs_ok, jobs_bloqueados)."""
    seen, failed = {}, set()
    for j in jobs:
        pc = RECIPES[j["recipe"]]["precheck"]
        pc = pc(j) if pc else None
        if not pc or pc["key"] in seen:
            if pc and not seen[pc["key"]]:
                failed.add(j["id"])
            continue
        chk = Path(out) / "checks" / pc["result"]
        ok = chk.exists() and json.loads(chk.read_text()).get("pass") is True
        if not ok and not a.skip_prechecks and not a.dry_run:
            (Path(out) / "checks").mkdir(parents=True, exist_ok=True)
            log = open(Path(out) / "checks" / f"{pc['key']}.log", "w")
            t0 = time.time()
            argv = list(pc["argv"])
            if dev == "cpu":                                    # humo en CPU: la misma prueba, diminuta y sin calentamiento (como X28_TINY en Kaggle)
                argv = [x for i, x in enumerate(argv) if x != "--warmup-steps" and (i == 0 or argv[i - 1] != "--warmup-steps")]
                argv += ["--adam", "6", "--smoke-pool", "32"] if "--smoke-pool" not in argv else []
                argv = [("6" if (i and argv[i - 1] == "--adam") else x) for i, x in enumerate(argv)]
            rc = subprocess.run([sys.executable, str(Path(bundle) / pc["script"])] + argv + ["--out", str(Path(out) / "checks"), "--device", dev], stdout=log, stderr=subprocess.STDOUT,
                                env=env_for(0), cwd=str(bundle)).returncode
            ok = chk.exists() and json.loads(chk.read_text()).get("pass") is True
            append_ledger(out, dict(event="precheck", key=pc["key"], rc=rc, passed=ok, seconds=round(time.time() - t0, 1)))
            print(f"prueba previa [{pc['key']}]: {'PASA' if ok else 'FALLA'} (rc {rc}, {time.time() - t0:.0f} s)", flush=True)
        elif a.skip_prechecks:
            ok = True
        seen[pc["key"]] = ok
        if not ok:
            failed.add(j["id"])
    return [j for j in jobs if j["id"] not in failed], [j for j in jobs if j["id"] in failed]


def cmd_run(a):
    man = load_manifest(a.manifest)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = Path(a.bundle).resolve() if a.bundle else None
    cpu = a.device == "cpu"
    dev = "cpu" if cpu else "cuda:0"
    gpus = [None] * a.slots if cpu else [g for g in a.gpus.split(",") if g != ""]
    slots = [(g, k) for g in gpus for k in range(1 if cpu else a.jobs_per_gpu)]
    scratch = Path(a.scratch) if a.scratch else Path(tempfile.gettempdir()) / "x28_colab_scratch"
    cutoff = a.cutoff_hours * 3600 if a.cutoff_hours else float("inf")
    hard = a.hard_cap_hours * 3600 if a.hard_cap_hours else float("inf")
    if a.max_compute_units:
        if not a.units_per_hour:
            sys.exit("--max-compute-units exige --units-per-hour (verifiquelo en su cuenta de Colab)")
        cutoff = min(cutoff, a.max_compute_units / a.units_per_hour * 3600)
        hard = min(hard, cutoff + 600)
    if hard < cutoff:
        sys.exit("--hard-cap-hours debe ser >= --cutoff-hours")
    est = {}
    if (out / "projection.json").exists():
        est = json.loads((out / "projection.json").read_text()).get("est_seconds", {})
    wanted = ordered(man["jobs"])
    if a.max_priority:
        wanted = [j for j in wanted if j["priority"] <= a.max_priority]
    for flag, key in ((a.only_group, "group"), (a.only_problem, "problem"), (a.only_arm, "arm")):
        if flag:
            wanted = [j for j in wanted if str(j.get(key)) in flag.split(",")]
    if a.only_seeds:
        wanted = [j for j in wanted if str(j.get("seed")) in a.only_seeds.split(",")]
    n_att = attempts(out)
    todo = [j for j in wanted if not is_done(j, out) and n_att.get(j["id"], 0) < a.max_attempts]
    n_done = sum(is_done(j, out) for j in wanted)
    info = dict(manifest=str(a.manifest), n_manifest=len(man["jobs"]), n_selected=len(wanted), n_done=n_done, n_todo=len(todo), n_exhausted=sum(1 for j in wanted if not is_done(j, out) and n_att.get(j["id"], 0) >= a.max_attempts),
                device=dev, gpus=gpus, jobs_per_gpu=a.jobs_per_gpu, slots=len(slots), cutoff_s=cutoff, hard_cap_s=hard, max_priority=a.max_priority, gpu_names=gpu_name(), python=sys.version.split()[0],
                platform=platform.platform(), wandb_mode="offline", scratch=str(scratch))
    print(json.dumps(info, default=str), flush=True)
    if a.dry_run:
        for j in todo:
            print("  [dry-run]", j["id"], "prio", j["priority"], "est_s", est.get(j["id"]))
        return 0
    append_ledger(out, dict(event="session_start", **info))
    if not todo:
        print("nada que hacer: todo lo seleccionado ya esta hecho")
        return 0
    if bundle is None or not bundle.exists():
        sys.exit("--bundle no existe")
    env_for = lambda g: child_env(bundle, cpu, g)  # noqa: E731
    todo, blocked = run_prechecks(todo, a, out, bundle, dev, env_for)
    for j in blocked:
        append_ledger(out, dict(event="skipped", job_id=j["id"], status="not_started_check_failed"))
    t_start = time.time()
    t0 = time.monotonic()
    queue, running, counts = list(todo), {}, dict(ok=0, failed=0, killed_hard_cap=0, not_started_cutoff=0, not_started_check_failed=len(blocked), not_started_stop=0, not_started_fit=0)

    def finish(slot, job, proc, wd, start, rc, status, elapsed):
        log = Path(wd) / "job.log"
        logs = out / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        pub_err = None
        if status == "ok":
            try:
                publish(wd, out / job["group"], job)
            except Exception as e:  # noqa: BLE001
                status, pub_err = "failed", f"{type(e).__name__}: {e}"
        if log.exists():
            shutil.copy2(log, logs / (f"{job['id']}.log" if status == "ok" else f"{job['id']}.{status}-{n_att.get(job['id'], 0) + 1}.log"))
        rec = dict(event="end" if status != "killed_hard_cap" else "killed", job_id=job["id"], group=job["group"], priority=job["priority"], slot=slot, start_s=round(start, 1), end_s=round(elapsed, 1),
                   wall_s=round(elapsed - start, 1), returncode=rc, status=status, error=pub_err)
        append_ledger(out, rec)
        counts[status] = counts.get(status, 0) + 1
        print(f"[{elapsed:8.0f}s] {status:16s} {job['id']} slot={slot} rc={rc} wall={elapsed - start:.0f}s", flush=True)
        if status in ("ok",) or not a.keep_scratch:
            shutil.rmtree(wd, ignore_errors=True)

    def progress(extra=""):
        done_now = sum(is_done(j, out) for j in wanted)
        (out / "progress.json").write_text(json.dumps(dict(updated=now(), wall_seconds=round(time.time() - t_start), done=done_now, selected=len(wanted), running=[r[0]["id"] for r in running.values()],
                                                              queue_left=len(queue), counts=counts, note=extra), indent=1))

    try:
        while queue or running:
            elapsed = time.monotonic() - t0
            for slot, (job, proc, wd, start) in list(running.items()):
                rc = proc.poll()
                if rc is not None:
                    finish(slot, job, proc, wd, start, rc, "ok" if rc == 0 and all((Path(wd) / f).exists() for f in expected_files(job)) else "failed", elapsed)
                    del running[slot]
            if elapsed >= hard:
                for slot, (job, proc, wd, start) in list(running.items()):
                    kill_tree(proc)
                    proc.wait()
                    finish(slot, job, proc, wd, start, None, "killed_hard_cap", elapsed)
                    del running[slot]
            stop = (out / "STOP").exists()
            if queue and (elapsed >= cutoff or stop or elapsed >= hard):
                why = "not_started_stop" if stop and elapsed < cutoff else "not_started_cutoff"
                for job in queue:
                    append_ledger(out, dict(event="skipped", job_id=job["id"], status=why))
                    counts[why] += 1
                print(f"[{elapsed:8.0f}s] sin despacho nuevo ({why}); quedan sin lanzar {len(queue)} trabajos; en curso {len(running)}", flush=True)
                queue = []
            for slot, sid in enumerate(slots):
                if queue and slot not in running:
                    pick = None
                    for i, job in enumerate(queue):
                        e = est.get(job["id"])
                        if a.fit_check and e and elapsed + 1.15 * e > hard:
                            continue                                           # no cabe antes del kill: se prueba con uno mas corto
                        pick = i
                        break
                    if pick is None:
                        for job in queue:
                            append_ledger(out, dict(event="skipped", job_id=job["id"], status="not_started_fit"))
                            counts["not_started_fit"] += 1
                        print(f"[{elapsed:8.0f}s] ningun trabajo restante cabe antes del kill; fin del despacho", flush=True)
                        queue = []
                        break
                    job = queue.pop(pick)
                    wd = scratch / job["id"]
                    shutil.rmtree(wd, ignore_errors=True)
                    wd.mkdir(parents=True, exist_ok=True)
                    extra = job["extra_args"] if job.get("extra_args") is not None else RECIPES[job["recipe"]]["extra"]
                    cmd = RECIPES[job["recipe"]]["cmd"](job, bundle, wd, dev, extra)
                    kw = dict(start_new_session=True) if os.name != "nt" else {}
                    proc = subprocess.Popen(cmd, stdout=open(wd / "job.log", "w"), stderr=subprocess.STDOUT, env=env_for(sid[0]), cwd=str(bundle), **kw)
                    running[slot] = (job, proc, wd, elapsed)
                    append_ledger(out, dict(event="start", job_id=job["id"], group=job["group"], priority=job["priority"], slot=slot, gpu=sid[0], start_s=round(elapsed, 1), cmd=[os.path.basename(c) if i < 2 else c for i, c in enumerate(cmd)],
                                            est_s=est.get(job["id"]), attempt=n_att.get(job["id"], 0) + 1))
                    print(f"[{elapsed:8.0f}s] start            {job['id']} slot={slot} gpu={sid[0]} prio={job['priority']} est={est.get(job['id'])}", flush=True)
            progress()
            if queue or running:
                time.sleep(a.poll)
    except KeyboardInterrupt:
        elapsed = time.monotonic() - t0
        for slot, (job, proc, wd, start) in list(running.items()):
            kill_tree(proc)
            proc.wait()
            finish(slot, job, proc, wd, start, None, "interrupted", elapsed)
        append_ledger(out, dict(event="session_end", interrupted=True, counts=counts))
        print("interrumpido: trabajos en curso descartados (se repiten al reanudar)")
        return 130
    wall = time.time() - t_start
    done_now = sum(is_done(j, out) for j in wanted)
    summary = dict(event="session_end", wall_seconds=round(wall), counts=counts, done=done_now, selected=len(wanted), complete=done_now == len(wanted))
    append_ledger(out, summary)
    progress("fin")
    print("fin de la sesion:", json.dumps(summary), flush=True)
    return 0


# ----------------------------------------------------------------------------------------------------------------------------------------------------------- probe / proyeccion
def _units(hours, rate):
    return None if not rate else hours * rate


def cmd_probe(a):
    man = load_manifest(a.manifest)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = Path(a.bundle).resolve()
    cpu = a.device == "cpu"
    dev = "cpu" if cpu else "cuda:0"
    pdir = Path(a.scratch) if a.scratch else Path(tempfile.gettempdir()) / "x28_colab_probe"
    shutil.rmtree(pdir, ignore_errors=True)
    targets = man.get("probe", {}).get("targets", [])
    if not targets:
        sys.exit("el manifiesto no declara probe.targets")
    n = a.steps
    every = max(1, n // 3)
    rows = []

    def run_target(t, tag, out_sub):
        rc_ = RECIPES[t["recipe"]]
        job = dict(id=f"probe-{tag}", recipe=t["recipe"], problem=t["problem"], arm=t["arm"], seed=t.get("seed", 4001), group="probe")
        wd = pdir / out_sub
        wd.mkdir(parents=True, exist_ok=True)
        fl = rc_["probe"](n, every) + (rc_.get("probe_scalar", []) if t["problem"] in SCALAR else []) + list(t.get("probe_extra", []))
        cmd = rc_["cmd"](job, bundle, wd, dev, fl)
        # los ejecutores de bloque aceptan semillas fuera de la receta solo en modo humo: --adam lo activa
        return job, subprocess.Popen(cmd, stdout=open(wd / "job.log", "w"), stderr=subprocess.STDOUT, env=child_env(bundle, cpu, a.gpu), cwd=str(bundle)), wd, time.time()

    def collect(t, job, proc, wd, t_start):
        rc = proc.wait()
        wall = time.time() - t_start
        row = dict(key=f"{t['problem']}:{t['arm']}", recipe=t["recipe"], problem=t["problem"], arm=t["arm"], rc=rc, wall_s=round(wall, 2), steps=n)
        files = list(wd.glob(f"*{t['problem']}*{t['arm']}*seed*.json")) if t["recipe"] != "slice" else list(wd.glob("*seed*.json"))
        files = [f for f in files if not f.name.endswith("note.json")]
        if rc == 0 and files:
            r = json.loads(files[0].read_text())
            pre_s, n_pre = float(r.get("pre_seconds", 0.0)), int(r.get("pre_recomputes", 0))
            adam_s = float(r.get("adam_seconds", 0.0))
            row.update(adam_seconds=adam_s, pre_seconds=pre_s, n_recomputes=n_pre, step_s=(adam_s - pre_s) / max(1, r.get("adam_steps", n)), pre_s_each=(pre_s / n_pre if n_pre else None),
                       score=r.get("primary_final", r.get("primary_adam")))
        else:
            tail = (wd / "job.log").read_text(errors="replace").strip().splitlines()[-3:] if (wd / "job.log").exists() else []
            row["error"] = " | ".join(tail)[:300]
        return row

    for t in targets:
        job, proc, wd, ts = run_target(t, f"{t['problem']}-{t['arm']}", f"{t['problem']}-{t['arm']}")
        row = collect(t, job, proc, wd, ts)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (out / "timing_probe.json").write_text(json.dumps(dict(partial=True, rows=rows), indent=1))
    conc = None
    if a.concurrency > 1 and rows and rows[0].get("step_s"):
        t = targets[0]
        k = a.concurrency
        procs = [run_target(t, f"c{i}", f"conc{i}") for i in range(k)]
        crow = [collect(t, *p) for p in procs]
        steps_c = [r["step_s"] for r in crow if r.get("step_s")]
        if len(steps_c) == k:
            conc = dict(target=rows[0]["key"], k=k, step_s_solo=rows[0]["step_s"], step_s_concurrent=statistics.mean(steps_c), throughput_gain=k * rows[0]["step_s"] / statistics.mean(steps_c))
            print("concurrencia:", json.dumps(conc), flush=True)
    res = dict(partial=False, written=now(), device=(gpu_name() or [dev]), steps=n, pre_every=every, rows=rows, concurrency=conc, dtype_note="float64 en piezo; float32 en los bloques (receta de cada ejecutor)")
    (out / "timing_probe.json").write_text(json.dumps(res, indent=1))
    shutil.rmtree(pdir, ignore_errors=True)
    a.slots = a.slots or 1
    a.throughput_gain = a.throughput_gain or (conc["throughput_gain"] if conc else 1.0)
    return cmd_project(a)


def project_jobs(man, rows, jobs):
    """Segundos estimados por trabajo desde las mediciones. Modo 'direct' (piezo: pasos x s/paso + recalculos x s/recalculo + fijo) o 'ratio' (bloques: est_s_t4 x medido/T4 del s/paso)."""
    by = {r["key"]: r for r in rows if r.get("step_s")}
    notes = []
    ref = man.get("probe", {}).get("t4_ref", {})
    ratios = {k: by[k]["step_s"] / ref[k]["step_s"] for k in by if k in ref and ref[k].get("step_s")}
    allr = list(ratios.values())
    est = {}
    for j in jobs:
        if "steps" in j:                                                                  # modo directo
            chain = j.get("probe_as", [j["arm"]])
            chain = [chain] if isinstance(chain, str) else chain
            pick = next((c for c in chain if f"{j['problem']}:{c}" in by), None)
            if pick is None:
                continue
            r = by[f"{j['problem']}:{pick}"]
            if pick != chain[0]:
                notes.append(f"{j['id']}: se uso {pick} (fallo la medicion de {chain[0]})")
            pre_each = r.get("pre_s_each") or 0.0
            est[j["id"]] = j["steps"] * r["step_s"] * j.get("step_factor", 1.0) + j.get("n_recomputes", 0) * pre_each + j.get("fixed_s", 0.0)
        elif "est_s_t4" in j:                                                             # modo razon
            k = j.get("ref_key", f"{j['problem']}:{j['arm']}")
            fam = [v for kk, v in ratios.items() if kk.split(":")[0] == j["problem"]]
            ratio = ratios.get(k) or (statistics.median(fam) if fam else None) or (statistics.median(allr) if allr else None)
            if ratio is None:
                continue
            if k not in ratios:
                notes.append(f"{j['id']}: razon sin medicion propia ({'mediana del problema' if fam else 'mediana global'})")
            est[j["id"]] = j["est_s_t4"] * ratio
    return est, sorted(set(notes))


def cmd_project(a):
    man = load_manifest(a.manifest)
    out = Path(a.out)
    tp = out / "timing_probe.json"
    if not tp.exists():
        sys.exit("falta timing_probe.json: corra primero `probe`")
    rows = json.loads(tp.read_text())["rows"]
    est, notes = project_jobs(man, rows, man["jobs"])
    gain = a.throughput_gain or 1.0
    slots = a.slots or 1
    missing = [j["id"] for j in man["jobs"] if j["id"] not in est]
    pend = [j for j in ordered(man["jobs"]) if not is_done(j, out) and j["id"] in est]
    (out / "projection.json").write_text(json.dumps(dict(written=now(), est_seconds={k: round(v, 1) for k, v in est.items()}, notes=notes, missing=missing), indent=1))
    print("\nProyeccion (solo trabajos pendientes; segundos de GPU por trabajo desde la prueba, sin contar ocio ni reintentos):")
    head = f"{'prioridad':>9} {'trabajos':>8} {'GPU-h':>8} {'pared-h':>8} {'acum. h':>8}" + (f" {'unidades':>9} {'acum. u':>8}" if a.units_per_hour else "")
    print(head)
    cum_h = cum_u = 0.0
    summ = []
    for pr in sorted({j["priority"] for j in pend}):
        js = [j for j in pend if j["priority"] == pr]
        gpu_h = sum(est[j["id"]] for j in js) / 3600
        wall_h = gpu_h / (slots * gain) if slots * gain else gpu_h
        cum_h += wall_h
        u = _units(wall_h, a.units_per_hour)
        cum_u += u or 0
        summ.append(dict(priority=pr, jobs=len(js), gpu_hours=gpu_h, wall_hours=wall_h, cum_wall_hours=cum_h, units=u, cum_units=cum_u if a.units_per_hour else None))
        print(f"{pr:>9} {len(js):>8} {gpu_h:>8.2f} {wall_h:>8.2f} {cum_h:>8.2f}" + (f" {u:>9.1f} {cum_u:>8.1f}" if a.units_per_hour else ""))
    print(f"(pared = GPU-h / ({slots} ranuras x ganancia de concurrencia {gain:.2f}); sin --units-per-hour no se calculan unidades: declare la tasa que muestra SU cuenta para la A100)")
    for n_ in notes[:10]:
        print("  nota:", n_)
    if missing:
        print("  sin estimacion (no hay medicion utilizable):", ", ".join(missing[:8]), "..." if len(missing) > 8 else "")
    (out / "projection_summary.json").write_text(json.dumps(dict(written=now(), slots=slots, throughput_gain=gain, units_per_hour=a.units_per_hour, by_priority=summ, notes=notes, missing=missing), indent=1))
    return 0


def cmd_status(a):
    man = load_manifest(a.manifest)
    out = Path(a.out)
    n_att = attempts(out) if (out / "ledger.jsonl").exists() else {}
    tab = {}
    for j in man["jobs"]:
        d = tab.setdefault(j["priority"], dict(total=0, done=0, failed=0))
        d["total"] += 1
        if is_done(j, out):
            d["done"] += 1
        elif n_att.get(j["id"]):
            d["failed"] += 1
    print(f"{'prioridad':>9} {'total':>6} {'hechos':>7} {'con fallos':>11}")
    for pr in sorted(tab):
        print(f"{pr:>9} {tab[pr]['total']:>6} {tab[pr]['done']:>7} {tab[pr]['failed']:>11}")
    return 0


def _selftest_job(a):
    time.sleep(a.seconds)
    if a.fail:
        sys.exit(3)
    Path(a.out, f"selftest-{a.id}.json").write_text(json.dumps(dict(id=a.id, ok=True)))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--manifest", required=True)
        p.add_argument("--out", required=True, help="carpeta de salida persistente (p. ej. en Google Drive montado)")

    def hw(p):
        p.add_argument("--bundle", help="carpeta plana con run_piezo.py / run_phase1d.py ... (BUNDLE_DIR)")
        p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
        p.add_argument("--scratch", default="", help="carpeta LOCAL de trabajo (defecto: temporal del sistema); lo terminado se copia a --out")

    def proj(p):
        p.add_argument("--units-per-hour", type=float, default=None, help="unidades de computo de Colab por hora de la A100: VERIFICAR en su cuenta; sin valor no se calculan unidades")
        p.add_argument("--slots", type=int, default=0, help="procesos simultaneos (GPUs x jobs-per-gpu) para convertir GPU-h en horas de pared")
        p.add_argument("--throughput-gain", type=float, default=0.0, help="ganancia medida de correr varios procesos por GPU (la prueba la mide con --concurrency)")

    r = sub.add_parser("run")
    common(r)
    hw(r)
    r.add_argument("--gpus", default="0", help="ids de GPU separados por coma")
    r.add_argument("--jobs-per-gpu", type=int, default=1)
    r.add_argument("--slots", type=int, default=1, help="solo --device cpu")
    r.add_argument("--cutoff-hours", type=float, default=0.0, help="sin trabajo nuevo despues de estas horas (0 = sin tope)")
    r.add_argument("--hard-cap-hours", type=float, default=0.0, help="matar los trabajos en curso a estas horas (0 = sin tope)")
    r.add_argument("--max-compute-units", type=float, default=0.0)
    r.add_argument("--units-per-hour", type=float, default=0.0)
    r.add_argument("--max-priority", type=int, default=0, help="solo trabajos con prioridad <= N (1 = 3 semillas)")
    r.add_argument("--only-group", default="")
    r.add_argument("--only-problem", default="")
    r.add_argument("--only-arm", default="")
    r.add_argument("--only-seeds", default="")
    r.add_argument("--max-attempts", type=int, default=2)
    r.add_argument("--fit-check", action="store_true", help="con projection.json: no despachar un trabajo que no cabe antes del kill")
    r.add_argument("--skip-prechecks", action="store_true")
    r.add_argument("--keep-scratch", action="store_true", help="conservar la carpeta de trabajo de los fallidos")
    r.add_argument("--poll", type=float, default=20.0)
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("probe")
    common(p)
    hw(p)
    proj(p)
    p.add_argument("--gpu", default="0")
    p.add_argument("--steps", type=int, default=300, help="pasos de Adam de cada prueba (200-500)")
    p.add_argument("--concurrency", type=int, default=1, help=">1: ademas mide k procesos simultaneos en la misma GPU")
    p.set_defaults(fn=cmd_probe)
    q = sub.add_parser("project")
    common(q)
    proj(q)
    q.set_defaults(fn=cmd_project)
    s = sub.add_parser("status")
    common(s)
    s.set_defaults(fn=cmd_status)
    t = sub.add_parser("_selftest_job")
    t.add_argument("--out")
    t.add_argument("--id")
    t.add_argument("--seconds", type=float, default=1)
    t.add_argument("--fail", type=int, default=0)
    t.set_defaults(fn=_selftest_job)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
