"""EXP-0028 Fase 1e (Enmienda v1.4): J e I (y B, F, E donde no hay 1d) con RECETA COMPLETA, solo PIKAN (RBF-KAN). NO modifica ningun ejecutor ya subido: este archivo solo FIJA la receta de cada
problema y delega, en el mismo proceso (runpy), en el ejecutor existente:

  ns      -> run_phase1c.py      --recipe full : Adam 20 000 (float32, lote completo) + L-BFGS 500 it. (la de la linea base del worker NS2d-CG). Brazos B, F, J, E, I.
  dfg     -> run_phase1c_dfg.py  --adam 20000 --lbfgs-iters 4000 : receta COMPLETA de DFG definida en budget_1e.md (Adam x10 de la fase 1c = los 20 000 pasos de dfg_long, mismo lazo/minibatch de
                                 train_dfg; L-BFGS de sweep.lbfgs_stage por ITERACIONES, tope 4 000 it. = la profundidad que alcanzo RBF16 en dfg_long en su tope de 600 s; sin tope de tiempo,
                                 igual para todos los brazos). Brazos B, F, J, E, I.
  ac, kpp, burgers -> run_phase1d.py --problems P : la receta de 1d tal cual (RECIPES de run_phase1d.py), solo brazos J e I (B es el de 1d, pareado por semilla; --include-b lo anade
                                 como contingencia si el B de 1d no existe o no es reproducible).

Semillas: 4101-4105 en todos los problemas (las de 1d). Fuera de humo se rechaza cualquier otra semilla. Se considera HUMO cualquier sustitucion de la receta (--adam, --lbfgs-iters,
--lbfgs-seconds, --n-col, --smoke-pool, --pre-every, --pre-n): el JSON queda marcado por el ejecutor delegado (smoke / smoke_pool) y la guardia de semillas se levanta.
Un JSON por corrida (`{problema}-{brazo}-seed{semilla}.json`, nombres de cada ejecutor), reanudable. Se escribe ademas `recipe_1e-{problema}.note.json` con la receta fijada.

Uso : python run_phase1e.py --problem ns|dfg|ac|kpp|burgers --arms J,I --seeds 4101,4102 --out DIR [--device cuda:0 --warmup-steps 50 --threads 2]
      python run_phase1e.py --problem ns --check-id --out DIR      (solo ns/dfg: prueba B == ID, ejecutor delegado)
Humo: python run_phase1e.py --problem dfg --arms B,J --seeds 4101 --out DIR --adam 12 --lbfgs-iters 5 --pre-every 5 --pre-n 64 --smoke-pool 128
"""
import argparse
import json
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS_1E = tuple(range(4101, 4106))
RECIPES_1E = {
    "ns": dict(script="run_phase1c.py", args=["--recipe", "full", "--lbfgs-iters", "500"], arms=("B", "F", "J", "E", "I"), adam=20000, lbfgs_iters=500,
               source="run_phase1c.py --recipe full: Adam 20 000 float32 lote completo + L-BFGS 500 it. (strong-Wolfe, chunk 50); GN 2048 por clave; 21 recalculos J/E/I"),
    "dfg": dict(script="run_phase1c_dfg.py", args=["--adam", "20000", "--lbfgs-iters", "4000"], arms=("B", "F", "J", "E", "I"), adam=20000, lbfgs_iters=4000,
                source="run_phase1c_dfg.py con --adam 20000 --lbfgs-iters 4000 (Adam x10 de 1c; L-BFGS 4 000 it. sobre el pool completo); GN 2048 por clave (4 608 pts); 21 recalculos J/E/I"),
    "ac": dict(script="run_phase1d.py", args=[], arms=("J", "I"), source="run_phase1d.py RECIPES['ac'] (Adam 40 000 f32 + L-BFGS f32 tope 300 s / 15 000 it.)"),
    "kpp": dict(script="run_phase1d.py", args=[], arms=("J", "I"), source="run_phase1d.py RECIPES['kpp'] (heredada de AC)"),
    "burgers": dict(script="run_phase1d.py", args=[], arms=("J", "I"), source="run_phase1d.py RECIPES['burgers'] (16 384 pts, Adam 24 000 f32 + L-BFGS f64 tope 1 800 s)"),
}
SMOKE_FLAGS = ("--adam", "--lbfgs-iters", "--lbfgs-seconds", "--n-col", "--smoke-pool", "--pre-every", "--pre-n")


def build_argv(problem, arms, seeds, out, device, threads, warmup_steps, smoke_args, check_id=False, timing=False, include_b=False):
    """Linea de comandos del ejecutor delegado. `smoke_args` = lista ya plana de (--flag valor) que sustituye la receta."""
    rec = RECIPES_1E[problem]
    given = {smoke_args[i] for i in range(0, len(smoke_args), 2)}
    argv = []
    if problem in ("ac", "kpp", "burgers"):
        argv += ["--problems", problem]
    base = list(rec["args"])
    drop = {i for i in range(0, len(base), 2) if base[i] in given or (base[i] == "--recipe" and "--adam" in given)}
    for i in range(0, len(base), 2):
        if i not in drop:
            argv += base[i:i + 2]
    argv += smoke_args
    if not (check_id or timing):
        argv += ["--arms", ",".join(arms), "--seeds", ",".join(str(s) for s in seeds)]
    elif check_id:
        argv += ["--check-id"]
    elif timing:
        argv += ["--timing"]
    argv += ["--out", str(out), "--device", device, "--threads", str(threads)]
    if warmup_steps:
        argv += ["--warmup-steps", str(warmup_steps)]
    return argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=tuple(RECIPES_1E))
    ap.add_argument("--arms", default="")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_1E))
    ap.add_argument("--include-b", action="store_true", help="anade B a ac/kpp/burgers (contingencia: B de 1d ausente o no reproducible)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--check-id", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adam", default=None)
    ap.add_argument("--lbfgs-iters", default=None)
    ap.add_argument("--lbfgs-seconds", default=None)
    ap.add_argument("--n-col", default=None)
    ap.add_argument("--smoke-pool", default=None)
    ap.add_argument("--pre-every", default=None)
    ap.add_argument("--pre-n", default=None)
    a = ap.parse_args()
    rec = RECIPES_1E[a.problem]
    smoke_args = []
    for flag in SMOKE_FLAGS:
        v = getattr(a, flag[2:].replace("-", "_"))
        if v is not None:
            smoke_args += [flag, str(v)]
    smoke = bool(smoke_args)
    if (a.check_id or a.timing) and a.problem not in ("ns", "dfg"):
        sys.exit("--check-id/--timing solo para ns y dfg (los de ac/kpp/burgers son de 1d)")
    seeds = [int(s) for s in a.seeds.split(",")]
    if not smoke and not (a.check_id or a.timing) and not set(seeds) <= set(SEEDS_1E):
        sys.exit(f"semillas {sorted(set(seeds) - set(SEEDS_1E))} fuera de 4101-4105 (las de 1d): use esas o un flag de humo (--adam ...)")
    arms = tuple(x for x in a.arms.split(",") if x) or rec["arms"]
    allowed = tuple(rec["arms"]) + (("B",) if a.include_b else ())
    bad = [x for x in arms if x not in allowed]
    if bad:
        sys.exit(f"brazos {bad} no permitidos en 1e/{a.problem} (permitidos: {allowed}; B de ac/kpp/burgers es el de 1d, --include-b lo anade)")
    if a.problem in ("ac", "kpp", "burgers") and a.include_b:
        arms = tuple(dict.fromkeys(("B",) + arms))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    script = HERE / rec["script"]
    argv = build_argv(a.problem, arms, seeds, out, a.device, a.threads, a.warmup_steps, smoke_args, a.check_id, a.timing, a.include_b)
    (out / f"recipe_1e-{a.problem}.note.json").write_text(json.dumps(dict(problem=a.problem, delegate=rec["script"], argv=argv, source=rec["source"], arms=arms, seeds=seeds, smoke=smoke,
                                                                         written_at=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1))
    sys.argv = [str(script)] + argv
    sys.path.insert(0, str(HERE))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
