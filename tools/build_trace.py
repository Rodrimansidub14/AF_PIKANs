"""Genera experiments/TRAZA.md: experimento -> estado -> decisión -> bloques de corridas -> notebooks, desde las fuentes.
No se edita a mano. Uso:  python tools/build_trace.py"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "experiments"
SALIDA = EXPS / "TRAZA.md"


def front(path):
    if not path.exists():
        return {}
    m = re.match(r"---\r?\n(.*?)\r?\n---", path.read_text(encoding="utf-8", errors="replace")[:4000], re.S)
    d = {}
    for linea in (m.group(1).splitlines() if m else []):
        if ":" in linea and not linea.startswith((" ", "-")):
            k, v = linea.split(":", 1)
            d[k.strip()] = v.split("#")[0].strip().strip('"')
    return d


def titulo(path):
    if path.exists():
        for l in path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
            if l.startswith("# "):
                return l[2:].strip()
    return ""


def runs():
    out = []
    f = ROOT / "registry" / "runs.jsonl"
    for l in f.read_text(encoding="utf-8").splitlines():
        if l.strip():
            out.append(json.loads(l))
    return out


def main():
    R = runs()
    carpetas = {}
    for d in sorted(EXPS.glob("*/EXP-*")):
        if d.is_dir():
            eid = "-".join(d.name.split("-")[:2])
            carpetas.setdefault(eid, []).append(d)
    ids = sorted(set(carpetas) | {"-".join(str(r["experiment"]).split("-")[:2]) for r in R if str(r["experiment"]).startswith("EXP-")} | {"EXP-0029"})
    nbs = [p for p in list((ROOT / "notebooks").rglob("*.ipynb")) + list(EXPS.rglob("*.ipynb")) if "wandb" not in p.parts]
    L = ["# Traza de experimentos", "",
         "Generado por `tools/build_trace.py` desde `experiments/*/EXP-*/{spec,decision}.md`, `registry/runs.jsonl` y los notebooks. **No editar a mano.**", "",
         "| EXP | Problema | Pregunta | Estado | Decisión | Bloques en el registro | Notebooks |", "|---|---|---|---|---|---|---|"]
    filas = []
    for eid in ids:
        ds = carpetas.get(eid, [])
        main_d = next((d for d in ds if (d / "spec.md").exists()), ds[0] if ds else None)
        if eid == "EXP-0029":
            main_d = carpetas["EXP-0028"][0]
        fm = front(main_d / "spec.md") if main_d else {}
        dec = front(main_d / ("decision-EXP-0029.md" if eid == "EXP-0029" else "decision.md")) if main_d else {}
        borr = list(main_d.glob("decision-borrador*.md")) if main_d else []
        if dec.get("decision"):
            decision = dec["decision"]
        elif borr:
            decision = "borrador sin firmar"
        else:
            decision = "—"
        estado = fm.get("status", "—")
        if eid == "EXP-0029":
            estado = "scored (spec dentro de EXP-0028)"
        pregunta = titulo(main_d / "spec.md").split("—", 1)[-1].strip()[:90] if main_d else "spec dentro de EXP-0028"
        if eid == "EXP-0029":
            pregunta = "Mecanismo final M en piezo (spec y firma en EXP-0029_spec.md)"
        prob = main_d.parent.name if main_d else "mechanisms"
        bloques = [r for r in R if str(r["experiment"]).startswith(eid) or (eid == "EXP-0029" and str(r["experiment"]) == "EXP-0029")]
        nb = [p for p in nbs if eid.lower().replace("-", "") in p.as_posix().lower().replace("-", "") or eid in p.as_posix()]
        L.append(f"| {eid} | {prob} | {pregunta} | {estado} | {decision} | {len(bloques)} | {len(nb)} |")
        filas.append(dict(id=eid, problema=prob, pregunta=pregunta, estado=estado, decision=decision, bloques=[b["run_id"] for b in bloques], notebooks=len(nb)))
    L += ["", "## Bloques de corridas (de `registry/runs.jsonl`)", "", "| run_id | EXP | Fecha | Semillas | Backend | Estado | Resultados |", "|---|---|---|---|---|---|---|"]
    for r in sorted(R, key=lambda r: (str(r.get("date", "")), r["run_id"])):
        L.append(f"| {r['run_id']} | {str(r['experiment'])[:8]} | {r.get('date', '')} | {r.get('seeds', '')} | {r.get('backend', '')} | {r.get('status', '')} | `{r.get('results', '')}` |")
    L += ["", "## Notebooks", "", "| Notebook | Rol (metadata.research) |", "|---|---|"]
    for p in sorted(nbs):
        try:
            rol = (json.loads(p.read_text(encoding="utf-8")).get("metadata", {}).get("research") or {}).get("role", "—")
        except Exception:
            rol = "ilegible"
        L.append(f"| `{p.relative_to(ROOT).as_posix()}` | {rol} |")
    SALIDA.write_text("\n".join(L) + "\n", encoding="utf-8")
    (EXPS / "status.json").write_text(json.dumps(filas, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"escrito {SALIDA.relative_to(ROOT)}: {len(ids)} experimentos, {len(R)} bloques, {len(nbs)} notebooks")


if __name__ == "__main__":
    main()
