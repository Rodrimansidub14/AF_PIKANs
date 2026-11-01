"""Comprueba que el repo cumple specs/conventions/convencion-nombres.md antes de subir a git.
Duro (sale con 1): nombres de notebooks, registro JSONL válido y sin run_id repetidos, archivos > 5 MB que git agregaría,
secretos, .env rastreado, borrador de decisión llamado decision.md sin firma.  Aviso: el resto.
Uso:  python tools/check_repo.py [--fast]   (--fast salta el escaneo de archivos que git agregaría)"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbstrip  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX = 5_000_000
SECRETOS = re.compile(r"(\bak-[A-Za-z0-9]{16,}|\bas-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]+|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|WANDB_API_KEY\s*=\s*[0-9a-f]{20,})")
TEXTO = {".py", ".md", ".json", ".jsonl", ".toml", ".yml", ".yaml", ".txt", ".mjs", ".ts", ".ipynb", ".cfg", ".ini"}
duros, avisos = [], []


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.splitlines()


def notebooks():
    for p in list((ROOT / "notebooks").rglob("*.ipynb")) + list((ROOT / "experiments").rglob("*.ipynb")):
        if "wandb" in p.parts or ".ipynb_checkpoints" in p.parts:
            continue
        n = p.name
        if re.search(r"[A-Z\s()]|^copy_of", n) or "é" in n:
            duros.append(f"notebook con mayúsculas/espacios/paréntesis: {p.relative_to(ROOT)}")
        elif "_" in n:
            avisos.append(f"notebook con guion bajo (preferir guion): {p.relative_to(ROOT)}")


def registro():
    vistos = set()
    for f in (ROOT / "registry").glob("*.jsonl"):
        for i, l in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not l.strip():
                continue
            try:
                j = json.loads(l)
            except Exception as e:
                duros.append(f"{f.name}:{i} JSON inválido ({e})")
                continue
            if f.name == "runs.jsonl":
                if j.get("run_id") in vistos:
                    duros.append(f"runs.jsonl: run_id repetido {j.get('run_id')}")
                vistos.add(j.get("run_id"))


def experimentos():
    ids = {}
    for d in sorted((ROOT / "experiments").glob("*/EXP-*")):
        if not d.is_dir():
            continue
        if not re.fullmatch(r"EXP-\d{4}-[a-z0-9]+(-[a-z0-9]+)*", d.name):
            avisos.append(f"nombre de experimento fuera de convención: {d.relative_to(ROOT)}")
        ids.setdefault(d.name[:8], set()).add(d.name)
        if not (d / "spec.md").exists() and (d / "results").exists():
            avisos.append(f"sin spec.md (carpeta de resultados): {d.relative_to(ROOT)}")
        dec = d / "decision.md"
        if dec.exists() and "BORRADOR" in dec.read_text(encoding="utf-8", errors="replace")[:600]:
            duros.append(f"decision.md es un borrador sin firmar: {dec.relative_to(ROOT)}")
    for k, v in ids.items():
        if len({n for n in v}) > 1:
            avisos.append(f"{k} aparece con varios slugs (puede ser espejo de resultados): {sorted(v)}")


def archivos_a_subir():
    lista = git("ls-files") + git("ls-files", "--others", "--exclude-standard")
    if (ROOT / ".env").exists() and ".env" in git("ls-files"):
        duros.append(".env está rastreado por git")
    for rel in lista:
        p = ROOT / rel.strip('"')
        if not p.is_file():
            continue
        s = p.stat().st_size
        if p.suffix == ".ipynb" and s > MAX:  # git lo guarda sin salidas (filtro nbstrip): medir el tamaño ya limpio
            try:
                s = len(json.dumps(nbstrip.limpiar(json.loads(p.read_text(encoding="utf-8"))), indent=1, ensure_ascii=False).encode())
            except Exception:
                pass
        if s > MAX:
            duros.append(f"archivo de {s / 1e6:.0f} MB que git agregaría: {rel}")
        if p.suffix.lower() in TEXTO and s < 3_000_000:
            try:
                m = SECRETOS.search(p.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                m = None
            if m:
                duros.append(f"posible secreto en {rel}")
    return len(lista)


if __name__ == "__main__":
    n = None
    notebooks(); registro(); experimentos()
    if "--fast" not in sys.argv:
        n = archivos_a_subir()
    for a in avisos:
        print("AVISO", a)
    for d in duros:
        print("ERROR", d)
    print(f"{len(duros)} errores, {len(avisos)} avisos" + (f"; {n} archivos que git agregaría revisados" if n is not None else ""))
    sys.exit(1 if duros else 0)
