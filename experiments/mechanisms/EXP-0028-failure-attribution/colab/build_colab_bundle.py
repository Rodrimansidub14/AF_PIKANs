"""Arma el zip de codigo+datos para Colab (EXP-0028). LOCAL; no sube ni ejecuta nada y no toca lo ya subido a Kaggle (solo LEE kaggle/dataset_1e y las fuentes de piezo / artifact_slice).
Contenido del zip (rutas relativas, orden y fechas fijos: el SHA256 es reproducible):
  colab_runner.py, manifest_{piezo,blocks,smoke,smoke_blocks}.json, README.md, BUNDLE_INFO.json, SHA256SUMS.txt (de todo lo demas)
  bundle_piezo/    = los 7 archivos de piezo/build_bundle_piezo.py SOURCES (run_piezo.py, artifacts.py, piezo_problem.py, gram_preconditioning.py, residual_terms.py, FEM converso y directo)
  bundle_blocks/   = kaggle/dataset_1e/* (lo mismo que se subio a Kaggle para 1e: run_phase1c/1c_dfg/1d/1e + datos) + artifact_slice/run_slice.py + artifacts_core.py
Dos carpetas porque piezo y 1c/1e usan versiones distintas de residual_terms.py.
Garantias: se excluyen .env, claves, __pycache__, *.pyc y metadatos de Kaggle; se ESCANEA cada archivo (nombre y contenido) buscando claves de W&B/Kaggle/tokens y se aborta si aparece una.
Uso: python build_colab_bundle.py [--out dist/exp0028_colab_bundle.zip]   (imprime el SHA256 del zip; tambien lo escribe en <zip>.sha256)"""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent
ROOT = HERE.parents[3]
SKIP_NAMES = {"dataset-metadata.json", "kernel-metadata.json"}
FORBIDDEN_NAME = re.compile(r"(^|/)(\.env[^/]*|.*\.pem|.*\.key|.*credentials.*|kaggle\.json|.*secret.*|id_rsa.*)$", re.I)
SECRET_PATTERNS = [
    re.compile(r"wandb_v1_[A-Za-z0-9_]{20,}"),
    re.compile(r"WANDB_API_KEY\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\"key\"\s*:\s*\"[0-9a-f]{32}\""),
    re.compile(r"(?i)(api[_-]?key|token|password)\s*[=:]\s*['\"][A-Za-z0-9_\-]{24,}['\"]"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}"),
]
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load_piezo_sources():
    spec = importlib.util.spec_from_file_location("bb_piezo", str(EXP28 / "piezo" / "build_bundle_piezo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)             # solo define constantes (SOURCES); no ejecuta main
    return {n: Path(p) for n, p in mod.SOURCES.items()}


def scan(name, data):
    if FORBIDDEN_NAME.search(name):
        raise SystemExit(f"ABORTA: nombre prohibido en el bundle: {name}")
    if name.endswith((".npz", ".mat", ".dat", ".pdf", ".png")):
        return
    text = data.decode("utf-8", errors="ignore")
    for pat in SECRET_PATTERNS:
        m = pat.search(text)
        if m:
            raise SystemExit(f"ABORTA: posible credencial en {name}: patron {pat.pattern[:40]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "dist" / "exp0028_colab_bundle.zip"))
    a = ap.parse_args()
    members = {}                              # ruta en el zip -> bytes
    for n, src in load_piezo_sources().items():
        members[f"bundle_piezo/{n}"] = src.read_bytes()
    for n in ("gram_preconditioning.py", "residual_terms.py"):
        assert members[f"bundle_piezo/{n}"] == (ROOT / "src/mechanisms" / n).read_bytes(), n
    d1e = EXP28 / "kaggle" / "dataset_1e"
    if not d1e.exists():
        raise SystemExit("falta kaggle/dataset_1e (se construye con kaggle/build_bundle_1e.py); no se inventa")
    for p in sorted(d1e.iterdir()):
        if p.name not in SKIP_NAMES and p.is_file():
            members[f"bundle_blocks/{p.name}"] = p.read_bytes()
    for n in ("run_slice.py", "artifacts_core.py"):
        members[f"bundle_blocks/{n}"] = (EXP28 / "artifact_slice" / n).read_bytes()
    assert members["bundle_blocks/gram_preconditioning.py"] == (ROOT / "src/mechanisms/gram_preconditioning.py").read_bytes(), "gram_preconditioning.py de dataset_1e difiere de src/"
    for n in ("colab_runner.py", "README.md", "manifest_piezo.json", "manifest_blocks.json", "manifest_smoke.json", "manifest_smoke_blocks.json"):
        if not (HERE / n).exists():
            raise SystemExit(f"falta colab/{n} (correr make_artifacts.py / escribir el README)")
        members[n] = (HERE / n).read_bytes()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT), timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    for name, data in members.items():
        scan(name, data)
    sums = "".join(f"{sha(data)}  {name}\n" for name, data in sorted(members.items()))
    members["SHA256SUMS.txt"] = sums.encode()
    members["BUNDLE_INFO.json"] = json.dumps(dict(experiment="EXP-0028", purpose="Colab A100 pay-as-you-go", git_commit=commit, n_files=len(members) - 1, credentials_included=False,
                                                  wandb="offline; sin clave", sources=dict(piezo="piezo/build_bundle_piezo.py SOURCES", blocks="kaggle/dataset_1e + artifact_slice/run_slice.py + artifacts_core.py")), indent=1).encode()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(members):
            zi = zipfile.ZipInfo(name, FIXED_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, members[name])
    digest = sha(out.read_bytes())
    Path(str(out) + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(json.dumps(dict(zip=str(out), sha256=digest, bytes=out.stat().st_size, files=len(members), git_commit=commit)))
    print("SHA256:", digest)


if __name__ == "__main__":
    sys.exit(main())
