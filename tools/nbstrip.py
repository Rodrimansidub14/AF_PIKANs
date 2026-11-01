"""Filtro 'clean' de git para .ipynb: quita salidas y contadores, deja metadatos de investigación (metadata.research).
Uso por git (stdin -> stdout). Manual:  python tools/nbstrip.py --check <archivos>   (sale con 1 si alguno trae salidas)."""
import json
import sys


def limpiar(nb):
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
        c.get("metadata", {}).pop("collapsed", None)
    nb.get("metadata", {}).pop("widgets", None)
    return nb


def tiene_salidas(nb):
    return any(c.get("outputs") for c in nb.get("cells", []) if c.get("cell_type") == "code")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        malos = [p for p in sys.argv[2:] if tiene_salidas(json.load(open(p, encoding="utf-8")))]
        print("\n".join(malos))
        sys.exit(1 if malos else 0)
    nb = limpiar(json.loads(sys.stdin.buffer.read().decode("utf-8")))
    sys.stdout.buffer.write((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
