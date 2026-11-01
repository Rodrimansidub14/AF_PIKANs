import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import nbstrip  # noqa: E402


class NbstripTest(unittest.TestCase):
    def test_quita_salidas_y_conserva_metadatos_de_investigacion(self):
        nb = {"metadata": {"research": {"notebook_id": "x", "role": "analysis"}, "widgets": {}},
              "cells": [{"cell_type": "code", "source": "1+1", "outputs": [{"text": "2"}], "execution_count": 3, "metadata": {"collapsed": True}},
                        {"cell_type": "markdown", "source": "hola", "metadata": {}}]}
        self.assertTrue(nbstrip.tiene_salidas(nb))
        limpio = nbstrip.limpiar(nb)
        self.assertFalse(nbstrip.tiene_salidas(limpio))
        self.assertEqual(limpio["cells"][0]["execution_count"], None)
        self.assertEqual(limpio["metadata"]["research"]["role"], "analysis")
        self.assertNotIn("widgets", limpio["metadata"])


class RegistroTest(unittest.TestCase):
    def test_run_id_unicos_y_jsonl_valido(self):
        import json
        ids = [json.loads(l)["run_id"] for l in (ROOT / "registry" / "runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
