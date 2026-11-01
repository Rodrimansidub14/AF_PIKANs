"""EXP-0024: los bloques de semillas de T0/T1, V y C son disjuntos entre sí y con las semillas ya usadas."""
import importlib.util
import os
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("seeds", str(ROOT / "experiments/mechanisms/EXP-0024-wind-tunnel-protocol/seeds.py"))
seeds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seeds)


def test_blocks_are_disjoint():
    t, v = set(seeds.T_BLOCK), set(seeds.V_BLOCK)
    assert not (t & v)
    assert not ((t | v) & seeds.USED)
    assert min(seeds.USED) < seeds.C_MIN and all(x < seeds.C_MIN for x in t | v)


def test_exp0023_confirmation_seeds_are_in_c_block():
    c = set(range(801, 806)) | set(range(811, 816))
    assert all(x >= seeds.C_MIN for x in c)
    assert not (c & (set(seeds.T_BLOCK) | set(seeds.V_BLOCK) | seeds.USED))
