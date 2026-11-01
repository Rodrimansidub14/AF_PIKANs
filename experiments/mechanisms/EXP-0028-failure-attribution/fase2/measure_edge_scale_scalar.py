#!/usr/bin/env python3
"""EXP-0028 fase 2 (ampliacion): estadisticos de escala de traza entre aristas (lo que reescala J) para AC/Burgers/KPP (gn_blocks escalar, mismo harness que measure_properties.py).
Misma definicion que measure_properties_terms.edge_trace_stats. Uso: python measure_edge_scale_scalar.py --problem ac_d1e-3 --seeds 4001-4003 --out CSV"""
import argparse, csv, sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import measure_properties as mp


def edge_trace_stats(blocks):
    tr = [B.reshape(-1, B.shape[-2], B.shape[-1]).diagonal(dim1=-2, dim2=-1).sum(-1) for B in blocks]
    allt = torch.cat(tr).clamp_min(1e-300)
    srt = torch.sort(allt, descending=True).values
    k = max(1, int(round(0.1 * srt.numel())))
    med = torch.stack([t.median() for t in tr])
    q = torch.quantile(allt, torch.tensor([0.05, 0.95], dtype=allt.dtype))
    return dict(edge_trace_log10_p95_p5=float(torch.log10(q[1] / q[0])), edge_trace_log10_max_min=float(torch.log10(srt[0] / srt[-1])),
                edge_trace_top10=float(srt[:k].sum() / srt.sum()), layer_trace_log10_ratio=float(torch.log10(med.max() / med.min())))


ap = argparse.ArgumentParser(); ap.add_argument("--problem", required=True); ap.add_argument("--seeds", default="4001-4003"); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args(); torch.set_num_threads(a.threads)
rows = []
for s in mp.parse_seeds(a.seeds):
    torch.manual_seed(s); b = mp.build(a.problem, s)
    st = edge_trace_stats(mp.edge_blocks(b))
    rows += [dict(problema=a.problem, semilla=s, propiedad=k, valor=repr(v)) for k, v in st.items()]
    print(a.problem, s, st, flush=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["problema", "semilla", "propiedad", "valor"]); w.writeheader(); w.writerows(rows)
print("escrito", a.out)
