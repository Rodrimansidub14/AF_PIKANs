"""Peso seguro de P2 sin entrenar: para b en {0.05, 0.1, 0.2, 0.3, 0.5, 1}, B_mix = (1-b) B/trB + b B_bal/trB_bal por arista, sobre los estados finales guardados de E (converso y directo, semillas 28-29, 128 puntos por clave).
Mide, frente al bloque real regularizado: cond efectiva cond(P_mix (Bn + 1e-3 I) P_mix) (con P de E vale 1) y cuanto cambia la direccion (coseno del autovector dominante de B_mix contra el de B)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.argv = [sys.argv[0]]
import check_P2 as c  # noqa: E402

BS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
norm = lambda B: B / (B.diagonal(dim1=-2, dim2=-1).sum(-1) / B.shape[-1]).clamp_min(1e-300)[..., None, None]


def pinv_sqrt(B):
    w, V = torch.linalg.eigh(B + 1e-3 * torch.eye(B.shape[-1], dtype=B.dtype))
    return V @ torch.diag_embed(w.clamp_min(1e-300).rsqrt()) @ V.transpose(-1, -2)


rows = []
for mode in ("converse", "direct"):
    prob = c.rt.install(c.rp.make_problem(mode, 0))
    for seed in (20260728, 20260729):
        X = c.rp.gp.subset(prob.pools(seed, c.dev, c.dt, 0), 128)
        m = c.load_model(prob, mode, "E", seed)
        coefs = tuple(q.detach() for q in c.rp.gp.effective_coefficients(m))
        B, Bb = c.rt.gn_blocks_terms_groups(prob, m, X, coefs, 16)
        for li, (b0, bb0) in enumerate(zip(B, Bb)):
            Bn, Bbn = norm(b0), norm(bb0)
            live = b0.diagonal(dim1=-2, dim2=-1).sum(-1) > 1e-12 * b0.diagonal(dim1=-2, dim2=-1).sum(-1).max()
            v0 = torch.linalg.eigh(Bn)[1][..., -1]
            for b in BS:
                Bm = (1 - b) * Bn + b * Bbn
                P = pinv_sqrt(norm(Bm))
                M = P @ (Bn + 1e-3 * torch.eye(Bn.shape[-1], dtype=Bn.dtype)) @ P
                e = torch.linalg.eigvalsh(M)
                cond = e[..., -1] / e[..., 0].clamp_min(1e-300)
                cos = (v0 * torch.linalg.eigh(norm(Bm))[1][..., -1]).sum(-1).abs()
                rows.append(dict(problem=mode, seed=seed % 100, layer=li, b=b, cond_med=float(cond[live].median()), cond_p90=float(cond[live].quantile(0.9)),
                                 cos_lt09=float((cos[live] < 0.9).double().mean())))
        print(mode, seed % 100, flush=True)
out = ["# Peso seguro de P2 (mezcla b), estados finales de E, 128 puntos por clave, CPU\n",
       "Mediana sobre semillas 28-29 y capas de: cond efectiva mediana por arista (1 = como E) / cond efectiva percentil 90 / fracción de aristas cuya dirección cambia (coseno < 0.9).\n",
       "| problema | b | cond med | cond p90 | aristas con cambio de dirección |", "|---|---|---|---|---|"]
for mode in ("converse", "direct"):
    for b in BS:
        s = [r for r in rows if r["problem"] == mode and r["b"] == b]
        out.append(f"| {mode} | {b} | {np.median([r['cond_med'] for r in s]):.2f} | {np.median([r['cond_p90'] for r in s]):.2f} | {np.median([r['cos_lt09'] for r in s]):.2f} |")
Path("check_P2_mezcla.md").write_text("\n".join(out), encoding="utf-8")
json.dump(rows, open("check_P2_mezcla.json", "w"))
print("\n".join(out))
