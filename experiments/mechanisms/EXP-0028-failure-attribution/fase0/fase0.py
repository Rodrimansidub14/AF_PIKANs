"""EXP-0028 Fase 0: reanalisis uniforme (CPU, datos guardados) de las palancas.

Criterio uniforme (spec, Enmienda v1.1), por palanca x problema, pareado por semilla con B:
  sirve:        mediana(e_X/e_B) <= 0.90, mejora (ratio<1) en >= 70 % de las semillas, ninguna semilla > 2x
  daña:         mediana >= 1.10 y empeora (ratio>1) en >= 70 %
  dependiente:  ni una ni otra y alguna semilla fuera de [0.5, 2]
  indiferente:  el resto
Solo lee archivos guardados. Ejecutar desde experiments/mechanisms.
"""
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.abspath(os.path.dirname(__file__))
os.chdir(ROOT)


def jl(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def gl(pat):
    return sorted(p.replace(chr(92), '/') for p in glob.glob(pat, recursive=True))


PROB = {'ac': 'allen_cahn_d1e-3', 'ac_d2e-4': 'allen_cahn_d2e-4', 'burgers': 'burgers',
        'kpp': 'fisher_kpp', 'heat': 'heat', 'allen_cahn': 'allen_cahn_d1e-3',
        'ns2d_cg': 'ns2d_cg', 'dfg': 'dfg', 'piezo': 'piezo'}

rows = []  # una fila por (exp, bloque, palanca-brazo, problema, semilla, brazo, base)


def add(exp, block, lever, arm, base, problem, seed, e_final, e_adam, note='', oracle=False, supervised=False):
    rows.append(dict(exp=exp, block=block, lever=lever, arm=arm, base=base, problem=PROB.get(problem, problem),
                     seed=int(seed), e_final=e_final, e_adam=e_adam, note=note, oracle=oracle,
                     supervised=supervised))


# ---- EXP-0023 (palanca 3: F, J, E, I vs B) ----
blocks23 = {'T1': 'EXP-0023-gram-preconditioning/results/t1-block/exp0023_t1block',
            'V': 'EXP-0023-gram-preconditioning/results/v-block/exp0023_vblock',
            'C1T1': 'EXP-0023-gram-preconditioning/results/c1-T1/exp0023_c1block',
            'C1V': 'EXP-0023-gram-preconditioning/results/c1-V/exp0023_c1vblock',
            'P': 'EXP-0023-gram-preconditioning/results/c1-P/exp0023_pblock'}
for blk, d in blocks23.items():
    for f in gl(d + '/*.json'):
        if f.endswith('.note.json') or 'ledger' in f or 'progress' in f:
            continue
        r = jl(f)
        if 'primary_final' not in r:
            continue
        add('EXP-0023', blk, 'gram', r['arm'], 'B', r['problem'], r['seed'], r['primary_final'], r['primary_adam'])

# ---- EXP-0024 T1 (palanca 2: D1-D3 vs CTRL; reubicacion de centros ocultos) ----
for f in gl('EXP-0024-wind-tunnel-protocol/results/t1-block/t1block/*.json'):
    if 'ledger' in f or 'progress' in f:
        continue
    r = jl(f)
    add('EXP-0024', 'T1', 'regrid', r['design'], 'CTRL', r['problem'], r['seed'], r['primary_final'], r['primary_adam'])
# EXP-0024 T0 (CPU, presupuesto corto a600/l250/n3000; AC y Heat)
for f in gl('EXP-0024-wind-tunnel-protocol/results/t0/*.json'):
    r = jl(f)
    add('EXP-0024', 'T0', 'regrid', r['design'], 'CTRL', r['problem'], r['seed'], r['final']['u'], r['adam']['u'],
        note='CPU, Adam 600/L-BFGS 250/n=3000')

# ---- EXP-0025 (palanca 1: R; observador O/Om/Orand; E=aG fijo) ----
for f in gl('EXP-0025-observer-v0/results/block-T1/exp0025_block_t1/*-seed*.json'):
    r = jl(f)
    lever = {'R': 'rad', 'E': 'gram(aG,K500)'}.get(r['arm'], 'observer')
    add('EXP-0025', 'T1', lever, r['arm'], 'B', r['problem'], r['seed'], r['primary_final'], r['primary_adam'])

# ---- EXP-0026 (E500/E1000/E500L/E1000L vs B) ----
for f in gl('EXP-0026-e-burgers-diagnostic/results/exp0026/*-*.json'):
    if any(s in f for s in ('ledger', 'rad_diag')):
        continue
    r = jl(f)
    if 'primary_final' not in r:
        continue
    s = r['seed']
    blk = 'S3000' if s >= 3000 else ('S101' if s >= 100 else 'S1-5')
    add('EXP-0026', blk, 'gram', r['arm'], 'B', r['problem'], s, r['primary_final'], r['primary_adam'])

# ---- EXP-0018 (palanca 4: P, H, N (control), O vs B; Adam compartido) ----
for f in gl('EXP-0018-responsibility-ceiling/results/ceiling/ceiling/*-seed*.json'):
    r = jl(f)
    for arm, v in r['arms'].items():
        add('EXP-0018', 'ceil', 'resp_focus', arm, 'B', r['problem'], r['seed'], v['global_l2'],
            r['start']['global_l2'], note='L-BFGS desde el mismo estado de Adam (e_adam identico: no informativo)',
            oracle=arm in ('P', 'H', 'N'))
# fix e_adam: identical => ratio 1.0; se marca NaN para no contarlo
for x in rows:
    if x['exp'] == 'EXP-0018':
        x['e_adam'] = np.nan

# ---- EXP-0016 (palanca 5: traspaso Adam->L-BFGS en k vs k=1.00) ----
a = jl('EXP-0016-stall-handoff/results/stage-a/stage_a_analysis.json')
for prob, rows16 in (('burgers', a['burgers']['per_seed']), ('ac', a['ac']['per_seed'])):
    for ps in rows16:
        for rr in ps['rows']:
            add('EXP-0016', 'stageA', 'handoff', 'k%.2f' % rr['f'], 'k1.00', prob, ps['seed'], rr['E'], rr['E_adam'],
                note='Adam parcial al paso k; "e_adam"= error del estado de Adam en k (no es comparable al final)')
# ---- EXP-0015 (palanca 2/7: bases/centros; AJUSTE SUPERVISADO, no entrenamiento fisico) ----
for f in gl('EXP-0015-layer0-basis-refinement/step0/results/*-seed*.json'):
    r = jl(f)
    m = re.match(r'(.+)-(C|O11|U41|X11)-seed(\d+)\.json', os.path.basename(f))
    prob, arm, seed = m.group(1), m.group(2), m.group(3)
    add('EXP-0015', 'step0', 'bases', arm, 'C', prob, seed, r['final']['u'], r['adam']['u'],
        note='ajuste supervisado a la referencia (MSE); C = 11 uniformes', supervised=True)

df = pd.DataFrame(rows)
df['isbase'] = df['arm'] == df['base']
b = df[df.isbase][['exp', 'block', 'problem', 'seed', 'e_final', 'e_adam']].rename(
    columns={'e_final': 'b_final', 'e_adam': 'b_adam'})
b = b.drop_duplicates(['exp', 'block', 'problem', 'seed'])
pr = df[~df.isbase].merge(b, on=['exp', 'block', 'problem', 'seed'], how='left')
pr['r_final'] = pr['e_final'] / pr['b_final']
pr['r_adam'] = pr['e_adam'] / pr['b_adam']
pr['no_pair'] = pr['b_final'].isna()
pr.to_csv(os.path.join(OUT, 'pairs.csv'), index=False)


def label(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 'sin pareo', np.nan
    med = float(np.median(r))
    n = len(r)
    imp = int((r < 1).sum())
    wor = int((r > 1).sum())
    if med <= 0.90 and imp / n >= 0.7 and r.max() <= 2.0:
        return 'sirve', med
    if med >= 1.10 and wor / n >= 0.7:
        return 'daña', med
    if (r.max() > 2.0) or (r.min() < 0.5):
        return 'dependiente de semilla', med
    return 'indiferente', med


def summarize(g, col):
    r = g[col].to_numpy(float)
    ok = np.isfinite(r)
    if ok.sum() == 0:
        return dict(n=0, k_mejora=0, med=np.nan, gm=np.nan, rmax=np.nan, rmin=np.nan, etiqueta='sin pareo')
    r = r[ok]
    lab, med = label(r)
    return dict(n=len(r), k_mejora=int((r < 1).sum()), med=med, gm=float(np.exp(np.mean(np.log(r)))),
                rmax=float(r.max()), rmin=float(r.min()), etiqueta=lab)


out = []
keys = ['exp', 'lever', 'arm', 'base', 'problem']
for k, g in pr.groupby(keys):
    for scope, gg in ([('pooled', g)] if k[0] != 'EXP-0024' else []) + [(b_, x) for b_, x in g.groupby('block')]:
        for stage, col in (('lbfgs_final', 'r_final'), ('adam_end', 'r_adam')):
            s = summarize(gg, col)
            seeds = sorted(gg.seed.unique())
            out.append(dict(zip(keys, k), block=scope, stage=stage, **s,
                            seeds=','.join(map(str, seeds)),
                            oracle=bool(gg.oracle.any()), supervised=bool(gg.supervised.any()),
                            n_menor_5=bool(s['n'] < 5), note=gg.note.iloc[0]))
tab = pd.DataFrame(out)
tab = tab.rename(columns={'med': 'mediana_ratio', 'gm': 'media_geom_ratio', 'rmax': 'ratio_max', 'rmin': 'ratio_min'})
tab.to_csv(os.path.join(OUT, 'table.csv'), index=False, encoding='utf-8')
print(len(pr), 'pares;', len(tab), 'filas de tabla')

# --------- consistencia: B de 0023 vs B de 0026 en las mismas semillas -----------
chk = []
for blkA, blkB in (('T1', 'S1-5'), ('V', 'S101')):
    for prob in ('burgers', 'allen_cahn_d1e-3'):
        a_ = df[(df.exp == 'EXP-0023') & (df.block == blkA) & (df.arm == 'B') & (df.problem == prob)].set_index('seed')
        b_ = df[(df.exp == 'EXP-0026') & (df.block == blkB) & (df.arm == 'B') & (df.problem == prob)].set_index('seed')
        j = a_[['e_final']].join(b_[['e_final']], lsuffix='_0023', rsuffix='_0026', how='inner')
        if len(j):
            q = j.e_final_0026 / j.e_final_0023
            chk.append(dict(problem=prob, bloque=f'{blkA}/{blkB}', n=len(j), ratio_B26_sobre_B23_mediana=float(q.median()),
                            ratio_max=float(q.max()), ratio_min=float(q.min())))
pd.DataFrame(chk).to_csv(os.path.join(OUT, 'check_B_repro.csv'), index=False)
print(pd.DataFrame(chk))

# ---------------- figuras ---------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

LAB_ABBR = {'sirve': 'S', 'daña': 'D', 'dependiente de semilla': 'd', 'indiferente': 'i', 'sin pareo': ''}
lev_name = {'gram': 'Gram', 'regrid': 'Centros ocultos', 'rad': 'RAD', 'observer': 'Observador',
            'gram(aG,K500)': 'Gram(aG)', 'resp_focus': 'Foco resp.', 'handoff': 'Traspaso', 'bases': 'Bases(aj.sup.)'}
pl = tab[(tab.block == 'pooled') & (tab.exp != 'EXP-0024')].copy()
t1 = tab[(tab.exp == 'EXP-0024') & (tab.block == 'T1')].copy()
t0 = tab[(tab.exp == 'EXP-0024') & (tab.block == 'T0')].copy()
pl = pd.concat([pl, t1, t0], ignore_index=True)


def rowname(x):
    s = f"{lev_name.get(x.lever, x.lever)} {x.arm} [{x.exp[4:]}"
    if x.exp == 'EXP-0024':
        s += ' T0 CPU' if x.block == 'T0' else ' T1 GPU'
    if x.oracle:
        s += ' ORACULO'
    return s + ']'


pl['row'] = pl.apply(rowname, axis=1)
order_exp = ['EXP-0023', 'EXP-0026', 'EXP-0025', 'EXP-0024', 'EXP-0018', 'EXP-0016', 'EXP-0015']
pl['oe'] = pl.exp.map({e: i for i, e in enumerate(order_exp)})
pl = pl.sort_values(['oe', 'row'])
rows_ord = list(dict.fromkeys(pl.row))
cols_ord = ['allen_cahn_d1e-3', 'allen_cahn_d2e-4', 'burgers', 'fisher_kpp', 'heat', 'ns2d_cg', 'dfg', 'piezo']


def heat(stage, fname, title):
    sub = pl[pl.stage == stage]
    M = np.full((len(rows_ord), len(cols_ord)), np.nan)
    A = [['' for _ in cols_ord] for _ in rows_ord]
    for _, x in sub.iterrows():
        i = rows_ord.index(x.row)
        j = cols_ord.index(x.problem)
        if x.n > 0 and np.isfinite(x.mediana_ratio):
            M[i, j] = np.log10(x.mediana_ratio)
            A[i][j] = f"{x.mediana_ratio:.2f}\n{int(x.k_mejora)}/{int(x.n)} {LAB_ABBR[x.etiqueta]}"
    fig, ax = plt.subplots(figsize=(1.35 * len(cols_ord) + 4, 0.42 * len(rows_ord) + 2.2))
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('#e6e6e6')
    im = ax.imshow(np.ma.masked_invalid(np.clip(M, -1.5, 1.5)), cmap=cm, norm=norm, aspect='auto')
    for i in range(len(rows_ord)):
        for j in range(len(cols_ord)):
            if A[i][j]:
                ax.text(j, i, A[i][j], ha='center', va='center', fontsize=6, color='k')
    ax.set_xticks(range(len(cols_ord)))
    ax.set_xticklabels(cols_ord, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(rows_ord)))
    ax.set_yticklabels(rows_ord, fontsize=7)
    ax.set_title(title + '\nlog10 mediana e_X/e_B (azul = mejora, rojo = empeora, gris = sin datos); celda: mediana, '
                 'k/n con mejora, S=sirve D=daña d=dep. de semilla i=indiferente', fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('log10 mediana (recortado a +-1.5)', fontsize=7)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{fname}.{ext}'), dpi=170)
    plt.close(fig)


heat('lbfgs_final', 'heatmap_palanca_problema_lbfgs', 'EXP-0028 Fase 0, error final (tras L-BFGS)')
heat('adam_end', 'heatmap_palanca_problema_adam', 'EXP-0028 Fase 0, error al final de Adam')

# ---------------- Spearman entre efectos por semilla -----------------
from scipy.stats import spearmanr, rankdata


def _avg_rank(a):
    lt = (a[:, :, None] > a[:, None, :]).sum(-1)
    eq = (a[:, :, None] == a[:, None, :]).sum(-1)
    return lt + (eq + 1) / 2.0


def boot_spearman(x, y, rng, B=2000):
    n = len(x)
    ix = rng.integers(0, n, (B, n))
    xs, ys = x[ix], y[ix]
    ok = (np.ptp(xs, axis=1) > 0) & (np.ptp(ys, axis=1) > 0)
    rx, ry = _avg_rank(xs[ok]), _avg_rank(ys[ok])
    rx = rx - rx.mean(1, keepdims=True)
    ry = ry - ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


rng = np.random.default_rng(0)
sp_rows = []
fig_data = {}
use = pr[(~pr.supervised) & (pr.exp.isin(['EXP-0023', 'EXP-0024', 'EXP-0025', 'EXP-0026', 'EXP-0018', 'EXP-0016']))]
use = use[~((use.exp == 'EXP-0024') & (use.block == 'T0'))]
use = use.assign(col=use.apply(lambda x: f"{x.arm}[{x.exp[4:]}]", axis=1))
for prob, g in use.groupby('problem'):
    M = g.pivot_table(index='seed', columns='col', values='r_final', aggfunc='first')
    M = np.log(M)
    cols = list(M.columns)
    S = np.full((len(cols), len(cols)), np.nan)
    N = np.zeros((len(cols), len(cols)), int)
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            m = M[[ci, cj]].dropna()
            N[i, j] = len(m)
            if i == j:
                S[i, j] = 1
                continue
            if len(m) >= 5:
                rho = spearmanr(m[ci], m[cj]).statistic
                S[i, j] = rho
                if j > i:
                    bs = boot_spearman(m[ci].to_numpy(), m[cj].to_numpy(), rng)
                    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan))
                    sp_rows.append(dict(problem=prob, a=ci, b=cj, n_semillas=len(m), spearman=rho,
                                        ic95_lo=lo, ic95_hi=hi))
    fig_data[prob] = (cols, S, N)
sp = pd.DataFrame(sp_rows)
sp.to_csv(os.path.join(OUT, 'spearman_efectos.csv'), index=False)
print(len(sp), 'pares con >=5 semillas comunes')

probs = []
for p, (c, S, N) in fig_data.items():
    off = S.copy()
    np.fill_diagonal(off, np.nan)
    if np.isfinite(off).any():
        probs.append(p)
if probs:
    fig, axs = plt.subplots(1, len(probs), figsize=(6.5 * len(probs), 6.5), squeeze=False)
    for ax, p in zip(axs[0], probs):
        cols, S, N = fig_data[p]
        off = S.copy()
        np.fill_diagonal(off, np.nan)
        keep = [i for i in range(len(cols)) if np.isfinite(off[i]).any()]
        S2 = S[np.ix_(keep, keep)]
        c2 = [cols[i] for i in keep]
        im = ax.imshow(S2, cmap='RdBu_r', vmin=-1, vmax=1)
        for i in range(len(keep)):
            for j in range(len(keep)):
                if np.isfinite(S2[i, j]) and i != j:
                    ax.text(j, i, f"{S2[i, j]:.1f}\nn={N[keep[i], keep[j]]}", ha='center', va='center', fontsize=5.5)
        ax.set_xticks(range(len(c2)))
        ax.set_xticklabels(c2, rotation=90, fontsize=6)
        ax.set_yticks(range(len(c2)))
        ax.set_yticklabels(c2, fontsize=6)
        ax.set_title(f"{p}: Spearman entre log(e_X/e_B) por semilla (>=5 semillas comunes)", fontsize=8)
    fig.colorbar(im, ax=axs[0].tolist(), fraction=0.02)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'spearman_efectos.{ext}'), dpi=160, bbox_inches='tight')
    plt.close(fig)
print('listo')
