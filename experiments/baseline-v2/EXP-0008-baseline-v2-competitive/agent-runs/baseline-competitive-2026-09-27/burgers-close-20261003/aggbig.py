import json, glob, csv, statistics as st, os
rows = []
for f in sorted(glob.glob('outputsbig/burgersbig/results/burgersbig/*/run.json')):
    r = json.load(open(f)); a = r['score_adam_final']['l2re_coleHopf']; b = r.get('score_lbfgs_final', {}).get('l2re_coleHopf')
    rows.append(dict(arm=r['arm'], width=r['width'] if r['arm'] == 'rbf' else '', seed=r['seed'], params=r['params'], adam_steps=r['adam_steps'], adam_s=round(r['adam_seconds']),
                     l2_adam=a, l2_lbfgs=b, lb_status=r.get('lbfgs_status'), lb_it=r.get('lbfgs_iterations'), lb_s=round(r.get('lbfgs_seconds', 0))))
w = csv.DictWriter(open('burgers-big-per-seed.csv', 'w', newline=''), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('n runs', len(rows))
for key in sorted({(r['arm'], r['width'], r['params']) for r in rows}):
    R = [r for r in rows if (r['arm'], r['width'], r['params']) == key]
    f = lambda k: '%.4e ± %.2e' % (st.mean(x[k] for x in R), st.stdev(x[k] for x in R)) if len(R) > 1 else str(R[0][k])
    print(key, 'n=%d' % len(R), 'adam', f('l2_adam'), '| lbfgs', f('l2_lbfgs'), '|', sorted({x['lb_status'] for x in R}), 'it', [x['lb_it'] for x in R])
