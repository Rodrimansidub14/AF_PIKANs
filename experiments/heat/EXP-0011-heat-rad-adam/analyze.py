import json, glob, statistics as st, math, os
d = 'results/run-20261004'
s = json.load(open(f'{d}/session_summary.json')); print('wall', round(s['wall']), [(x.get('job'), x.get('rc'), round(x.get('wall', 0))) for x in s['done']])
ctl = {x['seed']: x['adam_final_rel_l2'] for x in json.load(open('../EXP-0008-baseline-v2-competitive/results/heat-lbfgs-redo-20260930/results-summary.json', encoding='utf-8'))}
ctl_l = {x['seed']: x['rel_l2_512x256'] for x in json.load(open('../EXP-0008-baseline-v2-competitive/results/heat-lbfgs-redo-20260930/results-summary.json', encoding='utf-8'))}
rad, rep = {}, {}
for f in sorted(glob.glob(f'{d}/results/rad/*/result.json')):
    r = json.load(open(f)); rad[r['seed']] = r
for f in sorted(glob.glob(f'{d}/results/ctl/*/result.json')):
    r = json.load(open(f)); rep[r['seed']] = r
print('RAD seeds', sorted(rad), 'replicate seeds', sorted(rep))
print('seed | ctl(EXP-0008) | ctl replicate | RAD end-of-Adam | ratio RAD/ctl | RAD after LBFGS | ctl after LBFGS | ratio | adam_s | refresh_s | stop')
ratios, lratios = [], []
for sd in sorted(rad):
    r = rad[sd]; c = ctl[sd]; ra = r['end_of_adam_rel_l2_512x256']; ratios.append(ra / c)
    la = r['after_lbfgs_rel_l2_512x256']; lratios.append(la / ctl_l[sd])
    print(sd, '%.3e' % c, ('%.3e' % rep[sd]['end_of_adam_rel_l2_512x256']) if sd in rep else '-', '%.3e' % ra, '%.3f' % (ra / c), '%.3e' % la, '%.3e' % ctl_l[sd], '%.3f' % (la / ctl_l[sd]), round(r['adam_seconds']), round(r['refresh_seconds'], 1), r['lbfgs_stop_reason'], r['lbfgs_iters'])
gm = math.exp(st.mean(math.log(x) for x in ratios)); glm = math.exp(st.mean(math.log(x) for x in lratios))
print('GEOMEAN end-of-Adam ratio %.3f ; seeds with ratio<=0.80: %d/5 ; mean RAD %.3e +- %.2e vs ctl %.3e +- %.2e' % (gm, sum(x <= 0.8 for x in ratios), st.mean(rad[s_]['end_of_adam_rel_l2_512x256'] for s_ in rad), st.stdev(rad[s_]['end_of_adam_rel_l2_512x256'] for s_ in rad), st.mean(ctl[s_] for s_ in rad), st.stdev(ctl[s_] for s_ in rad)))
print('GEOMEAN after-LBFGS ratio %.3f (guard <= 2.0)' % glm)
json.dump({'ratios_end_of_adam': ratios, 'geomean': gm, 'n_le_0.8': sum(x <= 0.8 for x in ratios), 'ratios_after_lbfgs': lratios, 'geomean_lbfgs': glm}, open(f'{d}/analysis.json', 'w'), indent=1)
