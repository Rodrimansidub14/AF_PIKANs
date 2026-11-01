import json, glob, sys, os
d = sys.argv[1]
s = json.load(open(f'{d}/session_summary.json'))
print('wall', round(s['wall']), [(l.get('job'), l.get('rc')) for l in s['ledger']])
for f in sorted(glob.glob(f'{d}/results/*/run.json')):
    r = json.load(open(f)); a, b = r['score_adam_final'], r['score_lbfgs_final']
    print(os.path.basename(os.path.dirname(f)), '| adam u %.3f v %.3f tip %.2e | lbfgs u %.3f v %.3f phi %.4f tip %.2e it %d %s' % (a['u_relL2'], a['v_relL2'], a['tip_v'], b['u_relL2'], b['v_relL2'], b['phi_relL2'], b['tip_v'], r['lbfgs_iterations'], r['lbfgs_status']), {k: round(v, 2) for k, v in r.get('weights_final', {}).items()})
