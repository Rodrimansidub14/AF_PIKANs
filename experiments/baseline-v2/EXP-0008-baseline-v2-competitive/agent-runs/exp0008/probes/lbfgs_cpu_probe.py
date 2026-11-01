"""Tier-0 timing probe: L-BFGS continuation of a v1 Burgers checkpoint on CPU.
Read-only on repo artifacts; writes only to the scratchpad. Scoring vs Cole-Hopf uses
the verifier's reference function (import only, no modification)."""
import sys, time, json, os
import numpy as np, torch
sys.path.insert(0, r'C:\Users\rodri\Documents\AF_PIKANs')
from src.research_pipeline.sciml.model import build_model, derivatives
from evals.problems.burgers_pinnacle_1d import reference as ref

seed_dir, budget_s, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get('NT', '4')))
ck = torch.load(os.path.join(seed_dir, 'checkpoint-020000.pt'), map_location='cpu')
pp = ck['problem_parameters']; nu = pp['nu']; scale = pp['residual_scale']
model = build_model('burgers', {'id': 'rbf_fixed'}).double(); model.load_state_dict(ck['state_dict'])
d = np.load(os.path.join(seed_dir, 'pools.npz'))
T = lambda k: torch.tensor(d[k], dtype=torch.float64)
xi, xl, xr, x0 = T('interior'), T('left'), T('right'), T('initial')

cache = os.path.join(os.path.dirname(out), 'ch_ref_1025x129.npy')
gx, gt = ref.eval_grid()
if os.path.exists(cache): uref = np.load(cache)
else:
    t0 = time.time(); uref = ref.exact(gx, gt); np.save(cache, uref); print('ref time', time.time() - t0, flush=True)
G = torch.tensor(np.stack([gx, gt], 1))

def score():
    with torch.no_grad(): u = model(G).squeeze(-1).numpy()
    return float(np.linalg.norm(u - uref) / np.linalg.norm(uref))

def loss_fn():
    x = xi.clone().requires_grad_(True)
    u = model(x); ux, ut, uxx = derivatives(u, x)
    lp = (((ut + u * ux - nu * uxx) / scale) ** 2).mean()
    li = ((model(x0).squeeze(-1) + torch.sin(torch.pi * x0[:, 0])) ** 2).mean()
    ll = (model(xl) ** 2).mean(); lr = (model(xr) ** 2).mean()
    return lp + li + ll + lr

opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, max_eval=25, history_size=100,
                        tolerance_grad=1e-8, tolerance_change=0.0, line_search_fn='strong_wolfe')
def closure():
    opt.zero_grad(); l = loss_fn(); l.backward(); return l

rows = [{'outer': 0, 'sec': 0.0, 'loss': float(loss_fn()), 'l2_ch': score()}]
print(rows[-1], flush=True)
t0 = time.time(); k = 0
while time.time() - t0 < budget_s:
    opt.step(closure); k += 1
    if k in (1, 5) or k % 25 == 0:
        rows.append({'outer': k, 'sec': round(time.time() - t0, 1), 'loss': float(loss_fn()), 'l2_ch': score(),
                     'fevals': opt.state[opt._params[0]].get('func_evals')})
        print(rows[-1], flush=True)
        json.dump(rows, open(out, 'w'))
rows.append({'outer': k, 'sec': round(time.time() - t0, 1), 'loss': float(loss_fn()), 'l2_ch': score(), 'final': True})
print(rows[-1], flush=True); json.dump(rows, open(out, 'w'))
