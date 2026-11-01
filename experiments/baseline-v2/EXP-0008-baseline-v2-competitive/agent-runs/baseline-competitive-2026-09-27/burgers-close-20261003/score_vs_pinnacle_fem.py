"""CPU: score the saved Burgers checkpoints against PINNacle's author FEM/COMSOL reference (101 x 11 grid) for a like-for-like comparison with the published L2RE."""
import glob, json, math, os, sys, csv, statistics as st
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pikan_campaign as pc

def load_ref(path):
    lines = [l for l in open(path, encoding="utf-8").read().splitlines()]
    hdr = [l for l in lines if l.startswith("% X")][0]
    ts = [float(h.split("t=")[1].split()[0]) for h in hdr.split("u (1)")[1:]]  # t values
    data = np.array([[float(v) for v in l.split()] for l in lines if not l.startswith("%") and l.strip()])
    x = data[:, 0]; U = data[:, 1:]  # (nx, nt)
    return x, np.array(ts), U

x, ts, U = load_ref("reference/burgers1d_pinnacle_fem.dat")
print("ref", U.shape, ts)
X, T = np.meshgrid(x, ts, indexing="ij")
xt = torch.tensor(np.stack([X.ravel(), T.ravel()], 1), dtype=torch.float32)
rows = []
for run in sorted(glob.glob("outputs/burgers/results/burgers/*/run.json")):
    d = os.path.dirname(run); r = json.load(open(run))
    net = pc.BurgersProblem("reference/burgers_coleHopf_1025x129.npz").build(r["arm"], r["width"], r["bases"], r["seed"])
    out = {"arm": r["arm"], "width": r["width"] if r["arm"] == "rbf" else "", "seed": r["seed"]}
    for stage, f in (("adam", "checkpoint-adam-final.pt"), ("lbfgs", "checkpoint-lbfgs-final.pt")):
        net.load_state_dict(torch.load(os.path.join(d, f), map_location="cpu")["state_dict"]); net.eval()
        with torch.no_grad(): u = pc.BurgersProblem.transform(xt, net(xt)).numpy().reshape(U.shape)
        out[f"l2_fem_{stage}"] = float(np.linalg.norm(u - U) / np.linalg.norm(U))
    rows.append(out)
w = csv.DictWriter(open("burgers-vs-pinnacle-fem.csv", "w", newline=""), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for key in sorted({(r["arm"], r["width"]) for r in rows}, key=str):
    R = [r for r in rows if (r["arm"], r["width"]) == key]
    f = lambda k: "%.4e ± %.2e" % (st.mean(x[k] for x in R), st.stdev(x[k] for x in R))
    print(key, "n=%d" % len(R), "adam", f("l2_fem_adam"), "| lbfgs", f("l2_fem_lbfgs"))
