import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import matplotlib.tri as mtri
D = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 150, "savefig.bbox": "tight"})
NAMES = {"heat_1d": "Heat 1D", "burgers_1d": "Burgers 1D", "allen_cahn_1d": "Allen–Cahn 1D", "ns2d_cg": "NS2d-CG (Re=100)", "dfg_2d": "DFG 2D-1 (Re=20)", "piezo_2d": "Piezo, bimorfo PVDF"}
ARMNAME = {"rbf": "RBF-PIKAN", "mlp": "MLP"}
p = sys.argv[1]
z = np.load(D / f"fields_{p}.npz"); meta = json.load(open(D / f"fields_{p}.json"))
pts = z["points"]; comps = [k[4:] for k in z.files if k.startswith("ref_")]
arms = [a for a in ("rbf", "mlp") if f"{a}_{comps[0]}" in z.files]
stage = {"lbfgs_final": "tras L-BFGS", "adam_final": "fin de Adam"}[meta["stage"]]
sub = f"{meta['campaign']}, semilla {meta['seed']}, {stage}"

def cmap_norm(v):
    if v.min() < 0 < v.max():
        m = np.abs(v).max(); return "RdBu_r", TwoSlopeNorm(0, -m, m)
    return "viridis", plt.Normalize(v.min(), v.max())

if p in ("heat_1d", "burgers_1d", "allen_cahn_1d"):
    nx, nt = z["shape"]; X = pts[:, 0].reshape(nx, nt); T = pts[:, 1].reshape(nx, nt)
    ref = z["ref_u"].reshape(nx, nt)
    cm, nm = cmap_norm(ref)
    panels = [("Referencia", ref)] + [(f"{ARMNAME[a]} (L2 {meta['l2'][a+'_u']:.1e})", z[f"{a}_u"].reshape(nx, nt)) for a in arms]
    fig = plt.figure(figsize=(4.6 * len(panels), 4.2))
    for i, (t, F) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        ax.plot_surface(X, T, F, cmap=cm, norm=nm, rstride=max(nx // 120, 1), cstride=max(nt // 60, 1), linewidth=0, antialiased=True)
        ax.set_title(t); ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("u"); ax.view_init(28, -125)
    fig.suptitle(f"{NAMES[p]}: u(x, t) — {sub}", y=1.02)
    fig.savefig(D / f"fig_{p}_3d.png"); plt.close(fig)
    errs = [(a, np.abs(z[f"{a}_u"] - z["ref_u"]).reshape(nx, nt)) for a in arms]
    hi = max(e.max() for _, e in errs); lo = hi * 1e-4
    fig, axs = plt.subplots(1, len(errs), figsize=(5.2 * len(errs), 3.8), squeeze=False)
    for ax, (a, e) in zip(axs[0], errs):
        im = ax.pcolormesh(T, X, np.maximum(e, lo), norm=LogNorm(lo, hi), cmap="magma", shading="auto")
        ax.set_title(f"|error| {ARMNAME[a]} (L2 {meta['l2'][a+'_u']:.1e})"); ax.set_xlabel("t"); ax.set_ylabel("x")
    fig.colorbar(im, ax=axs[0].tolist(), label="|u − u_ref| (escala log)")
    fig.suptitle(f"{NAMES[p]}: dónde está el error — {sub}", y=1.03)
    fig.savefig(D / f"fig_{p}_error.png"); plt.close(fig)
else:
    x, y = pts[:, 0], pts[:, 1]
    tri = mtri.Triangulation(x, y)
    if p == "dfg_2d":
        cx, cy = x[tri.triangles].mean(1), y[tri.triangles].mean(1)
        tri.set_mask((cx - 0.2) ** 2 + (cy - 0.2) ** 2 < 0.05 ** 2)
    if p == "ns2d_cg":
        ed = np.hypot(np.diff(x[tri.triangles][:, [0, 1, 2, 0]], axis=1), np.diff(y[tri.triangles][:, [0, 1, 2, 0]], axis=1)).max(1)
        tri.set_mask(ed > 0.15)
    aspect = 40 if p == "piezo_2d" else 1  # piezo: espesor exagerado 40x
    ys = y * aspect
    trs = mtri.Triangulation(x, ys, tri.triangles, mask=tri.mask)
    cols = ["Referencia"] + [ARMNAME[a] for a in arms] + [f"|error| {ARMNAME[a]}" for a in arms]
    W = (x.max() - x.min()); H = (ys.max() - ys.min())
    pw = 3.6; ph = max(pw * H / W, 0.9)
    fig, axs = plt.subplots(len(comps), len(cols), figsize=(pw * len(cols), (ph + 0.9) * len(comps)), squeeze=False)
    for i, c in enumerate(comps):
        ref = z[f"ref_{c}"]; cm, nm = cmap_norm(ref)
        fields = [ref] + [z[f"{a}_{c}"] for a in arms]
        errs = [np.abs(z[f"{a}_{c}"] - ref) for a in arms]
        emax = max(e.max() for e in errs)
        for j, F in enumerate(fields):
            ax = axs[i, j]; im = ax.tripcolor(trs, F, cmap=cm, norm=nm, shading="gouraud")
            ax.set_title(cols[j] if j == 0 else f"{cols[j]} (L2 {meta['l2'][arms[j-1]+'_'+c]:.3f})", fontsize=9)
        fig.colorbar(im, ax=axs[i, :len(fields)].tolist(), shrink=0.8, label=c)
        for k, e in enumerate(errs):
            ax = axs[i, len(fields) + k]; ime = ax.tripcolor(trs, e, cmap="magma", vmin=0, vmax=emax, shading="gouraud"); ax.set_title(cols[len(fields) + k], fontsize=9)
        fig.colorbar(ime, ax=axs[i, len(fields):].tolist(), shrink=0.8, label=f"|error| {c}")
        for ax in axs[i]: ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    note = " (espesor exagerado 40×)" if p == "piezo_2d" else ""
    fig.suptitle(f"{NAMES[p]}: campos y error{note} — {sub}", y=1.01)
    fig.savefig(D / f"fig_{p}_fields.png"); plt.close(fig)
    prim = {"ns2d_cg": "u", "dfg_2d": "u", "piezo_2d": "v"}[p]
    panels = [("Referencia", z[f"ref_{prim}"])] + [(f"{ARMNAME[a]} (L2 {meta['l2'][a+'_'+prim]:.3f})", z[f"{a}_{prim}"]) for a in arms]
    cm, nm = cmap_norm(z[f"ref_{prim}"])
    fig = plt.figure(figsize=(4.6 * len(panels), 4.0))
    for i, (t, F) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        ax.plot_trisurf(trs, F, cmap=cm, norm=nm, linewidth=0, antialiased=True)
        ax.set_title(t); ax.set_xlabel("x"); ax.set_ylabel("y" + (" (×40)" if p == "piezo_2d" else "")); ax.set_zlabel(prim); ax.view_init(32, -120)
    fig.suptitle(f"{NAMES[p]}: {prim}(x, y) en 3D — {sub}", y=1.02)
    fig.savefig(D / f"fig_{p}_3d.png"); plt.close(fig)
print("ok", p)
