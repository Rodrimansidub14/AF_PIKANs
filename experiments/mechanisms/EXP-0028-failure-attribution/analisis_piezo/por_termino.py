"""Traza del bloque Gauss-Newton por TERMINO de la perdida (no por grupo), estado final de E (semilla 28), 128 puntos por clave, CPU. Responde por que la cuota del grupo electrico es casi nula."""
import sys

sys.argv = [sys.argv[0], "128"]
sys.path.insert(0, ".")
import check_P2 as c  # noqa: E402

for mode in ("converse", "direct"):
    prob = c.rt.install(c.rp.make_problem(mode, 0))
    s = prob.spec
    print("\n==", mode)
    print("escalas constitutivas cs:", {k: float(f"{v:.3g}") for k, v in s["cscales"].items()})
    print("escalas sc:", {k: float(f"{v:.3g}") for k, v in s["scales"].items() if isinstance(v, (int, float))})
    print("material eps1, eps2, e31_top, e33_top:", s["mat"]["eps1"], s["mat"]["eps2"], s["mat"]["e31_top"], s["mat"]["e33_top"])
    seed = 20260728
    X = c.rp.gp.subset(prob.pools(seed, c.dev, c.dt, 0), 128)
    m = c.load_model(prob, mode, "E", seed)
    coefs = tuple(q.detach() for q in c.rp.gp.effective_coefficients(m))
    tb = c.term_blocks(prob, m, X, chunk=8)
    terms = prob.residual_terms(c.rt.make_model_fn(m, coefs), {k: v[:2] for k, v in X.items()})
    wmap = {n: w for _, w, n in terms}
    tot = {n: sum(float(p.diagonal(dim1=-2, dim2=-1).sum()) for p in parts) * wmap[n] for n, parts in tb.items()}
    T = sum(tot.values())
    print("traza por termino (estado final de E, semilla 28), cuota del total:")
    for n, v in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"  {n:28s} {v / T:.6f}")
