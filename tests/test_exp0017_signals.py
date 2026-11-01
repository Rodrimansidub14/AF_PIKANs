"""EXP-0017: señales de la representacion, atribucion y estadisticos (pruebas previas del spec)."""
import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/mechanisms/EXP-0017-representation-battery"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
CAMPAIGN = BL / "burgers-close-20261003/pikan_campaign.py"
REF = BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sig = _load(EXP / "signals.py", "exp0017_signals")
st = _load(EXP / "stats.py", "exp0017_stats")


class IsolationTests(unittest.TestCase):
    def _imports(self, path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
        return mods

    def test_signals_do_not_import_the_verifier_or_any_reference_loader(self):
        mods = self._imports(EXP / "signals.py")
        self.assertLessEqual(mods, {"math", "numpy", "torch"})
        self.assertNotIn("evals", mods)

    def test_stats_only_numpy_and_scipy(self):
        self.assertLessEqual(self._imports(EXP / "stats.py"), {"numpy", "scipy"})

    def test_no_signal_function_takes_a_reference_argument(self):
        import inspect

        for name in ("s1_jacobian_norm", "s2_active_bases", "s3_curvature", "s5_resolution_mismatch", "responsibility", "movement"):
            params = inspect.signature(getattr(sig, name)).parameters
            for p in params:
                self.assertNotIn("ref", p.lower())


class StatsTests(unittest.TestCase):
    def test_spearman_and_overlap(self):
        a = np.arange(1000.0)
        self.assertAlmostEqual(st.spearman(a, a), 1.0)
        self.assertAlmostEqual(st.spearman(a, -a), -1.0)
        self.assertEqual(st.top_overlap(a, a), 1.0)
        self.assertEqual(st.top_overlap(a, a[::-1]), 0.0)
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(st.top_overlap(rng.normal(size=20000), rng.normal(size=20000)), 0.05, delta=0.01)

    def test_sign_flip_detects_a_consistent_positive_shift_and_not_noise(self):
        rng = np.random.default_rng(1)
        self.assertLess(st.sign_flip_pvalue(0.5 + 0.1 * rng.normal(size=25)), 0.001)
        self.assertGreater(st.sign_flip_pvalue(rng.normal(size=25)), 0.05)
        self.assertGreater(st.sign_flip_pvalue(-0.5 + 0.1 * rng.normal(size=25)), 0.99)

    def test_sign_flip_is_reproducible(self):
        d = np.random.default_rng(2).normal(size=10) + 0.2
        self.assertEqual(st.sign_flip_pvalue(d), st.sign_flip_pvalue(d))

    def test_holm(self):
        adj = st.holm([0.01, 0.04, 0.03, 0.5])
        np.testing.assert_allclose(adj, [0.04, 0.09, 0.09, 0.5])
        self.assertTrue((adj >= np.array([0.01, 0.04, 0.03, 0.5])).all())
        self.assertTrue((st.holm([0.2] * 5) == 1.0).all())


class SummaryHelpers(unittest.TestCase):
    def test_top_share(self):
        self.assertAlmostEqual(sig.top_share(np.ones(100), 0.2), 0.2)
        v = np.zeros(100)
        v[:5] = 1.0
        self.assertAlmostEqual(sig.top_share(v, 0.2), 1.0)
        self.assertTrue(np.isnan(sig.top_share(np.zeros(10))))

    def test_movement_is_zero_for_equal_states_scale_invariant_and_localised(self):
        c = torch.randn(4, 3, 5)
        before = {"layers.0.coefficients": c}
        after = {"layers.0.coefficients": c.clone()}
        self.assertEqual(float(sig.movement(before, after)[0].max()), 0.0)
        after["layers.0.coefficients"][:, 1, 2] += 0.5
        d = sig.movement(before, after)[0]
        self.assertEqual(d.shape, (3, 5))
        self.assertEqual(np.unravel_index(np.argmax(d), d.shape), (1, 2))
        self.assertEqual(int((d > 0).sum()), 1)
        d2 = sig.movement({"layers.0.coefficients": c * 7}, {"layers.0.coefficients": after["layers.0.coefficients"] * 7})[0]
        np.testing.assert_allclose(d, d2, rtol=1e-6)


@unittest.skipUnless(CAMPAIGN.exists() and REF.exists(), "faltan el codigo de la campana o la referencia")
class SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.pc = _load(CAMPAIGN, "exp0017_pc")
        cls.prob = cls.pc.make_problem("burgers", str(REF))
        cls.model = cls.prob.build("rbf", 6, 11, 301).to(dtype=torch.float32)
        g = torch.Generator().manual_seed(1)
        cls.xt = torch.stack([2 * torch.rand(300, generator=g) - 1, torch.rand(300, generator=g)], 1)

    def test_shapes_and_ranges(self):
        n = self.xt.shape[0]
        s2 = sig.s2_active_bases(self.model, self.xt)
        self.assertEqual(s2.shape, (n,))
        self.assertTrue(((s2 >= 0) & (s2 <= 2 * 6 * 11)).all())
        s3 = sig.s3_curvature(self.model, self.xt)
        self.assertEqual(s3.shape, (n,))
        self.assertTrue((s3 >= 0).all() and torch.isfinite(s3).all())
        s5, gn, stretch = sig.s5_resolution_mismatch(self.model, self.xt, self.prob.transform)
        self.assertTrue((s5 > 0).all() and torch.isfinite(s5).all() and (stretch > 0).all())
        s1 = sig.s1_jacobian_norm(self.model, self.xt, self.prob.transform, chunk=64)
        self.assertTrue((s1 >= 0).all() and torch.isfinite(s1).all())

    def test_s1_matches_a_brute_force_per_point_gradient(self):
        pts = self.xt[:5]
        s1 = sig.s1_jacobian_norm(self.model, pts, self.prob.transform, chunk=2)
        for i in range(5):
            self.model.zero_grad()
            u = self.prob.transform(pts[i:i + 1], self.model(pts[i:i + 1])).sum()
            u.backward()
            brute = torch.sqrt(sum(p.grad.square().sum() for p in self.model.parameters() if p.grad is not None))
            self.assertAlmostEqual(float(s1[i]), float(brute), delta=1e-4 * max(1.0, float(brute)))

    def test_s3_curvature_matches_autograd_for_the_first_layer(self):
        layer = self.model.layers[0]
        z = torch.tensor([[0.3, -0.6]], requires_grad=True)

        def edges(zz):
            basis = torch.exp(-((zz.unsqueeze(-1) - layer.centers) / layer.width).square())
            return torch.einsum("big,oig->boi", basis, layer.coefficients) + layer.base.unsqueeze(0) * torch.nn.functional.silu(zz).unsqueeze(1)

        total = 0.0
        for o in range(layer.coefficients.shape[0]):
            for i in range(2):
                g1 = torch.autograd.grad(edges(z)[0, o, i], z, create_graph=True)[0][0, i]
                g2 = torch.autograd.grad(g1, z)[0][0, i]
                total += abs(float(g2))
        u = (z.detach().unsqueeze(-1) - layer.centers) / layer.width
        phi_dd = (4.0 * u.square() - 2.0) / layer.width ** 2 * torch.exp(-u.square())
        f_dd = torch.einsum("nig,oig->noi", phi_dd, layer.coefficients) + layer.base.unsqueeze(0) * sig._silu_dd(z.detach()).unsqueeze(1)
        self.assertAlmostEqual(float(f_dd.abs().sum()), total, delta=1e-3 * max(1.0, total))

    def test_responsibility_peaks_at_the_basis_whose_support_holds_the_error(self):
        h0 = sig.layer_inputs(self.model, self.xt)[0]
        centre = float(self.model.layers[0].centers[7])
        w = (h0[:, 0] - centre).abs().lt(0.08).float()                              # el "error" vive en el soporte de la base 7 de x
        self.assertGreater(float(w.sum()), 0)
        A, O = sig.responsibility(self.model, self.xt, w)
        self.assertEqual(int(torch.argmax(A[0][0])), 7)
        self.assertEqual(A[0].shape, (2, 11))
        self.assertEqual(O[1].shape, (6, 11))
        self.assertTrue((O[0] > 0).all())

    def test_responsibility_is_scale_equivariant_in_the_weights(self):
        w = torch.rand(self.xt.shape[0])
        A1, _ = sig.responsibility(self.model, self.xt, w)
        A2, _ = sig.responsibility(self.model, self.xt, 3.0 * w)
        for a, b in zip(A1, A2):
            np.testing.assert_allclose((3.0 * a).numpy(), b.numpy(), rtol=1e-5)

    def test_uniform_weights_give_the_same_responsibility_everywhere(self):
        A, _ = sig.responsibility(self.model, self.xt, torch.ones(self.xt.shape[0]))
        for a in A:
            np.testing.assert_allclose(a.numpy(), 1.0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
