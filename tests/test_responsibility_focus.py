"""EXP-0018: pruebas previas (Tier 0/1, CPU) del foco por responsabilidad."""
import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

from src.mechanisms import responsibility_focus as rf

ROOT = Path(__file__).resolve().parents[1]
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
CAMPAIGN = BL / "burgers-close-20261003/pikan_campaign.py"
REF = BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz"
SWEEP = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff/sweep.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PureFunctionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.A = [rng.uniform(0.1, 5.0, size=(2, 11)), rng.uniform(0.1, 5.0, size=(6, 11)), rng.uniform(0.1, 5.0, size=(6, 11))]

    def test_scales_use_the_global_median_and_are_clipped_and_monotone(self):
        s = rf.scales_from_responsibility(self.A, beta=1.0)
        med = float(np.median(np.concatenate([a.reshape(-1) for a in self.A])))
        for a, si in zip(self.A, s):
            np.testing.assert_allclose(si, np.clip(np.sqrt(a / med), 0.1, 10.0))
            self.assertGreaterEqual(si.min(), 0.1)
            self.assertLessEqual(si.max(), 10.0)
        flat_a, flat_s = np.concatenate([a.reshape(-1) for a in self.A]), np.concatenate([x.reshape(-1) for x in s])
        order = np.argsort(flat_a)
        self.assertTrue((np.diff(flat_s[order]) >= -1e-12).all())

    def test_uniform_responsibility_gives_unit_scales_and_beta_zero_too(self):
        for s in rf.scales_from_responsibility([np.full((2, 11), 3.0), np.full((6, 11), 3.0)]):
            np.testing.assert_allclose(s, 1.0)
        for s in rf.scales_from_responsibility(self.A, beta=0.0):
            np.testing.assert_allclose(s, 1.0)

    def test_extreme_values_hit_the_clip(self):
        A = [np.array([[1e-9, 1.0, 1e9]])]
        s = rf.scales_from_responsibility(A)[0]
        self.assertEqual(float(s[0, 0]), 0.1)
        self.assertEqual(float(s[0, 2]), 10.0)

    def test_non_positive_median_is_rejected(self):
        with self.assertRaises(ValueError):
            rf.scales_from_responsibility([np.zeros((2, 3))])

    def test_permutation_keeps_each_layers_multiset_and_is_reproducible(self):
        s = rf.scales_from_responsibility(self.A)
        p1, p2 = rf.permute_within_layer(s, seed=3), rf.permute_within_layer(s, seed=3)
        for a, b, c in zip(s, p1, p2):
            np.testing.assert_array_equal(np.sort(a.reshape(-1)), np.sort(b.reshape(-1)))
            self.assertEqual(a.shape, b.shape)
            np.testing.assert_array_equal(b, c)
            self.assertFalse(np.array_equal(a, b))
        other = rf.permute_within_layer(s, seed=4)
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(p1, other)))

    def test_top_fraction_masks_pick_exactly_the_largest_bases_globally(self):
        masks = rf.top_fraction_masks(self.A, q=0.5)
        flat = np.concatenate([a.reshape(-1) for a in self.A])
        mflat = np.concatenate([m.reshape(-1) for m in masks])
        self.assertEqual(int(mflat.sum()), int(np.ceil(0.5 * flat.size)))
        self.assertGreaterEqual(flat[mflat].min(), flat[~mflat].max())
        self.assertEqual([m.shape for m in masks], [a.shape for a in self.A])


class IsolationTests(unittest.TestCase):
    def test_module_imports_only_numpy_torch_math(self):
        tree = ast.parse(Path(rf.__file__).read_text(encoding="utf-8"))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
        self.assertLessEqual(mods, {"math", "numpy", "torch"})
        self.assertNotIn("evals", mods)

    def test_no_function_takes_a_reference_argument(self):
        import inspect

        for name in ("scales_from_responsibility", "permute_within_layer", "top_fraction_masks", "apply_scales", "apply_freeze", "release"):
            for p in inspect.signature(getattr(rf, name)).parameters:
                self.assertNotIn("ref", p.lower())


@unittest.skipUnless(CAMPAIGN.exists() and REF.exists() and SWEEP.exists(), "faltan la campana, la referencia o sweep.py")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.pc = _load(CAMPAIGN, "exp0018_pc")
        cls.sw = _load(SWEEP, "exp0018_sweep")
        cls.prob = cls.pc.make_problem("burgers", str(REF))
        cls.prob.n_col = 256
        cls.P = cls.prob.pools(301, torch.device("cpu"), torch.float32)

    def model(self):
        m = self.prob.build("rbf", 6, 11, 301).to(dtype=torch.float32)
        # unas cuantas iteraciones de Adam para un estado no trivial
        self.sw.adam_stage(self.prob, m, self.P, steps=30)
        return m

    def _scales(self, m, seed=0):
        rng = np.random.default_rng(seed)
        return [rng.uniform(0.2, 4.0, size=(layer.coefficients.shape[1], layer.coefficients.shape[2])) for layer in m.layers]

    def test_unit_scales_do_not_change_the_forward_pass(self):
        m = self.model()
        x = self.P["residual"][:50]
        before = m(x).detach().clone()
        rf.apply_scales(m, [np.ones((l.coefficients.shape[1], l.coefficients.shape[2])) for l in m.layers])
        self.assertTrue(torch.equal(m(x).detach(), before))

    def test_scaled_model_has_the_same_function_and_the_gradient_is_s_times_the_original(self):
        m = self.model()
        ref = self.model()
        x = self.P["residual"][:64]
        s = self._scales(m)
        ref.zero_grad(set_to_none=True)
        m.zero_grad(set_to_none=True)
        plain_loss = self.prob.loss(ref, self.P)[0]
        plain_loss.backward()
        g_plain = [l.coefficients.grad.clone() for l in ref.layers]
        rf.apply_scales(m, s)
        self.assertTrue(torch.allclose(m(x), ref(x), rtol=1e-5, atol=1e-6))              # misma funcion al aplicar la reparametrizacion
        loss = self.prob.loss(m, self.P)[0]
        loss.backward()
        for l, layer in enumerate(m.layers):
            g_new = layer.coefficients.grad
            expected = g_plain[l] * torch.as_tensor(s[l], dtype=torch.float32).unsqueeze(0)
            self.assertTrue(torch.allclose(g_new, expected, rtol=1e-4, atol=1e-8), f"capa {l}")

    def test_unit_scales_reproduce_plain_lbfgs_bit_for_bit(self):
        a, b = self.model(), self.model()
        la = self.sw.lbfgs_stage(self.prob, a, self.P, seconds_cap=1e9, max_iter=30, chunk=10)
        rf.apply_scales(b, [np.ones((l.coefficients.shape[1], l.coefficients.shape[2])) for l in b.layers])
        lb = self.sw.lbfgs_stage(self.prob, b, self.P, seconds_cap=1e9, max_iter=30, chunk=10)
        rf.release(b)
        self.assertEqual(la["iterations"], lb["iterations"])
        for k, v in a.state_dict().items():
            self.assertTrue(torch.equal(v, b.state_dict()[k]), k)

    def test_non_unit_scales_change_the_endpoint(self):
        a, b = self.model(), self.model()
        self.sw.lbfgs_stage(self.prob, a, self.P, seconds_cap=1e9, max_iter=30, chunk=10)
        rf.apply_scales(b, self._scales(b))
        self.sw.lbfgs_stage(self.prob, b, self.P, seconds_cap=1e9, max_iter=30, chunk=10)
        rf.release(b)
        self.assertFalse(torch.equal(a.layers[1].coefficients, b.layers[1].coefficients))

    def test_frozen_bases_do_not_move_and_the_others_do(self):
        m = self.model()
        start = [l.coefficients.detach().clone() for l in m.layers]
        rng = np.random.default_rng(5)
        masks = [rng.random((l.coefficients.shape[1], l.coefficients.shape[2])) < 0.5 for l in m.layers]
        rf.apply_freeze(m, masks)
        self.sw.lbfgs_stage(self.prob, m, self.P, seconds_cap=1e9, max_iter=30, chunk=10)
        rf.release(m)
        moved_any = False
        for l, layer in enumerate(m.layers):
            frozen = torch.as_tensor(~masks[l]).unsqueeze(0).expand_as(start[l])
            self.assertTrue(torch.equal(layer.coefficients.detach()[frozen], start[l][frozen]), f"capa {l}: una base congelada se movio")
            free = ~frozen
            moved_any = moved_any or not torch.equal(layer.coefficients.detach()[free], start[l][free])
        self.assertTrue(moved_any)

    def test_release_returns_a_plain_model_with_the_original_coefficients(self):
        m = self.model()
        original = [l.coefficients.detach().clone() for l in m.layers]
        rf.apply_scales(m, self._scales(m))
        rf.release(m)
        for l, layer in enumerate(m.layers):
            self.assertFalse(hasattr(layer, "_rf_scale"))
            self.assertNotIn("forward", layer.__dict__)
            self.assertTrue(torch.allclose(layer.coefficients.detach(), original[l], rtol=1e-6, atol=0))   # c / s * s: solo redondeo
        self.assertTrue(all(isinstance(p, torch.nn.Parameter) for p in m.parameters()))
        self.assertEqual(sum(1 for _ in m.state_dict()), sum(1 for _ in self.model().state_dict()))

    def test_scale_and_mask_tensors_follow_the_device_of_the_coefficients(self):
        import inspect

        src = inspect.getsource(rf.apply_scales) + inspect.getsource(rf.apply_freeze)
        self.assertEqual(src.count("device=layer.coefficients.device"), 2)   # en GPU, un tensor creado en CPU rompe div_/where (fallo real de la primera prueba en T4)
        m = self.model()
        rf.apply_scales(m, self._scales(m))
        rf.apply_freeze(m, [np.ones((l.coefficients.shape[1], l.coefficients.shape[2]), dtype=bool) for l in m.layers])
        for layer in m.layers:
            self.assertEqual(layer._rf_scale.device, layer.coefficients.device)
            self.assertEqual(layer._rf_mask.device, layer.coefficients.device)


if __name__ == "__main__":
    unittest.main()
