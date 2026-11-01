import unittest

import numpy as np

try:
    import torch
except ImportError:
    raise unittest.SkipTest('Run SciML tests with var/agent_runs/sciml-env/Scripts/python.exe')

from src.sciml.model import build_model, derivatives


class ArchitectureScreenTests(unittest.TestCase):
    def test_screening_models_are_parameter_matched_and_satisfy_heat_boundaries(self):
        expected = {'rbf_fixed': 3648, 'mlp_tanh': 3655, 'bspline_cubic': 3648}
        points = torch.tensor(
            [[0.0, 0.2], [1.0, 0.7], [0.31, 0.0]], dtype=torch.float64
        )
        for model_id, parameter_count in expected.items():
            with self.subTest(model=model_id):
                model = build_model('heat', {'id': model_id}).double()
                self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count)
                output = model(points)
                torch.testing.assert_close(
                    output[:2], torch.zeros((2, 1), dtype=torch.float64), atol=1e-12, rtol=0
                )
                torch.testing.assert_close(
                    output[2], torch.sin(torch.pi * points[2, :1]), atol=1e-12, rtol=0
                )

    def test_each_screening_model_supports_second_derivatives_and_parameter_gradients(self):
        points = torch.tensor(
            [[-0.47, 0.19], [0.11, 0.43], [0.72, 0.86]],
            dtype=torch.float64,
            requires_grad=True,
        )
        for model_id in ('rbf_fixed', 'mlp_tanh', 'bspline_cubic'):
            with self.subTest(model=model_id):
                model = build_model('burgers', {'id': model_id}).double()
                output = model(points)
                ux, ut, uxx = derivatives(output, points)
                residual = ut + output * ux - 0.02 * uxx
                residual.square().mean().backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(ux).all())
                self.assertTrue(torch.isfinite(ut).all())
                self.assertTrue(torch.isfinite(uxx).all())
                self.assertTrue(
                    all(p.grad is not None and torch.isfinite(p.grad).all()
                        for p in model.parameters())
                )

    def test_model_trace_reports_architecture_specific_basis_signals(self):
        points = torch.tensor([[0.2, 0.3], [0.7, 0.8]], dtype=torch.float64)
        expected_family = {'rbf_fixed': 'rbf', 'bspline_cubic': 'bspline', 'mlp_tanh': 'mlp'}
        for model_id, family in expected_family.items():
            with self.subTest(model=model_id):
                model = build_model('heat', {'id': model_id}).double()
                output, layers = model.forward_with_trace(points)
                self.assertEqual(tuple(output.shape), (2, 1))
                self.assertEqual(len(layers), 3)
                self.assertTrue(all(torch.isfinite(item['activation']).all() for item in layers))
                self.assertEqual(model.model_family, family)
                has_basis = all('basis' in item for item in layers)
                self.assertEqual(has_basis, family in ('rbf', 'bspline'))

    def test_canonical_burgers_pina_conditions_use_direct_two_argument_equations(self):
        from pina import LabelTensor
        from src.sciml.sampling import pools
        from src.sciml.train import make_problem

        parameters = {
            'definition': 'raissi_pinnacle_1d',
            'nu': 0.01 / np.pi,
            'field_scale': 1.0,
            'residual_scale': 2.0 + 0.01 / np.pi,
        }
        problem = make_problem('burgers', pools('burgers', 101, 64, 32), parameters)
        cases = (
            ('left', -1.0, 0.5, 0.0),
            ('right', 1.0, 0.5, 0.0),
            ('initial', 0.5, 0.0, 1.0),
        )
        for name, x_value, t_value, expected in cases:
            with self.subTest(condition=name):
                inputs = LabelTensor(torch.tensor([[x_value, t_value]], dtype=torch.float64), ['x', 't'])
                output = LabelTensor(torch.zeros((1, 1), dtype=torch.float64), ['u'])
                residual = problem.conditions[name].equation.residual(inputs, output)
                torch.testing.assert_close(
                    residual.tensor, torch.tensor([[expected]], dtype=torch.float64)
                )


if __name__ == '__main__':
    unittest.main()
