import unittest
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Run SciML tests with var/agent_runs/sciml-env/Scripts/python.exe')
from src.research_pipeline.sciml.model import RBFPIKAN, derivatives
from src.research_pipeline.sciml.sampling import pools


class BaselineTests(unittest.TestCase):
    def test_sobol_separation_and_reproducibility(self):
        a=pools('burgers',101,64,32); b=pools('burgers',101,64,32)
        np.testing.assert_array_equal(a['interior'],b['interior'])
        self.assertTrue(np.all(a['interior'][:,0]>=-1))
        self.assertTrue(np.all(a['interior'][:,0]<=1))
        self.assertFalse(set(map(tuple,a['interior']))&set(map(tuple,a['diagnostic'])))
        self.assertEqual(len(np.unique(a['interior'][:,0])),64)
        with self.assertRaises(ValueError): pools('heat',101,60,32)

    def test_physical_derivatives_against_analytic(self):
        x=torch.tensor([[.2,.3],[.4,.2]],dtype=torch.float64,requires_grad=True)
        u=torch.sin(torch.pi*x[:,0:1])*torch.exp(-.1*torch.pi**2*x[:,1:2])
        ux,ut,uxx=derivatives(u,x)
        torch.testing.assert_close(ut,-.1*torch.pi**2*u)
        torch.testing.assert_close(uxx,-torch.pi**2*u)

    def test_hard_heat_conditions_and_parameter_budget(self):
        net=RBFPIKAN('heat').double()
        self.assertEqual(sum(p.numel() for p in net.parameters()),3648)
        pts=torch.tensor([[0.,.3],[1.,.5],[.3,0.]],dtype=torch.float64)
        y=net(pts).detach()
        torch.testing.assert_close(y[:2],torch.zeros((2,1),dtype=torch.float64),atol=1e-12,rtol=0)
        torch.testing.assert_close(y[2],torch.sin(torch.pi*pts[2,:1]))

    def test_second_derivatives_backpropagate(self):
        net=RBFPIKAN('burgers').double()
        x=torch.rand(8,2,dtype=torch.float64,requires_grad=True)
        y=net(x); ux,ut,uxx=derivatives(y,x)
        (ut+y*ux-.02*uxx).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_pina_preserves_physical_coordinate_order_and_residual(self):
        from pina import LabelTensor
        from pina.solver import PhysicsInformedSingleModelSolver
        from src.research_pipeline.sciml.train import make_problem,physics
        torch.set_default_dtype(torch.float64)
        for name in ('heat','burgers'):
            model=RBFPIKAN(name).double()
            problem=make_problem(name,pools(name,101,64,32))
            self.assertEqual(problem.input_variables,['x','t'])
            solver=PhysicsInformedSingleModelSolver(problem,model)
            x=LabelTensor(torch.tensor([[.2,.3],[.4,.1]],requires_grad=True),['x','t'])
            y=solver(x); r=physics(name)(x,y)
            z=x.tensor.detach().clone().requires_grad_(True)
            u=model(z); ux,ut,uxx=derivatives(u,z)
            expected=(ut-.1*uxx)/1.1 if name=='heat' else (ut+u*ux-.02*uxx)/.76
            torch.testing.assert_close(y.tensor,u)
            torch.testing.assert_close(r.tensor,expected)


if __name__=='__main__': unittest.main()
