"""Independent operator checks for the approved exploratory verifier extension."""
import unittest
import importlib
from pathlib import Path
import numpy as np
import yaml

from evals.problems.burgers_1d import reference as burgers
from evals.problems.allen_cahn_1d import reference as allen
from evals.problems.dfg_2d import reference as dfg
from evals.graders.exploratory import field_error, residual_score


class BurgersTests(unittest.TestCase):
    def test_traveling_wave_satisfies_pde_with_independent_differences(self):
        for nu in (0.02, 0.005):
            for c in (0., .5):
                t = np.full(31, .3)
                x = c*t - .25 + np.linspace(-2, 2, 31)*(2*nu/.5)
                h = 1e-5
                u = burgers.exact(x, t, nu=nu, c=c)
                ux = (burgers.exact(x+h,t,nu=nu,c=c)-burgers.exact(x-h,t,nu=nu,c=c))/(2*h)
                ut = (burgers.exact(x,t+h,nu=nu,c=c)-burgers.exact(x,t-h,nu=nu,c=c))/(2*h)
                uxx = (burgers.exact(x+h,t,nu=nu,c=c)-2*u+burgers.exact(x-h,t,nu=nu,c=c))/h**2
                np.testing.assert_allclose(burgers.residual(u,ut,ux,uxx,nu), 0, atol=3e-6)

    def test_front_follows_moving_wave(self):
        t = np.linspace(0,1,11)
        self.assertTrue(burgers.critical_mask(.5*t-.25,t,c=.5).all())
        self.assertFalse(burgers.critical_mask(np.ones(11),t,c=.5).any())

    def test_boundary_is_exact_time_dependent_trace(self):
        t=np.array([0.,1.])
        np.testing.assert_array_equal(burgers.boundary(-1,t,c=.5), burgers.exact(-np.ones(2),t,c=.5))
        with self.assertRaises(ValueError): burgers.boundary(0,t)

    def test_invalid_viscosity(self):
        for nu in (0., -1., np.nan):
            with self.assertRaises(ValueError): burgers.exact(np.array([0.]),np.array([0.]),nu=nu)


class AllenTests(unittest.TestCase):
    def test_hard_transform_satisfies_traces(self):
        x=np.linspace(-1,1,51)
        np.testing.assert_allclose(allen.transform(x,np.zeros(51),np.full(51,999.)),allen.initial(x))
        for side in (-1.,1.):
            np.testing.assert_allclose(allen.transform(np.full(51,side),np.linspace(0,1,51),np.full(51,999.)),-1.)

    def test_manufactured_polynomial_operator(self):
        x=np.linspace(-.8,.8,31); t=np.full(31,.3); u=x*x+t
        expected=1.-.002-5*(u-u**3)
        np.testing.assert_allclose(allen.residual(u,np.ones(31),np.full(31,2.)),expected)

    def test_equilibria(self):
        for v in (-1.,0.,1.):
            np.testing.assert_array_equal(allen.residual(np.full(3,v),np.zeros(3),np.zeros(3)),0.)


class DFGTests(unittest.TestCase):
    def test_grid_excludes_solid_and_is_deterministic(self):
        x,y=dfg.eval_grid(81,41)
        self.assertTrue(np.all((x-.2)**2+(y-.2)**2 > .05**2))
        self.assertTrue(np.all((x>0)&(x<2.2)&(y>0)&(y<.41)))
        np.testing.assert_array_equal(x,dfg.eval_grid(81,41)[0])

    def test_channel_poiseuille_solves_operator(self):
        # Manufactured channel field checks operator only, NOT cylinder BC.
        y=np.linspace(0,.41,41); u=dfg.inlet(y); z=np.zeros(41)
        uxx=z; uyy=np.full(41,-8*.3/.41**2)
        result=dfg.residual(u,z,z,4*.3*(.41-2*y)/.41**2,z,z,.001*uyy,z,uxx,uyy,z,z)
        for r in result.values(): np.testing.assert_allclose(r,0.,atol=1e-14)

    def test_outlet_is_laplacian_traction_without_extra_pressure_bc(self):
        a,b=dfg.outlet(np.array([2.]),np.array([3.]),np.array([.002]))
        np.testing.assert_allclose(a,0.); np.testing.assert_allclose(b,.003)

    def test_inlet_mean_and_boundary_geometry(self):
        y=np.linspace(0,.41,1001)
        values=dfg.inlet(y)
        integral=np.sum((values[:-1]+values[1:])*.5*np.diff(y))
        self.assertAlmostEqual(float(integral/.41),.2,places=6)
        boundaries=dfg.boundary_points(64)
        x,y=boundaries['cylinder']
        np.testing.assert_allclose((x-.2)**2+(y-.2)**2,.05**2,atol=1e-16)
        self.assertEqual(len(np.unique(np.stack([x,y],axis=1),axis=0)),64)


class GuardTests(unittest.TestCase):
    def test_reference_pending_cannot_be_bypassed_by_supplied_array(self):
        for problem in ('allen_cahn_1d','dfg_2d'):
            score=field_error(problem,np.ones(3),np.ones(3))
            self.assertIsNone(score.value)
            self.assertEqual(score.reason,'reference_not_verified')

    def test_partial_nan_is_not_filtered_into_success(self):
        score=residual_score(np.array([0.,np.nan]),scale=1.,equation='pde')
        self.assertIsNone(score.value)
        self.assertEqual(score.reason,'non_finite_input')

    def test_field_shape_and_quadrature(self):
        self.assertIsNone(field_error('burgers_1d',np.ones((2,1)),np.ones(2)).value)
        score=field_error('burgers_1d',np.array([2.,1.]),np.ones(2),weights=np.array([1.,3.]))
        self.assertAlmostEqual(score.value,.5)

    def test_invalid_scale_and_unknown_problem(self):
        self.assertIsNone(residual_score(np.ones(2),scale=0.,equation='pde').value)
        self.assertIsNone(field_error('poisson_2d',np.ones(2),np.ones(2)).value)

    def test_bad_grid_sizes(self):
        for module in (burgers,allen,dfg):
            with self.assertRaises(ValueError): module.eval_grid(1,3)

    def test_overflow_must_not_produce_perfect_error(self):
        with np.errstate(over='ignore',invalid='ignore'):
            score=field_error('burgers_1d',np.array([1e200,2.]),np.array([1e200,1.]))
        self.assertIsNone(score.value)


class ContractTests(unittest.TestCase):
    def test_suite_modules_and_locks_match_contracts(self):
        root=Path(__file__).resolve().parents[1]
        suite=yaml.safe_load((root/'suites/exploratory_rbf_signals.yaml').read_text())
        self.assertEqual(suite['problems'],['burgers_1d','allen_cahn_1d','dfg_2d'])
        self.assertEqual(suite['deferred_problems'][0]['id'],'poisson_2d')
        self.assertFalse(suite['budget']['approved'])
        self.assertFalse(suite['emits_scientific_evidence'])
        for name,path in suite['operator_modules'].items():
            module=importlib.import_module(path)
            config=yaml.safe_load((root/'problems'/name/'problem.yaml').read_text())
            self.assertEqual(config['problem']['id'],'problem.'+name)
            self.assertTrue(callable(module.residual))
            self.assertTrue(callable(module.eval_grid))
            if name in suite['field_error_locks']:
                self.assertEqual(config['field_error_status'],'locked_pending_reference')

    def test_dfg_pressure_convention_and_allen_boundary_version(self):
        root=Path(__file__).resolve().parents[1]/'problems'
        d=yaml.safe_load((root/'dfg_2d/problem.yaml').read_text())
        a=yaml.safe_load((root/'allen_cahn_1d/problem.yaml').read_text())
        self.assertEqual(d['boundary_conditions']['additional_pressure_condition'],'none')
        self.assertEqual(a['boundary_conditions']['type'],'dirichlet')
        self.assertEqual(a['pde']['parameters']['d'],.001)


if __name__=='__main__': unittest.main()
