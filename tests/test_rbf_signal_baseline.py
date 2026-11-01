import unittest

from src.research_pipeline.sciml.rbf_signal_baseline import (
    ARCHITECTURE_PROFILES,
    LITERATURE_COMPARATORS,
    PROBLEM_SIGNAL_REGISTRY,
    classify_literature_comparability,
    summarize_independent_runs,
)


class LiteratureComparabilityTests(unittest.TestCase):
    def setUp(self):
        self.contract = {
            "equation": "u_t + u*u_x = nu*u_xx",
            "domain": "x=[-1,1], t=[0,1]",
            "initial_condition": "-sin(pi*x)",
            "boundary_condition": "u(-1,t)=u(1,t)=0",
            "physical_parameters": {"nu": 0.01 / 3.141592653589793},
            "primary_metric": "relative_l2",
            "evaluation_protocol": "fixed held-out tensor grid v1",
            "architecture": "fixed RBF-PIKAN",
            "optimizer": "Adam",
            "training_budget": "20000 steps",
            "training_points": "8192 Sobol",
        }

    def test_all_contract_and_protocol_fields_are_required_for_direct_claim(self):
        result = classify_literature_comparability(self.contract, self.contract)
        self.assertEqual(result["status"], "directly_comparable")

    def test_same_pde_with_different_budget_is_context_only(self):
        literature = dict(self.contract, architecture="MLP", training_budget="50000 steps")
        result = classify_literature_comparability(self.contract, literature)
        self.assertEqual(result["status"], "same_pde_different_protocol")
        self.assertIn("architecture", result["mismatched_fields"])
        self.assertIn("training_budget", result["mismatched_fields"])

    def test_different_boundary_or_missing_contract_is_not_directly_comparable(self):
        periodic = dict(self.contract, boundary_condition="periodic")
        self.assertEqual(
            classify_literature_comparability(self.contract, periodic)["status"],
            "different_problem_contract",
        )
        incomplete = dict(self.contract)
        del incomplete["evaluation_protocol"]
        self.assertEqual(
            classify_literature_comparability(self.contract, incomplete)["status"],
            "contract_unverified",
        )


class SignalBaselineDesignTests(unittest.TestCase):
    def test_signals_are_specific_to_each_pde_family(self):
        self.assertIn("modal_decay", PROBLEM_SIGNAL_REGISTRY["heat"]["problem_specific"])
        self.assertIn("front_position_and_width", PROBLEM_SIGNAL_REGISTRY["burgers"]["problem_specific"])
        self.assertIn("interface_and_energy_dissipation", PROBLEM_SIGNAL_REGISTRY["allen_cahn"]["problem_specific"])
        self.assertIn("drag_lift_pressure_drop", PROBLEM_SIGNAL_REGISTRY["dfg_ns"]["problem_specific"])

    def test_rbf_interpretability_is_a_testable_hypothesis_and_bspline_is_a_basis_control(self):
        rbf = ARCHITECTURE_PROFILES["rbf_fixed"]
        spline = ARCHITECTURE_PROFILES["bspline_cubic"]
        mlp = ARCHITECTURE_PROFILES["mlp_tanh"]
        self.assertTrue(rbf["explicit_basis"])
        self.assertTrue(rbf["fixed_centers_and_widths"])
        self.assertTrue(spline["explicit_basis"])
        self.assertFalse(mlp["explicit_basis"])
        self.assertEqual(rbf["interpretability_claim_status"], "hypothesis_to_measure")
        self.assertIn("no está establecida", rbf["claim_limit"].lower())

    def test_frontier_score_is_recorded_with_local_comparability_warning(self):
        burgers = next(
            entry for entry in LITERATURE_COMPARATORS
            if entry["problem"] == "Burgers 1D" and entry["method"] == "vRBA"
        )
        self.assertIn("8.25e-9", burgers["published_primary_score"])
        self.assertEqual(burgers["contract_status"], "same_pde_different_protocol")
        self.assertIn("not_head_to_head", burgers["local_result_comparison"])

    def test_run_summary_keeps_seed_count_as_the_independent_unit(self):
        summary = summarize_independent_runs([1.0, 2.0, 3.0])
        self.assertEqual(summary["n_runs"], 3)
        self.assertAlmostEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["sample_std"], 1.0)


if __name__ == "__main__":
    unittest.main()
