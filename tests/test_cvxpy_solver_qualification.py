"""Qualify CVXPY solver results on representative queue fixtures."""

import unittest

import numpy as np

from lplearn.core import QueueParams, build_lp, conditional_policy, queue_wcmdp, solve_lp
from lplearn.workflow import occupation_feasibility


class CVXPYSolverQualificationTests(unittest.TestCase):
    def test_scipy_and_clarabel_find_feasible_near_equal_cost_solutions(self):
        fixtures = (
            (2, 3, "CJS"),
            (2, 3, "SJS"),
            (3, 4, "CJS"),
            (3, 4, "SJS"),
        )
        for k, horizon, case in fixtures:
            with self.subTest(K=k, T=horizon, case=case):
                model = queue_wcmdp(QueueParams(K=k, T=horizon, case=case))
                lp = build_lp(model)
                clarabel = solve_lp(lp, Gamma_y=0.0, solver="CLARABEL")
                scipy = solve_lp(lp, Gamma_y=0.0, solver="SCIPY")

                for solver_name, result in (("CLARABEL", clarabel), ("SCIPY", scipy)):
                    self.assertIn(result["status"], ("optimal", "optimal_inaccurate"))
                    self.assertEqual(result["solver"], solver_name)
                    feasibility = occupation_feasibility(result["y"], model)
                    self.assertTrue(feasibility["feasible"], feasibility)

                # Degenerate optima can induce visibly different policies. The
                # qualification gate is feasibility and objective agreement,
                # not elementwise equality of the selected policy.
                objective_delta = abs(clarabel["objective"] - scipy["objective"])
                self.assertLessEqual(
                    objective_delta,
                    1e-5,
                    msg=(f"K={k}, T={horizon}, case={case}: objective delta "
                         f"{objective_delta:.6g}; CLARABEL="
                         f"{clarabel['objective']:.12g}, SCIPY="
                         f"{scipy['objective']:.12g}"),
                )

                pi_clarabel = conditional_policy(clarabel["y"])
                pi_scipy = conditional_policy(scipy["y"])
                np.testing.assert_allclose(pi_clarabel.sum(axis=2), 1.0, atol=1e-12)
                np.testing.assert_allclose(pi_scipy.sum(axis=2), 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
