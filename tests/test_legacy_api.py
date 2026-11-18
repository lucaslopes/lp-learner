"""Public API checks for the pinned legacy compatibility implementation."""

import contextlib
import io
from pathlib import Path
import unittest
import warnings

import numpy as np

import lplearn as lpl


class LegacyApiTests(unittest.TestCase):
    def test_all_legacy_exports_resolve_without_overriding_new_api(self):
        expected = {
            "calculate_cost", "instantaneous_cost", "processing_cost", "storage_cost",
            "compute_lagrangian", "solver", "learner", "Model", "compute_U", "compute_V",
            "estimate_P", "compute_P", "compute_Q", "inst_prob", "inst_MDP",
            "simulate_next_state", "acceptance", "highness", "fullness", "kappa",
            "get_updated_params", "verify_aggregation", "compare_results", "METHODS",
            "PARAMETERS_DESCRIPTION", "policy_pie_plot",
        }
        self.assertTrue(expected.issubset(set(lpl.__all__)))
        for name in expected:
            with self.subTest(name=name):
                self.assertIsNotNone(getattr(lpl, name))
        self.assertTrue(callable(lpl.load_config))
        self.assertTrue(callable(lpl.solve))

    def test_fresh_model_regularized_lp_and_legacy_plot(self):
        # Reference: lp_learner-new c23e600, CLARABEL, K=2, T=3, case=1.
        # The old package required importing lp_learner.utils before Model.solve;
        # compatibility initializes it explicitly for a fresh public import.
        model = lpl.Model(K=2, T=3, case=1)
        with contextlib.redirect_stdout(io.StringIO()):
            result = model.solve("cvxpy", verbose=False)["solutions"]["cvxpy"]
        self.assertEqual(result["y"].shape, (3, 2, 2, 3))
        self.assertAlmostEqual(float(result["solution"]), 96.6830158452, places=3)
        np.testing.assert_allclose(
            result["y"][0, :, :, 0],
            [[0.249995979, 0.000000396], [0.500004569, 0.249999056]],
            atol=3e-4, rtol=0,
        )
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        fig, ax = lpl.policy_pie_plot(result["y"].copy(), show_legend=True)
        self.assertEqual(ax.get_xlabel(), "Arrival time (t)")
        self.assertEqual(ax.get_ylabel(), "Queue size (s)")
        self.assertIsNotNone(ax.get_legend())
        plt.close(fig)

    def test_legacy_plot_warns_before_hiding_invalid_occupation(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        good = np.zeros((2, 2, 2, 2))
        good[0, 0, 0, :] = 1
        with warnings.catch_warnings(record=True) as found:
            warnings.simplefilter("always", RuntimeWarning)
            fig, _ = lpl.policy_pie_plot(good.copy())
        self.assertFalse([w for w in found if issubclass(w.category, RuntimeWarning)])
        plt.close(fig)

        bad = good.copy()
        bad[0, 0, 0, 0] = 1.5
        bad[1, 1, 1, 0] = -0.1
        with self.assertWarnsRegex(RuntimeWarning, "negative occupations.*stage mass.*constraints were not checked"):
            fig, _ = lpl.policy_pie_plot(bad)
        self.assertEqual(bad[1, 1, 1, 0], 0.0)  # historical clipping is retained
        plt.close(fig)

        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(nonfinite=value):
                nonfinite = good.copy()
                nonfinite[0, 0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    lpl.policy_pie_plot(nonfinite)

    def test_legacy_learners_match_pinned_reference(self):
        # Reference arrays were generated from the named source commit with
        # Model(K=2, T=3, case=1, epochs=3, window=1), default seed 42.
        for method in ("gda", "sgda"):
            with self.subTest(method=method):
                model = lpl.Model(K=2, T=3, case=1, epochs=3, window=1)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    out = model.solve(method, verbose=False)["solutions"][method]
                actual = np.asarray(out["y"])
                reference = np.load(Path(__file__).parent / "fixtures" / f"legacy-{method}-c23e600.npy")
                np.testing.assert_allclose(actual, reference, atol=2e-11, rtol=2e-11)


if __name__ == "__main__":
    unittest.main()
