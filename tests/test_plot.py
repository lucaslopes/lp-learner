"""Installed CLI plot workflow, numerical parity and error handling."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class PlotTests(unittest.TestCase):
    def test_cli_saves_plots_and_matched_rust_numba_occupations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.toml"
            config.write_text('[model]\nK=2\nT=3\ncase="CJS"\n[learning]\n'
                              'n_iter=20\navg_window=10\nrecord_every=5\n'
                              'samples_per_pair=3\nseed=7\n')
            arrays = {}
            for method, backend in (("lp", "cvxpy"), ("sgda", "numba"), ("sgda", "rust")):
                output = root / f"{backend}.png"
                proc = subprocess.run([sys.executable, "-m", "lplearn", "plot-policy",
                                       "--config", str(config), "--method", method,
                                       "--backend", backend, "--output", str(output), "--allow-infeasible"],
                                      text=True, capture_output=True, timeout=180)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                reported = next(line.removeprefix("Policy plot: ")
                                for line in proc.stdout.splitlines()
                                if line.startswith("Policy plot: "))
                self.assertTrue(Path(reported).samefile(output))
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                manifest = json.loads(output.with_suffix(".json").read_text())
                array_path = output.with_suffix(".npy")
                arrays[backend] = np.load(array_path, allow_pickle=False)
                self.assertEqual(arrays[backend].shape, (3, 3, 4))
                self.assertEqual(manifest["backend"], backend)
                self.assertEqual(manifest["occupation_feasibility"]["feasible"], method == "lp")
                self.assertEqual(manifest["plot_kind"], "occupation" if method == "lp" else "infeasible_iterate_diagnostic")
                self.assertEqual(manifest["config_sha256"], hashlib.sha256(config.read_bytes()).hexdigest())
                self.assertEqual(manifest["occupation_sha256"], hashlib.sha256(array_path.read_bytes()).hexdigest())
                self.assertEqual(manifest["plot_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            np.testing.assert_allclose(arrays["numba"], arrays["rust"], atol=2e-11, rtol=2e-11)

    def test_infeasible_iterate_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, output = root / "config.toml", root / "plot.png"
            config.write_text('[model]\nK=2\nT=3\n[learning]\nn_iter=20\navg_window=10\nrecord_every=5\n')
            proc = subprocess.run([sys.executable, "-m", "lplearn", "plot-policy", "--config", str(config),
                                   "--method", "sgda", "--backend", "rust", "--output", str(output)],
                                  text=True, capture_output=True, timeout=30)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("infeasible", proc.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".npy").exists())
            self.assertFalse(output.with_suffix(".json").exists())

    def test_model_flow_violation_is_not_hidden_by_valid_mass(self):
        from lplearn.core import QueueParams, queue_wcmdp
        from lplearn.workflow import occupation_feasibility
        from lplearn.plotting import policy_pie_plot
        from matplotlib import pyplot as plt

        model = queue_wcmdp(QueueParams(K=1, T=2))
        y = np.zeros((2, 2, 4))
        y[:, 0, 0] = 1.0
        self.assertTrue(occupation_feasibility(y, model)["feasible"])
        y[1, 0, 0], y[1, 1, 0] = 0.0, 1.0
        report = occupation_feasibility(y, model)
        self.assertFalse(report["feasible"])
        self.assertEqual(report["stage_mass_residual"], 0.0)
        self.assertGreater(report["flow_residual"], 0.9)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            policy_pie_plot(y, feasibility=report)
        fig, ax = policy_pie_plot(y, feasibility=report, allow_infeasible=True)
        try:
            self.assertIn("INFEASIBLE ITERATE", ax.get_title())
        finally:
            plt.close(fig)

    def test_rejection_message_includes_initial_and_stage_mass_residuals(self):
        from lplearn.core import QueueParams, queue_wcmdp
        from lplearn.plotting import plot_command
        from lplearn.workflow import occupation_feasibility

        model = queue_wcmdp(QueueParams(K=1, T=2))
        occupation = np.zeros((model.T, model.S, model.U))
        feasibility = occupation_feasibility(occupation, model)
        self.assertEqual(feasibility["initial_residual"], 1.0)
        self.assertEqual(feasibility["stage_mass_residual"], 1.0)
        self.assertEqual(feasibility["flow_residual"], 0.0)
        self.assertEqual(feasibility["resource_violation"], 0.0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, output = root / "config.toml", root / "plot.png"
            config.write_text('[model]\nK=1\nT=2\n')
            args = SimpleNamespace(config=config, format=None, output=output,
                                   method="gda", backend="numba", rust_timeout=30.0,
                                   allow_infeasible=False, open=False)
            with patch("lplearn.plotting.solve", return_value=(occupation, {"occupation_feasibility": feasibility})):
                with self.assertRaisesRegex(ValueError, r"initial/flow/mass/resource violation 1\)"):
                    plot_command(args)
            self.assertFalse(output.exists())

    def test_invalid_format_fails_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, output = root / "config.toml", root / "plot.png"
            config.write_text('[model]\nK=2\nT=3\n')
            proc = subprocess.run([sys.executable, "-m", "lplearn", "plot-policy", "--config", str(config),
                                   "--method", "lp", "--backend", "cvxpy", "--output", str(output),
                                   "--format", "svg"], text=True, capture_output=True, timeout=30)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("extension", proc.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
