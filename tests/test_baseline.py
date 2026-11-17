"""Exercise the installed single-job API and command."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import lplearn
from lplearn.core import QueueParams, build_lp, queue_wcmdp


class BaselineTests(unittest.TestCase):
    def test_lp_solution_respects_flow_and_resource_constraints(self):
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.toml"
            config_path.write_text('[model]\nK=2\nT=3\ncase="CJS"\n')
            config = lplearn.load_config(config_path)
            y, metadata = lplearn.solve(config, "lp", "cvxpy")
            lp = build_lp(queue_wcmdp(QueueParams(K=2, T=3, case="CJS")))
            self.assertEqual(y.shape, (3, 3, 4))
            self.assertEqual(metadata["backend"], "cvxpy")
            self.assertIn(metadata["status"], ("optimal", "optimal_inaccurate"))
            np.testing.assert_allclose(lp.A @ y.ravel(), lp.b, atol=1e-6)
            self.assertLessEqual(np.max(lp.D @ y.ravel() - lp.r), 1e-6)

    def test_cli_sgda_is_repeatable_and_records_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.toml"
            config_path.write_text('[model]\nK=2\nT=3\ncase="SJS"\n[learning]\n'
                                   'n_iter=20\navg_window=10\nrecord_every=5\n'
                                   'samples_per_pair=3\nseed=7\n')
            arrays = []
            for n in range(2):
                out = root / ("output-0.npz" if n == 0 else "output-1.NPZ")
                proc = subprocess.run([sys.executable, "-m", "lplearn", "solve", "--config", str(config_path),
                                       "--method", "sgda", "--backend", "numba", "--output", str(out)],
                                      text=True, capture_output=True, timeout=180)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn(str(out), proc.stdout)
                self.assertTrue(out.is_file())
                with np.load(out, allow_pickle=False) as saved:
                    arrays.append(saved["occupation"])
                    self.assertEqual(json.loads(str(saved["config_json"]))["model"]["case"], "SJS")
                    self.assertEqual(json.loads(str(saved["metadata_json"]))["seed"], 7)
            np.testing.assert_array_equal(*arrays)
            config = lplearn.load_config(config_path)
            y, _ = lplearn.solve(config, "sgda", "numba")
            np.testing.assert_array_equal(y, arrays[0])

    def test_invalid_config_and_backend_fail_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.toml"
            config.write_text('[model]\nK=2\nT=3\nm0=[0.0, 0.0, 0.0]\n')
            out = root / "output.npz"
            proc = subprocess.run([sys.executable, "-m", "lplearn", "solve", "--config", str(config),
                                   "--method", "lp", "--backend", "cvxpy", "--output", str(out)],
                                  text=True, capture_output=True, timeout=30)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("m0", proc.stderr)
            self.assertFalse(out.exists())
            config.write_text('[model]\nK=2\nT=3\n')
            with self.assertRaisesRegex(ValueError, "unsupported"):
                lplearn.solve(lplearn.load_config(config), "lp", "numba")
            config.write_text('[model]\nK=2\nT=3\n[learning]\nsamples_per_pair=0\n')
            with self.assertRaisesRegex(ValueError, "SGDA requires"):
                lplearn.solve(lplearn.load_config(config), "sgda", "numba")

    def test_nonfinite_unselected_iterate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.toml"
            config.write_text('[model]\nK=2\nT=3\n')
            fake = {"y_avg": np.zeros((3, 3, 4)), "lam_res_last": np.array([float("nan")])}
            with patch("lplearn.workflow.run_gda", return_value=fake):
                with self.assertRaisesRegex(RuntimeError, "non-finite.*lam_res_last"):
                    lplearn.solve(lplearn.load_config(config), "gda", "numba")

    def test_divergent_numba_solve_fails_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, out = root / "config.toml", root / "output.npz"
            config.write_text('[model]\nK=2\nT=3\n[learning]\n'
                              'eta_y=1e308\nn_iter=4\navg_window=4\nrecord_every=1\n')
            proc = subprocess.run([sys.executable, "-m", "lplearn", "solve", "--config", str(config),
                                   "--method", "gda", "--backend", "numba", "--output", str(out)],
                                  text=True, capture_output=True, timeout=180)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("non-finite", proc.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
