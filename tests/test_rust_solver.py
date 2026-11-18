"""Compare installed Rust and Numba GDA/SGDA, including the CLI boundary."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import lplearn
from lplearn import rust
from lplearn.core import QueueParams, queue_wcmdp, run_gda


class RustSolverTests(unittest.TestCase):
    def test_weighted_iterates_and_traces_match(self):
        for case, capacity, horizon, samples, init, iterations in (
            ("CJS", 2, 3, 0, "uniform", 31),
            ("CJS", 2, 3, 7, "zeros", 31),
            ("CJS", 2, 3, 2, "uniform", 0),
            ("CJS", 5, 10, 3, "uniform", 50),
            ("SJS", 2, 3, 7, "uniform", 31),
        ):
            with self.subTest(case=case, capacity=capacity, samples=samples, iterations=iterations):
                model = queue_wcmdp(QueueParams(case=case, K=capacity, T=horizon))
                params = dict(Gamma_y=0.7, Gamma_lambda=0.2, Gamma_mu=0.3,
                              eta_y=0.04, eta_dual=0.02, n_iter=iterations,
                              samples_per_pair=samples, seed=7, init=init,
                              avg_window=11, record_every=6,
                              state_weights=np.arange(1, model.S + 1, dtype=float))
                expected = run_gda(model, **params)
                actual = rust.run_gda(model, **params)
                for key, value in expected.items():
                    if isinstance(value, np.ndarray):
                        np.testing.assert_allclose(actual[key], value, atol=2e-11, rtol=2e-11,
                                                   err_msg=key)
                self.assertEqual(actual["n_avg"], expected["n_avg"])
                self.assertGreaterEqual(actual["wall_time"], actual["execute_seconds"])

    def test_installed_cli_uses_same_config_for_both_backends(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for case, method in (("CJS", "gda"), ("CJS", "sgda"), ("SJS", "gda"), ("SJS", "sgda")):
                config = root / "policy.toml"
                config.write_text(f'[model]\nK=2\nT=3\ncase="{case}"\n[learning]\n'
                                  'n_iter=20\navg_window=10\nrecord_every=5\n'
                                  'samples_per_pair=3\nseed=7\n')
                results = {}
                for backend in ("numba", "rust"):
                    output = root / f"{case}-{method}-{backend}.npz"
                    proc = subprocess.run([sys.executable, "-I", "-m", "lplearn", "solve",
                                           "--config", str(config), "--method", method,
                                           "--backend", backend, "--output", str(output)],
                                          capture_output=True, text=True, timeout=180)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    with np.load(output, allow_pickle=False) as data:
                        results[backend] = data["occupation"]
                        metadata = json.loads(str(data["metadata_json"]))
                        self.assertEqual(metadata["backend"], backend)
                        if backend == "rust":
                            self.assertEqual(metadata["backend_version"], "0.6.0")
                            self.assertIs(metadata["backend_debug_assertions"], False)
                            self.assertGreaterEqual(metadata["backend_process_seconds"],
                                                    metadata["solver_reported_seconds"])
                np.testing.assert_allclose(results["rust"], results["numba"], rtol=2e-11, atol=2e-11)

    def test_solver_failure_and_timeout_never_fall_back(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "policy.toml"
            config.write_text('[model]\nK=2\nT=3\n[learning]\nn_iter=2\n')
            settings = lplearn.load_config(config)
            with mock.patch("lplearn.rust._executable", return_value="/not/a/lplearn-solver"):
                with self.assertRaises(rust.RustBackendError):
                    lplearn.solve(settings, "sgda", "rust")
            with mock.patch("lplearn.rust.subprocess.run", side_effect=subprocess.TimeoutExpired("solver", 1)):
                with self.assertRaisesRegex(rust.RustBackendError, "timed out"):
                    lplearn.solve(settings, "sgda", "rust", rust_timeout=1)
            with mock.patch("lplearn.rust.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "{}", "")):
                with self.assertRaisesRegex(rust.RustBackendError, "incompatible response"):
                    lplearn.solve(settings, "sgda", "rust")
            output = Path(temp) / "invalid.npz"
            proc = subprocess.run([sys.executable, "-I", "-m", "lplearn", "solve", "--config", str(config),
                                   "--method", "sgda", "--backend", "rust", "--rust-timeout", "0",
                                   "--output", str(output)], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("timeout must be positive and finite", proc.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
