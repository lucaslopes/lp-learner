"""Recovery must preserve the input and produce a valid relaxed occupation."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class RecoveryTests(unittest.TestCase):

    def test_conditional_policy_rejects_invalid_occupation_before_clipping(self):
        from lplearn.core.policy import conditional_policy

        for y in (np.array([[[-np.inf, 1.0]]]), np.empty((1, 1, 0))):
            with self.subTest(shape=y.shape):
                with self.assertRaisesRegex(ValueError, "occupation"):
                    conditional_policy(y)

    def test_fluid_policy_evaluation_validates_the_initial_distribution(self):
        from lplearn.core import QueueParams, queue_wcmdp
        from lplearn.core.policy import fluid_evaluation

        model = queue_wcmdp(QueueParams(K=2, T=3))
        policy = np.full((model.T, model.S, model.U), 1.0 / model.U)
        for initial in (1.0 / model.S, [1.0], np.ones(model.S)):
            with self.subTest(initial=initial):
                with self.assertRaisesRegex(ValueError, "initial distribution"):
                    fluid_evaluation(model, policy, initial)
        result = fluid_evaluation(model, policy)
        np.testing.assert_allclose(result["mass"].sum(axis=1), 1.0)
    def test_existing_sidecar_is_not_overwritten(self):
        from lplearn.plotting import plot_command

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, output = root / "config.toml", root / "plot.png"
            config.write_text('[model]\nK=1\nT=2\n')
            sidecar = root / "plot.policy.npy"
            sidecar.write_bytes(b"existing")
            args = SimpleNamespace(config=config, output=output, format=None,
                                   method="lp", backend="cvxpy", rust_timeout=30.0,
                                   allow_infeasible=False, recover=False, open=False)
            with self.assertRaises(FileExistsError):
                plot_command(args)
            self.assertEqual(sidecar.read_bytes(), b"existing")
            self.assertFalse(output.exists())

    def test_failed_publication_removes_its_outputs(self):
        import os
        from lplearn.plotting import _publish_artifacts

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stages = [root / "staged-a", root / "staged-b"]
            targets = [root / "a", root / "b"]
            for item in stages:
                item.write_bytes(b"content")
            real_link = os.link
            calls = 0

            def fail_second(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected write failure")
                return real_link(source, target)

            with patch("lplearn.plotting.os.link", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "injected"):
                    _publish_artifacts(list(zip(stages, targets)), targets)
            self.assertFalse(any(path.exists() for path in targets))

    def test_manifest_is_published_last_and_reserved_stem_rejected(self):
        import os
        from lplearn.plotting import _publish_artifacts, plot_command

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stages = [root / "stage-plot", root / "stage-json"]
            targets = [root / "result.png", root / "result.json"]
            for item in stages:
                item.write_bytes(b"content")
            real_link = os.link

            def check_manifest(source, target):
                self.assertFalse(targets[-1].exists())
                return real_link(source, target)

            with patch("lplearn.plotting.os.link", side_effect=check_manifest):
                _publish_artifacts(list(zip(stages, targets)), targets)
            self.assertTrue(all(path.exists() for path in targets))
            self.assertFalse((root / ".result.lplearn.lock").exists())
            config = root / "config.toml"
            config.write_text('[model]\nK=1\nT=2\n')
            args = SimpleNamespace(config=config, output=root / "result.POLICY.png", format=None,
                                   method="lp", backend="cvxpy", rust_timeout=30.0,
                                   allow_infeasible=False, recover=False, open=False)
            with self.assertRaisesRegex(ValueError, "reserved"):
                plot_command(args)

    def test_rollback_does_not_remove_replaced_output(self):
        import os
        from lplearn.plotting import _publish_artifacts

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stages = [root / "stage-a", root / "stage-b"]
            targets = [root / "a.png", root / "a.json"]
            for path in stages:
                path.write_bytes(b"our output")
            replacement = root / "replacement"
            replacement.write_bytes(b"another writer")
            real_link = os.link
            calls = 0

            def replace_then_fail(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    os.replace(replacement, targets[0])
                    raise OSError("injected publication failure")
                return real_link(source, target)

            with patch("lplearn.plotting.os.link", side_effect=replace_then_fail):
                with self.assertRaisesRegex(OSError, "injected"):
                    _publish_artifacts(list(zip(stages, targets)), targets)
            self.assertEqual(targets[0].read_bytes(), b"another writer")
            self.assertFalse(targets[1].exists())

    def test_diffuse_feasible_mass_is_visible(self):
        from lplearn.plotting import policy_pie_plot
        from matplotlib import pyplot as plt

        occupation = np.zeros((1, 101, 4))
        occupation[0, :, 0] = 1 / 101
        fig, ax = policy_pie_plot(occupation)
        try:
            self.assertEqual(sum(p.theta2 > p.theta1 for p in ax.patches), 101)
        finally:
            plt.close(fig)

    def test_projection_corrects_invalid_occupation_without_mutation(self):
        from lplearn.core import QueueParams, queue_wcmdp
        from lplearn.workflow import occupation_feasibility, recover_occupation

        model = queue_wcmdp(QueueParams(K=2, T=3))
        original = np.full((3, 3, 4), -0.5)
        saved = original.copy()
        recovered, report = recover_occupation(original, model)
        np.testing.assert_array_equal(original, saved)
        self.assertFalse(report["original_feasibility"]["feasible"])
        self.assertTrue(occupation_feasibility(recovered, model)["feasible"])
        self.assertGreater(report["distance_l2"], 0)
        with self.assertRaises(ValueError):
            recover_occupation(np.full((3, 3, 4), np.nan), model)

    def test_cli_recovery_preserves_iterate_and_exports_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.toml"
            config.write_text('[model]\nK=2\nT=3\n[learning]\nn_iter=20\navg_window=10\nrecord_every=5\n')
            values = {}
            for backend in ("numba", "rust"):
                output = root / f"{backend}.png"
                proc = subprocess.run([sys.executable, "-m", "lplearn", "plot-policy",
                                       "--config", str(config), "--method", "sgda", "--backend", backend,
                                       "--output", str(output), "--recover"],
                                      capture_output=True, text=True, timeout=180)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                manifest = json.loads(output.with_suffix(".json").read_text())
                self.assertEqual(manifest["plot_kind"], "recovered_occupation")
                recovery = manifest["recovery"]
                self.assertFalse(recovery["original_feasibility"]["feasible"])
                self.assertTrue(recovery["recovered_feasibility"]["feasible"])
                self.assertTrue(recovery["conditional_policy_feasibility"]["feasible"])
                original = np.load(output.with_suffix(".npy"), allow_pickle=False)
                recovered = np.load(root / f"{backend}.recovered.npy", allow_pickle=False)
                policy = np.load(root / f"{backend}.policy.npy", allow_pickle=False)
                self.assertGreater(np.linalg.norm(original - recovered), 0)
                np.testing.assert_allclose(policy.sum(axis=2), 1.0)
                values[backend] = recovered
            np.testing.assert_allclose(values["numba"], values["rust"], atol=2e-7, rtol=2e-7)


if __name__ == "__main__":
    unittest.main()
