"""Checks for benchmark gates that must prevent misleading comparisons."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import importlib.util

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).parents[1] / "scripts" / "benchmark_policy.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkGateTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform in {"darwin", "linux", "win32"}, "CI platforms")
    def test_hardware_identity_records_usable_physical_memory(self):
        identity = benchmark.hardware_identity()
        self.assertIsInstance(identity["memory_bytes"], int)
        self.assertGreater(identity["memory_bytes"], 0)

    def test_windows_timeout_refuses_to_claim_descendants_stopped(self):
        worker = Mock()
        with self.assertRaisesRegex(benchmark.UncertainWorkerTreeError, "descendant"):
            benchmark.stop_timed_out_worker(worker, is_windows=True)
        worker.kill.assert_called_once_with()
        worker.communicate.assert_not_called()

    def test_uncertain_worker_tree_stops_run_without_committing_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            config = root / "config.toml"
            config.write_text("a = 1\n")
            output = root / "run"
            argv = ["benchmark", "--command", shlex.join([sys.executable, "-c", "pass"]),
                    "--cwd", str(source), "--config", str(config), "--output-dir", str(output),
                    "--environment-python", sys.executable, "--case", "n:sgda:numba",
                    "--case", "r:sgda:rust", "--numba-cache-state", "per-process-empty",
                    "--repetitions", "1"]
            def uncertain(*args, **kwargs):
                args[8](12345)  # Record that the CLI started, then lose proof about its descendants.
                raise benchmark.UncertainWorkerTreeError("descendant may remain")
            with patch.object(sys, "argv", argv), patch.object(benchmark, "run_one", side_effect=uncertain) as run:
                self.assertEqual(benchmark.main(), 1)
            self.assertEqual(run.call_count, 1)
            attempts = list((output / "attempts").iterdir())
            self.assertEqual(len(attempts), 1)
            self.assertEqual(json.loads((attempts[0] / ".worker.json").read_text())["status"], "started")
            self.assertEqual(list((output / "shards").glob("*.json")), [])
            self.assertEqual(json.loads((output / "progress.json").read_text())["status"], "interrupted")
            self.assertFalse((output / "terminal.json").exists())
    def test_recovery_is_requested_only_for_learning_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "c.toml"
            config.write_text("x=1\n")
            for method, backend, expected in (("lp", "cvxpy", False), ("sgda", "numba", True), ("gda", "rust", True)):
                case = {"name": method, "method": method, "backend": backend}
                result = benchmark.run_one([sys.executable, "-c", "pass"], root, config, case, 0, root, 5, {},
                                           recover_learning=True)
                self.assertEqual(result["recovery_requested"], expected)
                self.assertEqual("--recover" in result["command"], expected)

    def test_recovery_verifier_checks_hashes_policy_and_constraints(self):
        from lplearn.core import QueueParams, queue_wcmdp
        from lplearn.core.policy import conditional_policy, fluid_evaluation
        from lplearn.workflow import occupation_feasibility
        from dataclasses import asdict

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "plot.png"
            fields = asdict(QueueParams(K=1, T=2))
            model = queue_wcmdp(QueueParams(**fields))
            pi = np.zeros((2, 2, 4)); pi[:, :, 0] = 1
            recovered = fluid_evaluation(model, pi)["y"]
            policy = conditional_policy(recovered)
            original = recovered - 0.1
            recovered_path = root / "plot.recovered.npy"
            policy_path = root / "plot.policy.npy"
            np.save(recovered_path, recovered); np.save(policy_path, policy)
            raw = occupation_feasibility(original, model)
            rcheck = occupation_feasibility(recovered, model)
            pcheck = occupation_feasibility(fluid_evaluation(model, policy)["y"], model)
            report = {"method": "euclidean_projection_lp_relaxation", "solver": "CLARABEL", "status": "optimal",
                      "seconds": 0.01, "distance_l2": float(np.linalg.norm(recovered - original)),
                      "occupation_path": str(recovered_path), "occupation_sha256": benchmark.digest(recovered_path),
                      "conditional_policy_path": str(policy_path), "conditional_policy_sha256": benchmark.digest(policy_path),
                      "original_feasibility": raw, "recovered_feasibility": rcheck, "conditional_policy_feasibility": pcheck}
            manifest = {"config": {"model": fields}, "occupation_feasibility": raw, "recovery": report}
            self.assertEqual(benchmark.verify_recovery(manifest, original, output), (recovered_path, policy_path))
            np.save(policy_path, np.roll(policy, 1, axis=2))
            with self.assertRaisesRegex(ValueError, "path or hash"):
                benchmark.verify_recovery(manifest, original, output)
            report["conditional_policy_sha256"] = benchmark.digest(policy_path)
            with self.assertRaisesRegex(ValueError, "does not correspond"):
                benchmark.verify_recovery(manifest, original, output)
            np.save(policy_path, policy); report["conditional_policy_sha256"] = benchmark.digest(policy_path)
            np.save(recovered_path, recovered - 0.1); report["occupation_sha256"] = benchmark.digest(recovered_path)
            with self.assertRaises(ValueError):
                benchmark.verify_recovery(manifest, original, output)

    def test_recovery_parity_required_before_ratio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, a, b, policy = (root / name for name in ("raw.npy", "a.npy", "b.npy", "policy.npy"))
            np.save(raw, np.ones((1, 1, 1))); np.save(a, np.ones((1, 1, 1)))
            np.save(b, np.full((1, 1, 1), 1.001)); np.save(policy, np.ones((1, 1, 1)))
            cases = [{"name": "n", "method": "sgda", "backend": "numba"},
                     {"name": "r", "method": "sgda", "backend": "rust"}]
            records = []
            for rep in range(5):
                for case in cases:
                    manifest = {"config": {}, "seed": 0, "iterations": 10, "samples_per_pair": 1,
                                "averaged_iterations": 10, "occupation_feasibility": {"feasible": False},
                                "solve_seconds": 1.0, "plot_seconds": 1.0, "plot_kind": "recovered_occupation",
                                "recovery": {"method": "euclidean_projection_lp_relaxation", "solver": "CLARABEL", "seconds": 0.1}}
                    records.append({"case": case, "repetition": rep, "artifact_verified": True,
                                    "array": str(raw), "wall_seconds": 1.0, "manifest": manifest,
                                    "recovery_requested": True, "recovered_array": str(a if case["backend"] == "numba" else b),
                                    "policy_array": str(policy)})
            pair = benchmark.summarize(cases, records, 2e-11, 2e-11, 100, 0)["paired"][0]
            self.assertTrue(all(p["numeric_parity"] for p in pair["pairs"]))
            self.assertTrue(all(p["conditional_policy_parity"] for p in pair["pairs"]))
            self.assertFalse(pair["all_pairs_match"])
            self.assertIsNone(pair["ratio_estimate"])

    def test_orphan_with_live_worker_or_missing_worker_state_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaisesRegex(ValueError, "worker state unknown"):
                benchmark.orphan_worker_alive(folder)
            pid = os.getpgrp() if hasattr(os, "getpgrp") else os.getpid()
            (folder / ".worker.json").write_text(json.dumps({"status": "started", "pid": pid}))
            self.assertTrue(benchmark.orphan_worker_alive(folder))

    def test_exact_lock_rejects_mismatches_duplicates_and_nonpinned_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lock.txt"
            lock.write_text("# full environment\nFoo_Bar==1.2\nnumpy==2.0\n")
            installed = [("foo-bar", "1.2"), ("numpy", "2.0")]
            self.assertTrue(benchmark.lock_matches_environment(lock, installed)[0])
            self.assertFalse(benchmark.lock_matches_environment(lock, installed + [("extra", "1")])[0])
            self.assertFalse(benchmark.lock_matches_environment(lock, [("foo.bar", "1.3"), ("numpy", "2.0")])[0])
            lock.write_text("numpy>=2.0\n")
            self.assertFalse(benchmark.lock_matches_environment(lock, installed)[0])
            lock.write_text("foo-bar==1.2\nfoo_bar==1.2\n")
            self.assertFalse(benchmark.lock_matches_environment(lock, installed)[0])

    def test_matching_arrays_do_not_override_changed_effective_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            array = Path(directory) / "y.npy"
            np.save(array, np.ones((1, 1, 1)))
            cases = [{"name": "n", "method": "sgda", "backend": "numba"},
                     {"name": "r", "method": "sgda", "backend": "rust"}]
            records = []
            for rep in range(5):
                for case in cases:
                    manifest = {"config": {"seed": 0}, "seed": int(case["backend"] == "rust"),
                                "iterations": 10, "samples_per_pair": 1, "averaged_iterations": 10,
                                "occupation_feasibility": {"feasible": True},
                                "solve_seconds": 1.0, "plot_seconds": 1.0}
                    records.append({"case": case, "repetition": rep, "artifact_verified": True,
                                    "array": str(array), "wall_seconds": 1.0, "manifest": manifest})
            pair = benchmark.summarize(cases, records, 2e-11, 2e-11, 100, 0)["paired"][0]
            self.assertTrue(all(p["numeric_parity"] for p in pair["pairs"]))
            self.assertFalse(pair["all_pairs_match"])
            self.assertIsNone(pair["ratio_estimate"])

    def test_success_exit_with_missing_timing_is_not_a_valid_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text("x = 1\n")
            out = root / "n-000.png"
            out.write_bytes(b"image")
            np.save(root / "n-000.npy", np.ones((1, 1, 1)))
            manifest = {"method": "sgda", "backend": "numba", "config": {},
                        "config_sha256": benchmark.digest(config), "plot_sha256": benchmark.digest(out),
                        "occupation_sha256": benchmark.digest(root / "n-000.npy")}
            out.with_suffix(".json").write_text(json.dumps(manifest))
            row = benchmark.run_one([sys.executable, "-c", "pass"], root, config,
                {"name": "n", "method": "sgda", "backend": "numba"}, 0, root, 1, {})
            self.assertFalse(row.get("artifact_verified", False))
            self.assertIn("solve_seconds", row["error"])

    def test_infeasible_artifact_requires_opt_in_and_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text("x = 1\n")
            out = root / "n-000.png"
            out.write_bytes(b"diagnostic")
            array = root / "n-000.npy"
            np.save(array, np.ones((1, 1, 1)))
            manifest = {"method": "sgda", "backend": "numba", "config": {},
                        "config_sha256": benchmark.digest(config), "plot_sha256": benchmark.digest(out),
                        "occupation_sha256": benchmark.digest(array), "solve_seconds": 1., "plot_seconds": 1.,
                        "seed": 0, "iterations": 1, "samples_per_pair": 1, "averaged_iterations": 1,
                        "occupation_feasibility": {"feasible": False}, "plot_kind": "infeasible_iterate_diagnostic"}
            out.with_suffix(".json").write_text(json.dumps(manifest))
            args = ([sys.executable, "-c", "pass"], root, config,
                    {"name": "n", "method": "sgda", "backend": "numba"}, 0, root, 1, {})
            self.assertFalse(benchmark.run_one(*args).get("artifact_verified", False))
            self.assertTrue(benchmark.run_one(*args, allow_infeasible=True)["artifact_verified"])
            manifest["plot_kind"] = "occupation"
            out.with_suffix(".json").write_text(json.dumps(manifest))
            self.assertFalse(benchmark.run_one(*args, allow_infeasible=True).get("artifact_verified", False))

    def test_boundary_resume_and_artifact_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            config = root / "model.toml"
            config.write_text("a = 1\n")
            fake = source / "fake_cli.py"
            fake.write_text('''import argparse, hashlib, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument("command")
for name in ("config", "method", "backend", "output"): p.add_argument("--" + name, required=True)
p.add_argument("--rust-timeout", type=float)
p.add_argument("--allow-infeasible", action="store_true")
a=p.parse_args()
with open("invocations.log", "a") as f: f.write(a.backend + "\\n")
out=Path(a.output); out.write_bytes(b"plot")
array=out.with_suffix(".npy"); np.save(array, np.ones((1,1,1)))
digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=dict(method=a.method,backend=a.backend,config={},config_sha256=digest(a.config),
 plot_sha256=digest(out),occupation_sha256=digest(array),solve_seconds=.1,plot_seconds=.1,
 seed=0,iterations=1,samples_per_pair=1,averaged_iterations=1,
 occupation_feasibility=dict(feasible=True),plot_kind="occupation")
out.with_suffix(".json").write_text(json.dumps(manifest))
''')
            output = root / "run"
            cmd = [sys.executable, benchmark.__file__, "--command", shlex.join([sys.executable, str(fake)]),
                   "--cwd", str(source), "--config", str(config), "--output-dir", str(output),
                   "--environment-python", sys.executable, "--case", "n:sgda:numba", "--case", "r:sgda:rust",
                   "--numba-cache-state", "per-process-empty", "--repetitions", "2"]
            interrupted = subprocess.run(cmd + ["--stop-after", "2"], capture_output=True, text=True)
            self.assertEqual(interrupted.returncode, 130, interrupted.stderr)
            self.assertEqual(len(list((output / "shards").glob("*.json"))), 2)
            self.assertEqual(json.loads((output / "progress.json").read_text())["status"], "interrupted")
            orphan = output / "attempts" / "unfinished-attempt"
            orphan.mkdir()
            (orphan / "partial.png").write_bytes(b"partial")
            unknown = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
            self.assertNotEqual(unknown.returncode, 0)
            self.assertIn("orphan worker state unknown", unknown.stderr)
            self.assertEqual(len((source / "invocations.log").read_text().splitlines()), 2)
            (orphan / ".worker.json").write_text(json.dumps({"status": "finished", "pid": 999999}))
            resumed = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(len(list((output / "shards").glob("*.json"))), 4)
            self.assertEqual(len((source / "invocations.log").read_text().splitlines()), 4)
            progress = json.loads((output / "progress.json").read_text())
            self.assertEqual(progress["status"], "complete")
            self.assertIn(orphan.name, progress["orphaned_attempts"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["execution_segments"]), 2)
            self.assertTrue(summary["paired"][0]["all_pairs_match"])
            self.assertIsNone(summary["paired"][0]["ratio_estimate"])
            repeated = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertEqual(len((source / "invocations.log").read_text().splitlines()), 4)
            shard = json.loads(next((output / "shards").glob("*.json")).read_text())["payload"]
            (output / "attempts" / shard["attempt"] / next(n for n in shard["artifacts"] if n.endswith(".png"))).write_bytes(b"tampered")
            tampered = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("changed attempt artifacts", tampered.stderr)


if __name__ == "__main__":
    unittest.main()
