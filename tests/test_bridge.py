"""Checks against an installed wheel, including worker failure paths."""

import json
from importlib import metadata
from pathlib import Path
import subprocess
import sys
import sysconfig
import unittest
from unittest import mock

import lplearn
from lplearn import backend


class BridgeTests(unittest.TestCase):
    def test_python_module_invokes_installed_backend(self):
        result = subprocess.run(
            [sys.executable, "-I", "-m", "lplearn", "backend", "ping"],
            text=True, capture_output=True, check=True,
        )
        self.assertEqual(json.loads(result.stdout)["backend"], "rust")

    def test_native_cli_launcher_invokes_installed_python_cli(self):
        is_windows = sys.platform == "win32"
        name = "lplearn.cmd" if is_windows else "lplearn"
        launcher = Path(sysconfig.get_path("scripts")) / name
        result = subprocess.run(
            f'"{launcher}" backend ping' if is_windows else [str(launcher), "backend", "ping"],
            shell=is_windows,
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["backend"], "rust")

    def test_ping_and_info(self):
        for command, call in (("ping", lplearn.backend_ping), ("info", lplearn.backend_info)):
            result = call()
            self.assertEqual(result["schema"], "lplearn-bridge/1")
            self.assertEqual(result["backend"], "rust")
            self.assertEqual(result["command"], command)
            self.assertEqual(result["capabilities"], ["gda", "sgda"])
            self.assertEqual(result["version"], metadata.version("lplearn"))

    def test_worker_rejects_bad_input(self):
        dist = metadata.distribution("lplearn")
        name = "lplearn-native.exe" if sys.platform == "win32" else "lplearn-native"
        executable = next(dist.locate_file(f) for f in dist.files if f.name == name)
        for data in ("{", json.dumps({"schema": "bad", "command": "ping"}),
                     json.dumps({"schema": "lplearn-bridge/1", "command": "solve"})):
            result = subprocess.run([str(executable)], input=data, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertTrue(result.stderr)

    def test_worker_matches_versioned_protocol_fixture(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "ping.json").read_text())
        dist = metadata.distribution("lplearn")
        name = "lplearn-native.exe" if sys.platform == "win32" else "lplearn-native"
        executable = next(dist.locate_file(f) for f in dist.files if f.name == name)
        result = subprocess.run(
            [str(executable)], input=json.dumps(fixture["request"]),
            text=True, capture_output=True, check=True,
        )
        self.assertEqual(json.loads(result.stdout), fixture["response"])
        self.assertEqual(result.stderr, "")

    def test_failure_and_timeout_do_not_fall_back(self):
        with mock.patch.object(backend.subprocess, "run", return_value=subprocess.CompletedProcess([], 4, "", "failed")):
            with self.assertRaisesRegex(lplearn.BackendError, "exited with 4"):
                lplearn.backend_ping()
        with mock.patch.object(backend.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 1)):
            with self.assertRaisesRegex(lplearn.BackendError, "timed out"):
                lplearn.backend_ping(timeout=1)
        with mock.patch.object(backend.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "not json", "")):
            with self.assertRaisesRegex(lplearn.BackendError, "invalid JSON"):
                lplearn.backend_ping()


if __name__ == "__main__":
    unittest.main()
