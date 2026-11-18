"""Exercise the installed experiment runner through its public entry points."""

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lplearn.experiments.io import atomic_savez, load_arrays
from lplearn.experiments.runner import job_id


class ExperimentTests(unittest.TestCase):
    def test_cli_rejects_invalid_memory_floor_before_creating_results(self):
        with tempfile.TemporaryDirectory() as temp:
            results = Path(temp) / 'results'
            for value in ('nan', 'inf', '-1', '1e308'):
                with self.subTest(value=value):
                    proc = subprocess.run(
                        [sys.executable, '-I', '-m', 'lplearn', 'experiments', 'run',
                         'smoke', f'--min-free-gb={value}', '--results-dir', str(results)],
                        capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(proc.returncode, 2, proc.stderr)
                    self.assertIn('--min-free-gb must be finite', proc.stderr)
                    self.assertFalse(results.exists())

    @unittest.skipUnless(shutil.which('git'), 'git is required for caller checkout fixture')
    def test_caller_checkout_is_not_reported_as_installed_runtime(self):
        import lplearn

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            caller = root / 'unrelated-project'
            caller.mkdir()
            subprocess.run(['git', 'init', '-q', str(caller)], check=True)
            (caller / 'pyproject.toml').write_text('[project]\nname="unrelated"\nversion="0.0.1"\n')
            subprocess.run(['git', 'add', 'pyproject.toml'], cwd=caller, check=True)
            subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                            'commit', '-qm', 'unrelated'], cwd=caller, check=True)
            caller_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=caller, text=True).strip()
            config, results = root / 'smoke.toml', root / 'results'
            config.write_text('sizes=[4]\nalphas=[0.5]\nseeds=[0]\nsleep_s=0\n')
            proc = subprocess.run([sys.executable, '-I', '-m', 'lplearn', 'experiments', 'run', 'smoke',
                                   '--config', str(config), '--results-dir', str(results),
                                   '--workers', '1', '--min-free-gb', '0'], cwd=caller,
                                  capture_output=True, text=True, timeout=90)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads(next((results / 'manifests').glob('*.json')).read_text())
            self.assertEqual(manifest['schema'], 2)
            env = manifest['environment']
            self.assertEqual(Path(env['caller_project']['root']), caller.resolve())
            self.assertEqual(env['caller_project']['git']['commit'], caller_commit)
            runtime = env['installed_runtime']
            self.assertEqual(Path(runtime['import_path']), Path(lplearn.__file__).resolve())
            self.assertNotEqual(Path(runtime['package_root']), caller.resolve())
            self.assertTrue(runtime['distribution_matches_import'])
            self.assertTrue(runtime['distribution_record_sha256'])
            source = Path(runtime['package_root']) / 'experiments' / 'runner.py'
            self.assertEqual(runtime['runtime_files']['experiments/runner.py'], hashlib.sha256(source.read_bytes()).hexdigest())
            workers = runtime['packaged_workers']
            self.assertTrue(any(name.startswith('lplearn-solver') for name in workers))
            for entry in workers.values():
                self.assertEqual(entry['sha256'], hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest())

    def test_pickle_free_archive_and_stable_identity(self):
        self.assertEqual(job_id('smoke', '1', {'alpha': 0.5, 'seed': 1, 'size': 4}),
                         'd5cba1e9e7a80c63')
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'array.npz'
            atomic_savez(target, {'x': np.array([1., 2.])})
            self.assertEqual(set(load_arrays(target)), {'x'})
            with self.assertRaisesRegex(ValueError, 'object arrays'):
                atomic_savez(target, {'bad': np.array([object()], dtype=object)})

    def test_installed_cli_list_run_resume_status_summarize(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, results = root / 'smoke.toml', root / 'results'
            config.write_text('sizes=[4]\nalphas=[0.5]\nseeds=[0,1]\nsleep_s=0\n', encoding='utf-8')
            command = [sys.executable, '-I', '-m', 'lplearn', 'experiments']
            common = ['smoke', '--config', str(config), '--results-dir', str(results)]

            def call(*args):
                return subprocess.run([*command, *args], capture_output=True, text=True, timeout=90)

            listing = call('list')
            self.assertEqual(listing.returncode, 0, listing.stderr)
            self.assertIn('smoke', listing.stdout)
            self.assertIn('not written yet', listing.stdout)
            unavailable = call('status', 'finite_n', '--config', str(config), '--results-dir', str(results))
            self.assertEqual(unavailable.returncode, 2)
            self.assertIn('does not exist yet', unavailable.stderr)
            missing = call('status', *common)
            self.assertEqual(missing.returncode, 1)
            first = call('run', *common, '--workers', '1', '--min-free-gb', '0')
            self.assertEqual(first.returncode, 0, first.stderr)
            records = list((results / 'smoke').glob('*.json'))
            self.assertEqual(len(records), 2)
            self.assertTrue(all(json.loads(p.read_text())['status'] == 'ok' for p in records))
            second = call('run', *common, '--workers', '1', '--min-free-gb', '0')
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn('2 skipped', second.stderr)
            status = call('status', *common)
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertIn('complete   2', status.stdout)
            summary = call('summarize', *common)
            self.assertEqual(summary.returncode, 0, summary.stderr)
            rows = (results / 'smoke' / 'summary' / 'smoke_summary.csv').read_text().splitlines()
            self.assertEqual(len(rows), 3)

            # Test the packaged executable as well: spawned workers re-import
            # its entry point, which must not recursively rerun the CLI.
            executable = Path(sys.executable).parent / ('lplearn.cmd' if sys.platform == 'win32' else 'lplearn')
            if sys.platform != 'win32':
                other_results = root / 'executable-results'
                direct = subprocess.run([str(executable), 'experiments', 'run', 'smoke', '--config', str(config),
                                         '--results-dir', str(other_results), '--workers', '1', '--min-free-gb', '0'],
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(direct.returncode, 0, direct.stderr)


if __name__ == '__main__':
    unittest.main()
