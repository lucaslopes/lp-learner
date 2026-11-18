"""Backend identity, cache separation and scientific parity for learners."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lplearn.experiments import runner
from lplearn.experiments.stages import learners


def config(backend=None):
    cfg = {'K5T10': {'cases': ['CJS'], 'gda_seeds': [0], 'sgda_methods': ['D', 'E'],
                     'sgda_seeds': [0], 'hyper': {'n_iter': 3, 'avg_window': 2, 'samples_per_pair': 2}}}
    if backend is not None:
        cfg['backend'] = backend
    return cfg


class ExperimentBackendTests(unittest.TestCase):
    def test_identity_and_validation(self):
        default = learners.make_jobs(config())
        numba = learners.make_jobs(config('numba'))
        rust = learners.make_jobs(config('rust'))
        self.assertEqual(default, numba)
        self.assertEqual([j['method'] for j in numba], ['C', 'D', 'E'])
        for left, right in zip(numba, rust):
            self.assertNotEqual(runner.job_id('learners', learners.VERSION, left),
                                runner.job_id('learners', learners.VERSION, right))
            self.assertEqual(left['seed'], right['seed'])
        with self.assertRaisesRegex(runner.ConfigError, 'backend'):
            learners.make_jobs(config('unknown'))
        with self.assertRaisesRegex(runner.ConfigError, 'rust_timeout'):
            learners.make_jobs({**config('rust'), 'rust_timeout': 0})
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / 'worker'
            binary.write_bytes(b'first backend')
            with patch('lplearn.rust._executable', return_value=binary):
                old = learners.make_jobs(config('rust'))[0]
                binary.write_bytes(b'updated backend')
                new = learners.make_jobs(config('rust'))[0]
            self.assertNotEqual(old['backend_binary_sha256'], new['backend_binary_sha256'])
            self.assertNotEqual(runner.job_id('learners', learners.VERSION, old),
                                runner.job_id('learners', learners.VERSION, new))
        with patch('lplearn.experiments.stages.learners.metadata.version', return_value='1.2.3'):
            old = learners.make_jobs(config('numba'))[0]
        with patch('lplearn.experiments.stages.learners.metadata.version', return_value='1.2.4'):
            new = learners.make_jobs(config('numba'))[0]
        self.assertNotEqual(runner.job_id('learners', learners.VERSION, old),
                            runner.job_id('learners', learners.VERSION, new))

    def test_same_iterates_and_aggregation_weights(self):
        for a, b in zip(learners.make_jobs(config('numba')), learners.make_jobs(config('rust'))):
            with self.subTest(method=a['method']):
                left, right = learners.run_job(a), learners.run_job(b)
                for key in ('y_avg', 'y_last', 'lam_res_avg', 'lam_res_last', 'mass_avg', 'mass_last',
                            'trace_iter', 'trace_y', 'trace_lam_res', 'trace_mu_norm'):
                    np.testing.assert_allclose(left[key], right[key], atol=2e-11, rtol=2e-11, err_msg=key)
                for key in ('occupation_min', 'flow_violation', 'budget_violation', 'nonneg_violation'):
                    self.assertAlmostEqual(left[key], right[key], delta=2e-11, msg=key)
                self.assertEqual(left['backend'], 'numba')
                self.assertEqual(right['backend'], 'rust')
                self.assertRegex(right['backend_binary_sha256'], r'^[0-9a-f]{64}$')
                self.assertGreaterEqual(right['wall_time'], right['backend_execute_seconds'])

    def test_missing_worker_records_failed_without_fallback(self):
        job = learners.make_jobs(config('rust'))[0]
        jid = runner.job_id('learners', learners.VERSION, job)
        with tempfile.TemporaryDirectory() as temp:
            with patch('lplearn.rust._executable', return_value=Path(temp) / 'missing-solver'):
                outcome = runner._execute_job(learners.__name__, learners.NAME, learners.VERSION,
                                              temp, jid, job)
            self.assertEqual(outcome.status, 'failed')
            record = json.loads((Path(temp) / f'{jid}.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertIn('missing-solver', record['error'])
            self.assertFalse((Path(temp) / f'{jid}.npz').exists())

    def test_installed_runner_keeps_both_backend_caches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {}
            for backend in ('numba', 'rust'):
                path = root / f'{backend}.toml'
                path.write_text(f'backend="{backend}"\n[K5T10]\ncases=["CJS"]\n'
                                'gda_seeds=[0]\nsgda_methods=["D","E"]\nsgda_seeds=[0]\n'
                                '[K5T10.hyper]\nn_iter=3\navg_window=2\nsamples_per_pair=2\n')
                files[backend] = path
            base = [sys.executable, '-I', '-m', 'lplearn', 'experiments']
            results = root / 'results'
            def call(action, backend, *extra):
                return subprocess.run([*base, action, 'learners', '--config', str(files[backend]),
                                       '--results-dir', str(results), *extra],
                                      capture_output=True, text=True, timeout=180)
            for backend in ('numba', 'rust'):
                done = call('run', backend, '--workers', '1', '--min-free-gb', '0')
                self.assertEqual(done.returncode, 0, done.stderr)
                out = root / f'summary-{backend}'
                summary = call('summarize', backend, '--out-dir', str(out))
                self.assertEqual(summary.returncode, 0, summary.stderr)
                index = (out / 'solutions_index.csv').read_text()
                self.assertIn('backend', index.splitlines()[0])
                self.assertIn(backend, index)
            records = [json.loads(p.read_text()) for p in (results / 'learners').glob('*.json')]
            self.assertEqual(len(records), 6)
            self.assertEqual({r['job']['backend'] for r in records}, {'numba', 'rust'})
            again = call('run', 'numba', '--workers', '1', '--min-free-gb', '0')
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertIn('3 skipped', again.stderr)


if __name__ == '__main__':
    unittest.main()
