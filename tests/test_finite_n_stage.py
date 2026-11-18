"""Small installed finite-N stage and learned-iterate diagnostic checks."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lplearn.experiments.stages import finite_n


class FiniteNStageTests(unittest.TestCase):
    def test_small_exact_saddle_and_lp_update_through_runner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, results = root / 'finite.toml', root / 'results'
            config.write_text('K=1\nT=2\ncases=["CJS"]\nmethods=["A","C"]\n'
                              'lp_update_methods=["A"]\nindependent_methods=["A"]\n'
                              'saddle_source="exact_saddle"\nN_values=[4]\n'
                              'n_chunks=1\nreps_per_chunk=2\nseed=7\n')
            prefix = [sys.executable, '-I', '-m', 'lplearn', 'experiments']
            common = ['finite_n', '--config', str(config), '--results-dir', str(results)]
            run = subprocess.run([*prefix, 'run', *common, '--workers', '1', '--min-free-gb', '0'],
                                 capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 0, run.stderr)
            records = list((results / 'finite_n').glob('*.json'))
            self.assertEqual(len(records), 4)
            keys = set()
            indices = []
            for record in records:
                data = json.loads(record.read_text())
                self.assertEqual(data['status'], 'ok')
                job = data['job']
                keys.add((job['method'], job['policy']))
                with np.load(record.with_suffix('.npz'), allow_pickle=False) as result:
                    indices.append(result['rep_index'])
                    self.assertTrue(np.isfinite(result['cost']).all())
                    if job['mechanism'] == 'StrictBudget':
                        self.assertEqual(float(result['viol_max'].max()), 0)
            self.assertEqual(keys, {('A', 'projection'), ('A', 'lp_update'),
                                    ('A', 'independent'), ('C', 'projection')})
            for value in indices:
                np.testing.assert_array_equal(value, [0, 1])
            summary = subprocess.run([*prefix, 'summarize', *common], capture_output=True, text=True, timeout=120)
            self.assertEqual(summary.returncode, 0, summary.stderr)
            self.assertEqual(len((results / 'finite_n' / 'summary' / 'finite_n_summary.csv').read_text().splitlines()), 5)

    def test_learned_diagnostics_before_clipping_and_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'learned.npz'
            y = np.zeros((2, 2, 4))
            y[0, 0, 0] = -2.0
            y[0, 1, 1] = 0.25
            np.savez_compressed(archive, **{'CJS/D/seed0/y_avg': y})
            cfg = {'K': 1, 'T': 2, 'cases': ['CJS'], 'methods': ['D'], 'N_values': [4],
                   'n_chunks': 1, 'reps_per_chunk': 2, 'seed': 7, 'saddle_source': 'learned',
                   'learned_file': str(archive), 'learned_seeds': [0]}
            planned = finite_n.make_jobs(cfg)[0]
            result = finite_n.run_job(planned)
            self.assertEqual(result['learned_raw_min'], -2.0)
            np.testing.assert_array_equal(result['learned_raw_mass'], [-1.75, 0.0])
            self.assertEqual(result['learned_nonneg_violation'], 2.0)
            self.assertGreater(result['learned_flow_violation'], 0)
            self.assertEqual(result['learned_fallback_rows'], 3)
            self.assertEqual(result['learned_empty_stages'], 1)
            self.assertEqual(float(result['viol_max'].max()), 0.0)
            y[0, 1, 1] = 0.5
            np.savez_compressed(archive, **{'CJS/D/seed0/y_avg': y})
            with self.assertRaisesRegex(ValueError, 'changed after job planning'):
                finite_n.run_job(planned)
            y[0, 0, 0] = np.nan
            np.savez_compressed(archive, **{'CJS/D/seed0/y_avg': y})
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                finite_n.run_job(finite_n.make_jobs(cfg)[0])


if __name__ == '__main__':
    unittest.main()
