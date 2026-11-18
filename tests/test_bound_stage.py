"""Small end-to-end bound stage checks against the pinned Python reference."""

import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lplearn.analysis import bound as study


class BoundStageTests(unittest.TestCase):
    def test_real_job_resume_status_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, results = root / 'bound.toml', root / 'results'
            # One E-partition queue, one regularization and one lifting map.
            # This is a bounded fixture, not the manuscript's full workload.
            config.write_text('K=5\nT=3\ncases=["CJS"]\npartitions=["E"]\n'
                              'lemma_factors=[2.0]\ngamma_grid=[]\nconventions=["consistent"]\n'
                              'eta_grid_points=2\nsweep_p=[]\nsweep_b_high=[]\n'
                              'include_lumpable=false\ninclude_k11=false\n')
            command = [sys.executable, '-I', '-m', 'lplearn', 'experiments']
            common = ['bound', '--config', str(config), '--results-dir', str(results)]

            def call(*args):
                return subprocess.run([*command, *args], capture_output=True, text=True, timeout=120)

            status = call('status', *common)
            self.assertEqual(status.returncode, 1, status.stderr)
            first = call('run', *common, '--workers', '1', '--min-free-gb', '0')
            self.assertEqual(first.returncode, 0, first.stderr)
            rerun = call('run', *common, '--workers', '1', '--min-free-gb', '0')
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertIn('1 skipped', rerun.stderr)
            status = call('status', *common)
            self.assertEqual(status.returncode, 0, status.stderr)
            summary = call('summarize', *common)
            self.assertEqual(summary.returncode, 0, summary.stderr)

            directory = results / 'bound'
            archives = list(directory.glob('*.npz'))
            self.assertEqual(len(archives), 1)
            with np.load(archives[0], allow_pickle=False) as data:
                rows = json.loads(str(data['rows_json']))
                self.assertEqual(int(data['n_rows']), 1)
            row = rows[0]
            self.assertEqual((row['case'], row['label'], row['convention'], row['regime']),
                             ('CJS', 'E', 'consistent', 'lemma2'))
            self.assertTrue(row['check_ok'])
            self.assertGreaterEqual(row['B_spec'], row['dist_total'] - 1e-9)
            # Obtained from the same job at Python reference c23e600; allow
            # for variation between supported BLAS/solver implementations.
            self.assertAlmostEqual(row['B_lemma1'], 0.5645015865222078, delta=1e-7)
            self.assertAlmostEqual(row['B_spec'], 0.28225079329510977, delta=1e-7)
            self.assertAlmostEqual(row['dist_total'], 0.2774686054215667, delta=1e-7)
            summary_dir = directory / 'summary'
            expected = {'bound_rows.csv', 'bound_rank.json', 'deltas_cjs_sjs.csv',
                        'deltas_sweep.csv', 'bound_table.tex', 'deltas_table.tex'}
            self.assertEqual({p.name for p in summary_dir.iterdir()}, expected)
            with (summary_dir / 'bound_rows.csv').open(newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            checks = json.loads((summary_dir / 'bound_rank.json').read_text())['checks']
            self.assertEqual(checks['rows'], 1)
            self.assertGreaterEqual(checks['min_slack'], -1e-9)

    def test_nonfinite_json_and_invalid_bound_checks(self):
        value = json.loads(study.dumps_compact({'finite': 1.0, 'invalid': math.nan}))
        self.assertEqual(value, {'finite': 1.0, 'invalid': None})
        row = {'B_lemma1': 0.1, 'B_spec': 0.2, 'dist_total': 0.3,
               'label': 'E', 'case': 'CJS', 'regime': 'lemma2'}
        with self.assertRaisesRegex(RuntimeError, 'bound violated'):
            study.check_rows([row])


if __name__ == '__main__':
    unittest.main()
