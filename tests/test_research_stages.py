"""Exercise small reference experiment jobs and their generated summaries."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lplearn.experiments.stages import learners


class ResearchStageTests(unittest.TestCase):
    def test_stage_jobs_summaries_and_reference_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lp_config, learner_config, results = root / 'lp.toml', root / 'learners.toml', root / 'results'
            lp_config.write_text('kinds=["lp"]\n[K5T10]\ncases=["CJS"]\npartitions=["none"]\n')
            learner_config.write_text('[K5T10]\ncases=["CJS"]\ngda_seeds=[0]\n'
                                      'sgda_methods=["D","E"]\nsgda_seeds=[0]\n[K5T10.hyper]\n'
                                      'n_iter=3\navg_window=2\nsamples_per_pair=2\n')
            for stage, config in [('lp_solutions', lp_config), ('learners', learner_config)]:
                prefix = [sys.executable, '-I', '-m', 'lplearn', 'experiments']
                args = [stage, '--config', str(config), '--results-dir', str(results)]
                run = subprocess.run([*prefix, 'run', *args, '--workers', '1', '--min-free-gb', '0'],
                                     capture_output=True, text=True, timeout=120)
                self.assertEqual(run.returncode, 0, run.stderr)
                summary = subprocess.run([*prefix, 'summarize', *args], capture_output=True, text=True, timeout=120)
                self.assertEqual(summary.returncode, 0, summary.stderr)
                self.assertTrue((results / stage / 'summary').is_dir())

            lp_summary = results / 'lp_solutions' / 'summary' / 'lp_solutions_summary.csv'
            learner_summary = results / 'learners' / 'summary' / 'solutions_K5T10.npz'
            self.assertTrue(lp_summary.is_file())
            self.assertTrue(learner_summary.is_file())
            table = (results / 'learners' / 'summary' / 'comparison_table.tex').read_text()
            self.assertIn('lplearn experiments summarize learners', table)
            # Reference values generated with the exact same job configuration at
            # c23e600; tiny learner runs deliberately do not claim convergence.
            expected = {'C': -14292.102542467259, 'D': -14311.174591541254,
                        'E': -14214.960176767212}
            found = set()
            for record in (results / 'learners').glob('*.json'):
                data = json.loads(record.read_text())
                method = data['job']['method']
                with np.load(record.with_suffix('.npz'), allow_pickle=False) as arrays:
                    self.assertEqual(arrays['y_avg'].shape, (10, 6, 4))
                    self.assertAlmostEqual(float(arrays['y_avg'].sum()), expected[method], delta=1e-7)
                found.add(method)
            self.assertEqual(found, {'C', 'D', 'E'})

    def test_nonfinite_learner_output_fails_job(self):
        cfg = {'K5T10': {'cases': ['CJS'], 'gda_seeds': [0], 'sgda_methods': [],
                         'hyper': {'n_iter': 3, 'avg_window': 2}}}
        job = learners.make_jobs(cfg)[0]
        with patch('lplearn.experiments.stages.learners.run_learner',
                   return_value={'y_avg': np.array([float('nan')])}):
            with self.assertRaisesRegex(RuntimeError, 'non-finite'):
                learners.run_job(job)


if __name__ == '__main__':
    unittest.main()
