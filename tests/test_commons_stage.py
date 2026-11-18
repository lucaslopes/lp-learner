"""Small installed commons-stage checks against the pinned Python reference."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lplearn.commons.experiment import Game, _simulate, solve_method_a
from lplearn.commons.mechanisms import Unenforced as CommonsUnenforced
from lplearn.core import QueueParams, conditional_policy, queue_wcmdp
from lplearn.finite_n import LPProjectionController, Unenforced as FiniteUnenforced, simulate


class CommonsStageTests(unittest.TestCase):
    def test_small_reference_jobs_and_summaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, results = root / 'commons.toml', root / 'results'
            config.write_text('K=1\nT=2\ncases=["CJS"]\nns=[4]\nmechanisms=["M0","M1","M2"]\n'
                              'tie_breaks=["no_request"]\nn_chunks=1\nreps_per_chunk=3\nseed=7\n'
                              'table_n=4\nsensitivity=[]\n')
            command = [sys.executable, '-I', '-m', 'lplearn', 'experiments']
            common = ['commons', '--config', str(config), '--results-dir', str(results)]

            def call(*args):
                return subprocess.run([*command, *args], capture_output=True, text=True, timeout=120)

            self.assertEqual(call('status', *common).returncode, 1)
            run = call('run', *common, '--workers', '1', '--min-free-gb', '0')
            self.assertEqual(run.returncode, 0, run.stderr)
            rerun = call('run', *common, '--workers', '1', '--min-free-gb', '0')
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertIn('3 skipped', rerun.stderr)
            self.assertEqual(call('status', *common).returncode, 0)
            summary = call('summarize', *common)
            self.assertEqual(summary.returncode, 0, summary.stderr)

            # Per-replication outputs from the identical configuration in the
            # pinned c23e600 reference. Separate mechanism behavior and stream
            # ordering are checked, not just aggregate means.
            expected = {
                'M0': {'coord': [104.835, 79.585, 104.835],
                       'indep': [105.13, 104.835, 104.54],
                       'selfish': [105.22, 79.97, 105.22]},
                'M1': {'coord': [104.835, 79.585, 104.835],
                       'indep': [104.835, 104.835, 104.54],
                       'selfish': [104.835, 79.585, 104.835]},
                'M2': {'coord': [104.835, 79.585, 104.835],
                       'indep': [104.88, 104.835, 104.54],
                       'selfish': [104.97, 104.97, 104.97]},
            }
            directory = results / 'commons'
            records = list(directory.glob('*.json'))
            self.assertEqual(len(records), 3)
            found = set()
            for record in records:
                job = json.loads(record.read_text())
                self.assertEqual(job['status'], 'ok')
                self.assertEqual(job['job']['seed'], 7)
                self.assertEqual(job['job']['rep_start'], 0)
                mechanism = job['job']['mechanism']
                found.add(mechanism)
                with np.load(record.with_suffix('.npz'), allow_pickle=False) as arrays:
                    self.assertAlmostEqual(float(arrays['lp_value']), 100.65774596095265, delta=1e-7)
                    for name, target in expected[mechanism].items():
                        np.testing.assert_allclose(arrays[f'{name}__cost'], target, atol=1e-8, rtol=0)
                    np.testing.assert_array_equal(arrays['policy_selfish_u'], [[3, 1], [2, 0]])
                    for name in ('lp_nonnegativity_violation', 'lp_initial_residual', 'lp_flow_residual',
                                 'lp_stage_mass_residual', 'lp_resource_violation'):
                        self.assertLessEqual(float(arrays[name]), float(arrays['lp_feasibility_tolerance']))
                    self.assertLessEqual(float(arrays['coord__use_a'].max()), 0.75)
                    self.assertLessEqual(float(arrays['coord__use_b'].max()), 0.25)
            self.assertEqual(found, set(expected))
            summary_dir = directory / 'summary'
            self.assertEqual({p.name for p in summary_dir.iterdir()}, {
                'commons_summary.csv', 'commons_paired.csv', 'commons_dilemma.csv',
                'commons_incentives.csv', 'commons_fluid.csv', 'commons_stages.csv', 'commons_table.tex'})
            with (summary_dir / 'commons_summary.csv').open(newline='') as file:
                self.assertEqual(len(list(csv.DictReader(file))), 21)
            with (summary_dir / 'commons_fluid.csv').open(newline='') as file:
                fluid = list(csv.DictReader(file))
                self.assertEqual(len(fluid), 3)
                self.assertTrue(all(float(row['lp_flow_residual']) < 1e-6 for row in fluid))

    def test_common_random_numbers_match_finite_population_simulator(self):
        params = QueueParams(K=2, T=3, case='SJS')
        model = queue_wcmdp(params)
        lp = solve_method_a(model)
        pi = conditional_policy(lp['y'])
        game = Game(params, model, np.asarray(lp['lam_res']))
        factory = lambda: LPProjectionController(pi, model.budgets)
        ref = simulate(model, factory, 5, 3, 2026, FiniteUnenforced(), rep_offset=0)
        actual = _simulate(game, factory(), CommonsUnenforced(params.b_low, params.b_high),
                           5, range(3), 2026, detailed=True)
        np.testing.assert_allclose(actual['cost'], ref['cost'], atol=1e-9, rtol=0)
        # Commons stores stage-level values in float32; finite_n stores float64.
        np.testing.assert_allclose(actual['cost_t'], ref['cost_t'], atol=5e-6, rtol=0)
        np.testing.assert_allclose(actual['use_a'], ref['adm_t'], atol=1e-7, rtol=0)

    def test_infeasible_lp_is_rejected_before_policy_conversion(self):
        model = queue_wcmdp(QueueParams(K=1, T=2))
        invalid = np.zeros((2, 2, 4))
        invalid[0, 0, 0] = -0.5
        with patch('lplearn.commons.experiment.solve_lp', return_value={'y': invalid}):
            with self.assertRaisesRegex(RuntimeError, 'infeasible before policy conversion'):
                solve_method_a(model)


if __name__ == '__main__':
    unittest.main()
