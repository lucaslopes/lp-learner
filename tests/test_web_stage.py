"""Synthetic end-to-end web export; no manuscript data needed."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lplearn.experiments.runner import StageError
from lplearn.experiments.stages import web
from lplearn.visualizations import tutorial_assets as ta
from lplearn.visualizations.web_export import _r


class WebStageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / 'input'
        common = self.input / 'commons' / 'summary_tutorial'
        finite = self.input / 'finite_n' / 'summary'
        common.mkdir(parents=True)
        finite.mkdir(parents=True)
        self.model = {'K': 1, 'T': 2, 'p': .7, 'b_low': .01, 'b_high': .1, 'alpha': .75, 'beta': .25, 'gamma': 100., 'gamma_bar': 100.}
        objective = ta.solve_tutorial_model(ta.base_params(self.model)).lp['objective']
        base = {'case': 'CJS', 'tie_break': 'no_request', 'reps': 3}
        summary, paired, dilemma = [], [], []
        for scenario in ['base', 'beta=1.00', 'alpha=1.00', 'beta=0.10', 'beta=0.50']:
            for n in [10, 100, 1000]:
                for mech in ['M1', 'M2']:
                    shared = dict(base, scenario=scenario, N=n, mechanism=mech)
                    paired.append(dict(shared, comparison='selfish_minus_coord', diff_mean=1., diff_ci95=.1, pct_of_coord=2.))
                    for pop in ['coord', 'selfish', 'price']:
                        summary.append(dict(shared, population=pop, cost_mean=10., cost_ci95=.1, requested_admit=.5, requested_fast=.2, overload_admit=.01, overload_fast=.01, full_mean_per_stage=.1, not_admitted_per_queue_horizon=1., blocked_per_queue_horizon=.1))
                    for _, name in ta.DEVIATION_TESTS:
                        dilemma.append(dict(shared, test=name, diff_mean=.1, diff_ci95=.01, verdict='synthetic'))
                    dilemma.append(dict(shared, test='verdict: social dilemma (collective worse and individual rational)', diff_mean=.1, diff_ci95=.01, verdict='synthetic'))
        for name, rows in [('commons_summary.csv', summary), ('commons_paired.csv', paired), ('commons_dilemma.csv', dilemma), ('commons_stages.csv', [{'dummy': 1}]), ('commons_fluid.csv', [dict(scenario=s, case='CJS', lp_value=objective) for s, _, _ in ta.WHATIF])]:
            pd.DataFrame(rows).to_csv(common / name, index=False)
        rows = [dict(case='CJS', method=m, policy=p, N=n, lp_objective=objective, cost_mean=10., repairs_frac=.01, source_mass_final=.99, n_reps=3) for m, p, _ in ta.COORDINATORS for n in (10, 100, 1000)]
        pd.DataFrame(rows).to_csv(finite / 'finite_n_summary.csv', index=False)
        self.cfg = {'results_root': str(self.input), 'model': self.model, 'alphas': [.75], 'betas': [.25]}

    def test_real_export_validates_mass_and_budgets(self):
        out = self.root / 'summary'
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.input.rglob('*.csv')}
        web.summarize([], self.cfg, out)
        model = json.loads((out / 'model.json').read_text())
        reference = json.loads((out / 'reference.json').read_text())
        manifest = json.loads((out / 'web_manifest.json').read_text())
        self.assertEqual(model['results']['replications'], 3)
        self.assertEqual(len(model['lpGrid']), 1)
        row = model['lpGrid'][0]
        y = np.array(row['y'])
        self.assertEqual(y.shape, (2, 2, 4))
        self.assertGreaterEqual(y.min(), 0)
        np.testing.assert_allclose(y.sum(axis=(1, 2)), 1, atol=1e-6)
        self.assertTrue(np.all(y[:, :, 2:].sum(axis=(1, 2)) <= .75 + 1e-6))
        self.assertTrue(np.all(y[:, :, [1, 3]].sum(axis=(1, 2)) <= .25 + 1e-6))
        self.assertTrue(row['occupationValidation']['raw']['feasible'])
        self.assertTrue(row['occupationValidation']['exported']['feasible'])
        self.assertTrue(reference['lpOccupationValidation']['feasible'])
        self.assertIn(reference['gda']['kind'], ['iterate', 'infeasible_iterate_diagnostic'])
        np.testing.assert_allclose(np.array(reference['P']).sum(axis=-1), 1, atol=1e-10)
        self.assertEqual(manifest['inputs'], {str(p): h for p, h in hashes.items()})
        self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes})
        self.assertFalse((self.root / 'web').exists())

    def test_missing_input_and_invalid_grid_are_rejected(self):
        out = self.root / 'missing'
        out.mkdir()
        previous = out / 'model.json'
        previous.write_text('previous output')
        (self.input / 'commons' / 'summary_tutorial' / 'commons_paired.csv').unlink()
        with self.assertRaisesRegex(StageError, 'missing input'):
            web.summarize([], self.cfg, out)
        self.assertEqual(previous.read_text(), 'previous output')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            _r(float('nan'))

    def test_invalid_grid_is_rejected(self):
        with self.assertRaisesRegex(StageError, 'alpha and beta grids'):
            web.summarize([], {**self.cfg, 'alphas': []}, self.root / 'invalid')
        self.assertFalse((self.root / 'invalid' / 'model.json').exists())

    def test_mixed_replication_counts_are_rejected(self):
        path = self.input / 'finite_n' / 'summary' / 'finite_n_summary.csv'
        data = pd.read_csv(path)
        data.loc[0, 'n_reps'] = 4
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(StageError, 'replication count'):
            web.summarize([], self.cfg, self.root / 'mixed')
        self.assertFalse((self.root / 'mixed' / 'model.json').exists())

    @unittest.skipUnless(os.environ.get('WEB_WHEEL_CLI_TEST'), 'requires an installed wheel')
    def test_installed_cli_end_to_end(self):
        config = self.root / 'web.toml'
        config.write_text('results_root = ' + json.dumps(str(self.input)) + '\nalphas = [0.75]\nbetas = [0.25]\n[model]\n' +
                          ''.join(f'{key} = {value}\n' for key, value in self.model.items()))
        output = self.root / 'run'
        command = [sys.executable, '-I', '-m', 'lplearn', 'experiments']
        for action in ('run', 'summarize'):
            args = [*command, action, 'web', '--config', str(config), '--results-dir', str(output)]
            if action == 'run':
                args += ['--workers', '1', '--min-free-gb', '0']
            result = subprocess.run(args, cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = output / 'web' / 'summary'
        self.assertEqual(json.loads((out / 'model.json').read_text())['results']['replications'], 3)
        self.assertTrue((out / 'reference.json').is_file())
        self.assertFalse((self.root / 'web').exists())


if __name__ == '__main__':
    unittest.main()
