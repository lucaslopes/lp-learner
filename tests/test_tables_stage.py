"""Exercise table generation from explicit, synthetic upstream summaries."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

# Synthetic-input output hashes from reference c23e600, after updating only
# the generated comment to the new `lplearn experiments` command name.
REFERENCE_HASHES = {
    'bound': '4143d1f7e7758fbf393984caa4eed43ee3a493d94a70e5572d3b5109e0bad452',
    'commons': '8a1cab8a47bf1ca3e9c5530411fd5913f6a5c8c5691a309b1dd2c4226a08d4c0',
    'finite_n': '8aad7c5a5a86601ccd64a1f71a78ca0fc8b350997151489c61932718b5a96813',
    'lp_comparison': 'a94a903e3fcacf94f8750a20b202285b4598f7c9c385ad059f64d185eeb809d5',
}


def fixture_inputs(root):
    """Populate schema-complete synthetic inputs, never research results."""
    for stage in ('learners', 'bound', 'finite_n', 'commons'):
        (root / stage / 'summary').mkdir(parents=True, exist_ok=True)
    for case in ('CJS', 'SJS'):
        rows = []
        for i, label in enumerate('ABCDEFGHIJK'):
            rows.append(dict(label=label, n_seeds=2, objective_mean=3+i, objective_sd=0.1,
                             gap_A_pct_mean=0.2, mass_last_mean=1.0, dist_B_mean=0.3,
                             dist_B_sd=0.04, aggregation_effect=0.05, run_to_run_y_mean=0.06,
                             lam_dist_B_mean=0.7, lam_dist_B_sd=0.08))
        pd.DataFrame(rows).to_csv(root / 'learners' / 'summary' / f'comparison_{case}.csv', index=False)
    rows = []
    for label in 'EFGHIJK':
        for case, regime, convention in (('CJS', 'lemma2', 'consistent'), ('CJS', 'lemma2', 'paper'),
                                          ('CJS', 'grid2e-05', 'consistent'), ('SJS', 'lemma2', 'consistent')):
            rows.append(dict(K=5, T=10, case=case, label=label, regime=regime, convention=convention,
                             B_spec=2.0, dist_total=1.0, delta_c=0.2, delta_P=0.1, delta_0=0.0,
                             B_lemma1=3.0, kappa_max=4.0))
    pd.DataFrame(rows).to_csv(root / 'bound' / 'summary' / 'bound_rows.csv', index=False)
    rows = [dict(case=case, method='A', policy='projection', N=4, lp_objective=2.0, viol_max=0,
                 cost_mean=2.1, cost_ci95=0.1, n_reps=5, n_seeds=1) for case in ('CJS', 'SJS')]
    pd.DataFrame(rows).to_csv(root / 'finite_n' / 'summary' / 'finite_n_summary.csv', index=False)
    rows, pairs = [], []
    selections = [('coord', 'M1'), ('indep', 'M0'), ('indep', 'M1'), ('selfish', 'M0'),
                  ('selfish', 'M1'), ('selfish', 'M2')]
    for case in ('CJS', 'SJS'):
        for population, mechanism in selections:
            common = dict(scenario='base', case=case, N=1000, mechanism=mechanism, tie_break='no_request')
            rows.append(dict(**common, population=population, overload_admit=0.1, overload_fast=0.2,
                             excess_fast_mean=0.03, cost_mean=2.1, cost_ci95=0.1,
                             full_mean_per_stage=0.04, reps=5))
            if population != 'coord':
                pairs.append(dict(**common, comparison=f'{population}_minus_coord', diff_mean=0.1,
                                  diff_ci95=0.02, pct_of_coord=5.0))
    pd.DataFrame(rows).to_csv(root / 'commons' / 'summary' / 'commons_summary.csv', index=False)
    pd.DataFrame(pairs).to_csv(root / 'commons' / 'summary' / 'commons_paired.csv', index=False)


class TableStageTests(unittest.TestCase):
    def test_installed_cli_generates_all_tables_and_reports_missing_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / 'results'
            fixture_inputs(results)
            config = root / 'tables.toml'
            config.write_text(f'results_root = {json.dumps(str(results))}\n'
                              'tables = ["lp_comparison", "bound", "finite_n", "commons"]\n')
            cli = [sys.executable, '-I', '-m', 'lplearn', 'experiments']
            common = ['tables', '--config', str(config), '--results-dir', str(results)]

            def call(*args):
                return subprocess.run([*cli, *args], cwd=root, text=True, capture_output=True, timeout=90)

            missing = call('status', *common)
            self.assertEqual(missing.returncode, 1, missing.stderr)
            run = call('run', *common, '--workers', '1', '--min-free-gb', '0')
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(call('status', *common).returncode, 0)
            again = call('run', *common, '--workers', '1', '--min-free-gb', '0')
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertIn('1 skipped', again.stderr)
            summary = call('summarize', *common)
            self.assertEqual(summary.returncode, 0, summary.stderr)
            out = results / 'tables' / 'summary'
            manifest = json.loads((out / 'tables_manifest.json').read_text())
            self.assertEqual(set(manifest['written']), {'lp_comparison', 'bound', 'finite_n', 'commons'})
            self.assertEqual(manifest['failed'], {})
            for name in manifest['written']:
                table = (out / f'{name}.tex').read_text()
                self.assertIn('\\begin{table}', table)
                self.assertIn('lplearn experiments summarize tables', table)
                self.assertEqual(hashlib.sha256(table.encode()).hexdigest(), REFERENCE_HASHES[name])
            self.assertFalse((root / 'papers').exists())

            # One missing input must fail without writing any newly selected table.
            (results / 'bound' / 'summary' / 'bound_rows.csv').unlink()
            other = root / 'other'
            failure = call('summarize', *common, '--out-dir', str(other))
            self.assertEqual(failure.returncode, 2)
            self.assertIn('missing input', failure.stderr)
            self.assertFalse(any(other.glob('*.tex')))
            failed = json.loads((other / 'tables_manifest.json').read_text())
            self.assertEqual(failed['written'], [])
            self.assertIn('bound', failed['failed'])

    def test_rejects_invalid_table_selection_and_bound_violation(self):
        from lplearn.experiments.runner import StageError
        from lplearn.experiments.stages import tables

        for bad in ([], 'bound', ['unknown'], ['bound', 'bound']):
            with self.assertRaises(StageError):
                tables.make_jobs({'tables': bad})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_inputs(root)
            source = root / 'bound' / 'summary' / 'bound_rows.csv'
            df = pd.read_csv(source)
            df.loc[0, 'B_spec'] = 0.5
            df.to_csv(source, index=False)
            with self.assertRaisesRegex(StageError, 'spectral bound'):
                tables.bound_table(root)

    def test_finite_population_requires_observed_budget_checks(self):
        from lplearn.experiments.runner import StageError
        from lplearn.experiments.stages import tables

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_inputs(root)
            source = root / 'finite_n' / 'summary' / 'finite_n_summary.csv'
            frame = pd.read_csv(source)
            frame.loc[0, 'viol_max'] = float('nan')
            frame.to_csv(source, index=False)
            with self.assertRaisesRegex(StageError, 'missing or invalid'):
                tables.finite_n_table(root)
            frame['policy'] = 'independent'
            frame.to_csv(source, index=False)
            with self.assertRaisesRegex(StageError, 'missing or invalid'):
                tables.finite_n_table(root)


if __name__ == '__main__':
    unittest.main()
