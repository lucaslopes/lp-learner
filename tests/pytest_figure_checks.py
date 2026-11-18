'''
Tests for the publication figures (lp_learner/visualizations/paper_figures.py) and the ``figures`` stage.

Every figure is drawn from small synthetic inputs that follow the schemas the stages write (see
reports/figures/README.md), and must produce a non-empty PDF with embedded TrueType fonts. The layout conversion, the
wedge order of the pies, the schema checks, the reporting of missing inputs and the stage contract are checked
directly. No test needs the real results.
'''

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

from lplearn.experiments import runner
from lplearn.experiments.runner import StageError
from lplearn.experiments.stages import figures as figures_stage
from lplearn.visualizations import paper_figures as pf


REPO = Path(__file__).resolve().parents[1]
K5_LABELS = tuple('ABCDEFGHIJK')
BOUND_REGIMES = (('lemma2', 'Lemma 1 regime'), ('grid2e-05', 'Experimental regime'))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261008)


@pytest.fixture
def threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, '_make_executor', lambda workers: ThreadPoolExecutor(max_workers=workers))


def random_policy(rng: np.random.Generator, T: int, S: int) -> np.ndarray:
    '''
    An occupation measure y[t, s, u] that leaks mass per stage and has a small negative entry, like learned solutions.
    '''

    mass = rng.uniform(0.05, 1.0, size=(T, S))
    mass[0, 1:] = 0.0
    pi = rng.dirichlet(np.ones(4), size=(T, S))
    y = 0.97 * mass[:, :, None] * pi
    y[min(1, T - 1), min(2, S - 1), 3] = -1e-4
    return y


def dual_path(rng: np.random.Generator, T: int, scale: float) -> np.ndarray:
    return np.linspace(0.2, 0.6, T)[None, :] * np.ones((2, 1)) + scale * rng.normal(size=(2, T))


def write_lp_record(directory: Path, name: str, job: dict, arrays: dict) -> None:
    '''
    One lp_solutions job: the record JSON and the arrays NPZ, as the runner writes them.
    '''

    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{name}.json').write_text(json.dumps({'job': job, 'status': 'ok', 'job_id': name}), encoding='utf-8')
    np.savez(directory / f'{name}.npz', **arrays)


def write_synthetic_results(root: Path, rng: np.random.Generator) -> None:
    '''
    Writes every input of the figures under ``root`` with the schemas of the stages.
    '''

    (root / 'learners').mkdir(parents=True, exist_ok=True)
    learned = {}
    for case in pf.CASES:
        for label in K5_LABELS[2:]:
            for seed in (0, 1):
                learned[f'{case}/{label}/seed{seed}/y_avg'] = random_policy(rng, 10, 6)
                learned[f'{case}/{label}/seed{seed}/y_last'] = random_policy(rng, 10, 6)
                learned[f'{case}/{label}/seed{seed}/lam_res_avg'] = dual_path(rng, 10, 0.05)
                learned[f'{case}/{label}/seed{seed}/lam_res_last'] = dual_path(rng, 10, 0.05)
    np.savez(root / 'learners' / 'solutions_K5T10.npz', **learned)
    np.savez(root / 'learners' / 'solutions_K11T30.npz', **{
        'CJS/D/seed0/y_avg': random_policy(rng, 30, 12),
        'CJS/K11/seed0/y_avg': random_policy(rng, 30, 12),
        'CJS/P2/seed0/y_avg': random_policy(rng, 30, 12),
    })
    for case in pf.CASES:
        matrix = rng.uniform(0.3, 2.0, size=(11, 11))
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0.0)
        np.savez(root / 'learners' / f'pairwise_{case}.npz', labels=np.array(K5_LABELS), seed_mean=matrix, seed0=matrix)

    lp_dir = root / 'lp_solutions'
    for case in pf.CASES:
        for kind, seed_name in (('lp', 'A'), ('lp_reg', 'B')):
            job = {'instance': 'K5T10', 'K': 5, 'T': 10, 'case': case, 'kind': kind, 'partition': 'none'}
            arrays = {'y': random_policy(rng, 10, 6), 'lam_res': dual_path(rng, 10, 0.0)}
            write_lp_record(lp_dir, f'{case}_{seed_name}', job, arrays)
    write_lp_record(lp_dir, 'K11_B', {'instance': 'K11T30', 'K': 11, 'T': 30, 'case': 'CJS', 'kind': 'lp_reg',
                                      'partition': 'none'}, {'y': random_policy(rng, 30, 12), 'lam_res': dual_path(rng, 30, 0.0)})

    write_finite_n(root / 'finite_n' / 'finite_n_summary.csv')
    write_bound(root / 'bound' / 'bound_rows.csv', rng)
    write_commons(root / 'commons' / 'commons_summary.csv')
    write_commons_stages(root / 'commons' / 'commons_stages.csv')


def finite_n_table() -> pd.DataFrame:
    '''
    Rows in the layout of the stage's finite_n_summary.csv (one row per case, method, policy and N).
    '''

    rows = []
    for case in pf.CASES:
        for method in ('A', 'B', 'E', 'E-LP'):
            for policy in ('projection', 'lp_update'):
                for N in (10, 100, 1000):
                    mean = 0.8 + 0.4 / np.sqrt(N) + 0.01 * K5_LABELS.index(method[0]) + (0.02 if policy == 'lp_update' else 0.0)
                    rows.append({'case': case, 'method': method, 'policy': policy, 'mechanism': 'StrictBudget', 'N': N,
                                 'n_reps': 40, 'cost_mean': mean, 'cost_ci_lo': mean - 0.01, 'cost_ci_hi': mean + 0.01,
                                 'lp_objective': 0.8})
    return pd.DataFrame(rows)


def write_finite_n(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    finite_n_table().to_csv(path, index=False)


def bound_table(rng: np.random.Generator) -> pd.DataFrame:
    '''
    Rows in the layout of the stage's bound_rows.csv (the columns the figures read are among them).
    '''

    rows = []
    for case in pf.CASES:
        for regime, _ in BOUND_REGIMES:
            for convention in ('paper', 'consistent'):
                for partition in 'EFGHIJK':
                    true = 10 ** rng.uniform(-3, -1)
                    spectral = true * rng.uniform(1.5, 20.0)
                    lemma = true * rng.uniform(2.0, 8.0) if regime == 'lemma2' else np.nan
                    rows.append({'instance': 'queue', 'K': 5, 'T': 10, 'case': case, 'label': partition,
                                 'partition': '{0}{1,2}{3}', 'convention': convention, 'regime': regime, 'family': 'lemma',
                                 'dist_total': true, 'B_lemma1': lemma, 'B_spec': spectral,
                                 'B_best': spectral if np.isnan(lemma) else min(lemma, spectral), 'check_ok': 1})
    rows.append({'instance': 'queue', 'K': 11, 'T': 30, 'case': 'CJS', 'label': 'K11', 'partition': '{0}', 'convention': 'paper',
                 'regime': 'lemma2', 'family': 'lemma', 'dist_total': 0.1, 'B_lemma1': 0.3, 'B_spec': 0.5, 'B_best': 0.3,
                 'check_ok': 1})
    return pd.DataFrame(rows)


def write_bound(path: Path, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bound_table(rng).to_csv(path, index=False)


def commons_table() -> pd.DataFrame:
    '''
    Rows in the layout of the stage's commons_summary.csv (one per scenario, case, N, mechanism, tie-break, population).
    '''

    rows = []
    for mechanism in ('M0', 'M1'):
        for population, cost in (('coord', 1.0), ('indep', 1.3), ('selfish', 1.4), ('price', 1.2)):
            rows.append({'scenario': 'base', 'case': 'CJS', 'N': 10, 'mechanism': mechanism, 'tie_break': 'no_request',
                         'population': population, 'reps': 20, 'cost_mean': cost, 'cost_ci95': 0.05})
    return pd.DataFrame(rows)


def write_commons(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    commons_table().to_csv(path, index=False)


def commons_stage_table() -> pd.DataFrame:
    '''
    Stage rows of the no-coordinator experiment: requests divided by budget per stage (the table the commons stage
    is asked to write, see reports/figures/README.md).
    '''

    rows = []
    for population, level in (('coord', 0.9), ('indep', 1.3), ('selfish', 1.2)):
        for stage in range(10):
            for metric, value in (('ratio_admission', level), ('ratio_high_rate', level - 0.05)):
                rows.append({'case': 'CJS', 'N': 10, 'mechanism': 'M0', 'tie_break': 'no_request', 'population': population,
                             'stage': stage, 'metric': metric, 'mean': value, 'ci_low': value - 0.02, 'ci_high': value + 0.02})
    return pd.DataFrame(rows)


def write_commons_stages(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    commons_stage_table().to_csv(path, index=False)


def dual_frame(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for case in pf.CASES:
        for method in ('B', 'C'):
            for resource in ('admission', 'high_rate'):
                for t in range(8):
                    rows.append({'case': case, 'method': method, 'resource': resource, 't': t,
                                 'mean': 0.1 * t + rng.random(), 'sd': 0.05})
    return pd.DataFrame(rows)


def assert_embedded_pdf(path: Path) -> None:
    data = path.read_bytes()
    assert data.startswith(b'%PDF'), path
    assert len(data) > 2000, path
    assert b'/FontFile2' in data, f'{path} does not embed TrueType fonts'
    assert b'CreationDate' not in data, f'{path} carries a creation date, so it is not reproducible'


# ---------------------------------------------------------------------------
# Occupation measures and pies
# ---------------------------------------------------------------------------


def test_layout_round_trip_and_action_order(rng):
    y = rng.random((7, 4, 4))
    legacy = pf.to_legacy_layout(y)
    assert legacy.shape == (4, 2, 2, 7)
    assert np.array_equal(pf.from_legacy_layout(legacy), y)
    assert np.array_equal(pf.to_legacy_layout(pf.from_legacy_layout(legacy)), legacy)
    t, s = 5, 2
    assert legacy[s, 1, 0, t] == y[t, s, 2]  # u = 2a + b with a = 1, b = 0
    assert legacy[s, 0, 1, t] == y[t, s, 1]  # a = 0, b = 1


def test_layout_conversions_do_not_alias_their_input(rng):
    y = rng.random((3, 2, 4))
    before = y.copy()
    legacy = pf.to_legacy_layout(y)
    legacy[...] = 0.0
    assert np.array_equal(y, before)
    legacy_in = y.reshape(3, 2, 2, 2).transpose(1, 2, 3, 0).copy()  # a legacy array of shape (S, 2, 2, T)
    legacy_before = legacy_in.copy()
    back = pf.from_legacy_layout(legacy_in)
    back[...] = 0.0
    assert np.array_equal(legacy_in, legacy_before)


def test_pie_wedges_follow_the_action_order():
    y = np.zeros((3, 2, 4))
    y[1, 0] = [0.4, 0.1, 0.3, 0.2]  # the only mass of stage 1 sits in state 0
    original = y.copy()
    ax = Figure().add_subplot(111)
    pf.draw_policy_pies(ax, y, pie_fontsize=8, label_fontsize=9)
    wedges = [patch for patch in ax.patches if isinstance(patch, Wedge)]
    spans = [(w.theta2 - w.theta1) / 360 for w in wedges]
    assert np.allclose(spans, [0.4, 0.1, 0.3, 0.2])
    for k, wedge in enumerate(wedges):
        assert to_rgba(wedge.get_facecolor()) == to_rgba(pf.PIE_COLORS[k])
    assert np.array_equal(y, original)  # the caller's array is not clipped by the helper


def test_schema_and_shape_errors_name_the_problem(rng):
    plot_bound = pd.DataFrame({'case': ['CJS'], 'partition': ['E'], 'convention': ['paper'], 'regime': ['lemma2'],
                               'true_error': [0.1], 'bound': [0.3]})
    with pytest.raises(pf.FigureInputError, match='missing columns'):
        pf.plot_bound_validation(plot_bound.drop(columns=['bound']), partitions=('E',), regimes=BOUND_REGIMES)
    bad = finite_n_table()
    bad.loc[0, 'cost_mean'] = bad.loc[0, 'cost_ci_hi'] + 1.0
    with pytest.raises(pf.FigureInputError, match='outside'):
        pf.plot_finite_n_costs(bad, methods=('A', 'B', 'E'))
    with pytest.raises(pf.FigureInputError, match='shape'):
        pf.plot_heatmap_policies_compare(list('ABC'), np.zeros((3, 3)), np.zeros((4, 4)))
    with pytest.raises(pf.FigureInputError, match=r'\(T, S, 4\)'):
        pf.plot_policy_solutions_agg([pf.PolicyPanel('x', np.zeros((3, 4)))], [pf.PolicyPanel('x', np.zeros((3, 4)))])


# ---------------------------------------------------------------------------
# Every figure is drawn as an embedded, reproducible PDF
# ---------------------------------------------------------------------------


def test_every_figure_function_writes_an_embedded_pdf(tmp_path: Path, rng):
    panels = [pf.PolicyPanel(f'panel {k}', random_policy(rng, 10, 6)) for k in range(4)]
    k11 = [pf.PolicyPanel(f'K11 {k}', random_policy(rng, 30, 12)) for k in range(4)]
    labels = list(K5_LABELS)
    cjs = rng.uniform(0.2, 2.0, size=(11, 11))
    cjs = (cjs + cjs.T) / 2
    np.fill_diagonal(cjs, 0.0)
    stages = commons_stage_table()
    costs = pd.DataFrame({'mechanism': ['M0', 'M0', 'M0'], 'population': ['coord', 'indep', 'selfish'],
                          'cost_mean': [1.0, 1.3, 1.4], 'cost_ci_lo': [0.95, 1.25, 1.35], 'cost_ci_hi': [1.05, 1.35, 1.45]})
    drawings = {
        'policy_solutions_non_agg': lambda out: pf.plot_policy_solutions_non_agg(panels, panels, out),
        'policy_solutions_agg': lambda out: pf.plot_policy_solutions_agg(panels, panels, out),
        'policies_K11T30': lambda out: pf.plot_policies_K11T30(k11, out),
        'heatmap_policies_compare': lambda out: pf.plot_heatmap_policies_compare(labels, cjs, cjs, out),
        'finite_n_costs': lambda out: pf.plot_finite_n_costs(finite_n_table(), out, methods=('A', 'B', 'E')),
        'bound_validation': lambda out: pf.plot_bound_validation(
            _plot_bound_frame(rng), out,
            partitions=('E', 'F', 'G'), regimes=BOUND_REGIMES),
        'dual_prices': lambda out: pf.plot_dual_prices(dual_frame(rng), out, methods=('B', 'C')),
        'commons_overload': lambda out: pf.plot_commons_overload(stages, costs, out, populations=('coord', 'indep', 'selfish'),
                                                                 mechanisms=('M0',)),
    }
    assert set(drawings) == set(pf.FIGURE_NAMES)
    for name, draw in drawings.items():
        out = tmp_path / f'{name}.pdf'
        figure = draw(out)
        assert isinstance(figure, Figure)
        assert_embedded_pdf(out)


def _plot_bound_frame(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for case in pf.CASES:
        for regime, _ in BOUND_REGIMES:
            for convention in ('paper', 'consistent'):
                for partition in 'EFG':
                    true = 10 ** rng.uniform(-3, -1)
                    rows.append({'case': case, 'partition': partition, 'convention': convention, 'regime': regime,
                                 'true_error': true, 'bound': true * 3 if partition != 'G' else np.inf})
    return pd.DataFrame(rows)


def test_pdf_output_is_byte_reproducible(tmp_path: Path, rng):
    panels = [pf.PolicyPanel('p', random_policy(rng, 10, 6))] * 4
    first, second = tmp_path / 'first.pdf', tmp_path / 'second.pdf'
    pf.plot_policy_solutions_non_agg(panels, panels, first)
    pf.plot_policy_solutions_non_agg(panels, panels, second)
    assert first.read_bytes() == second.read_bytes()


# ---------------------------------------------------------------------------
# Loading the recorded results and the driver
# ---------------------------------------------------------------------------


def test_build_figures_draws_every_figure_from_recorded_inputs(tmp_path: Path, rng):
    results, out = tmp_path / 'results', tmp_path / 'figures'
    write_synthetic_results(results, rng)
    run = pf.build_figures(results, out)
    assert not run.failed, run.failed
    assert set(run.written) == set(pf.FIGURE_NAMES)
    for name in pf.FIGURE_NAMES:
        assert_embedded_pdf(out / f'{name}.pdf')
    labels = {entry['label'] for entry in run.written['policy_solutions_non_agg']['inputs']}
    assert {'solutions_K5T10', 'lp_solutions/K5T10/CJS/lp', 'lp_solutions/K5T10/CJS/lp_reg'} <= labels
    assert {entry['label'] for entry in run.written['heatmap_policies_compare']['inputs']} == {'pairwise_CJS', 'pairwise_SJS'}
    assert run.written['commons_overload']['inputs'][0]['sha256']


def test_missing_inputs_are_reported_and_existing_pdfs_preserved(tmp_path: Path):
    out = tmp_path / 'figures'
    out.mkdir()
    (out / 'finite_n_costs.pdf').write_bytes(b'stale')
    run = pf.build_figures(tmp_path / 'empty-results', out)
    assert not run.written
    assert set(run.failed) == set(pf.FIGURE_NAMES)
    assert all(message.startswith('FigureInputError') for message in run.failed.values()), run.failed
    assert not run.tracebacks  # missing inputs are expected failures, not bugs
    assert (out / 'finite_n_costs.pdf').read_bytes() == b'stale'
    assert set(run.preserved) == {'finite_n_costs'}


def test_save_failure_preserves_previous_pdf(tmp_path: Path, monkeypatch):
    out = tmp_path / 'existing.pdf'
    out.write_bytes(b'previous')
    figure = Figure()

    def broken_save(path, **kwargs):
        Path(path).write_bytes(b'partial')
        raise OSError('simulated interrupted save')

    monkeypatch.setattr(figure, 'savefig', broken_save)
    with pytest.raises(OSError, match='simulated interrupted'):
        pf.save_figure(figure, out)
    assert out.read_bytes() == b'previous'
    assert sorted(tmp_path.iterdir()) == [out]


def test_infeasible_occupation_is_labeled_on_figure(rng):
    panel = pf.PolicyPanel('learned', random_policy(rng, 3, 3))
    fig = pf.plot_policy_solutions_non_agg([panel], [panel])
    assert all('Infeasible iterate (clipped)' in ax.get_title() for ax in fig.axes)


def test_a_partial_result_root_draws_only_the_figures_it_can(tmp_path: Path):
    results, out = tmp_path / 'results', tmp_path / 'figures'
    write_finite_n(results / 'finite_n' / 'finite_n_summary.csv')
    run = pf.build_figures(results, out, names=['finite_n_costs', 'bound_validation'])
    assert set(run.written) == {'finite_n_costs'}
    assert 'bound_rows' in run.failed['bound_validation']
    assert_embedded_pdf(out / 'finite_n_costs.pdf')


def test_commons_figure_names_the_missing_stage_table(tmp_path: Path, rng):
    results = tmp_path / 'results'
    write_commons(results / 'commons' / 'commons_summary.csv')
    run = pf.build_figures(results, tmp_path / 'figures', names=['commons_overload'])
    assert 'commons_stages' in run.failed['commons_overload']


def test_lp_solutions_records_are_keyed_by_kind_and_partition(tmp_path: Path, rng):
    results = tmp_path / 'results'
    write_synthetic_results(results, rng)
    write_lp_record(results / 'lp_solutions', 'duplicate', {'instance': 'K5T10', 'K': 5, 'T': 10, 'case': 'CJS', 'kind': 'lp',
                                                            'partition': 'none'}, {'y': random_policy(rng, 10, 6),
                                                                                   'lam_res': dual_path(rng, 10, 0.0)})
    run = pf.build_figures(results, tmp_path / 'figures', names=['policy_solutions_non_agg'])
    assert 'expected one lp_solutions record for CJS/A' in run.failed['policy_solutions_non_agg']


def test_options_refuse_unknown_keys_and_bad_values():
    with pytest.raises(ValueError, match='unknown figure option'):
        pf.options_from_mapping({'learned_fiel': 'y_avg'})
    with pytest.raises(ValueError, match='unknown figure input'):
        pf.options_from_mapping({'inputs': {'bound_row': 'x.csv'}})
    with pytest.raises(ValueError, match='learned_field'):
        pf.options_from_mapping({'learned_field': 'y_mean'})
    with pytest.raises(ValueError, match='distance_key'):
        pf.options_from_mapping({'distance_key': 'mean'})
    settings, inputs = pf.options_from_mapping({'learned_field': 'y_last', 'inputs': {'bound_rows': 'b/x.csv'},
                                                'bound_regimes': [['lemma2', 'L']]})
    assert settings.learned_field == 'y_last' and inputs.bound_rows == 'b/x.csv'
    assert settings.bound_regimes == (('lemma2', 'L'),)


def test_build_figures_refuses_unknown_names(tmp_path: Path):
    with pytest.raises(ValueError, match='unknown figure'):
        pf.build_figures(tmp_path, tmp_path / 'out', names=['not_a_figure'])


# ---------------------------------------------------------------------------
# The figures stage
# ---------------------------------------------------------------------------


def test_stage_has_one_job_and_refuses_unknown_figures():
    jobs = figures_stage.make_jobs({})
    assert len(jobs) == 1 and jobs[0]['figures'] == list(pf.FIGURE_NAMES)
    with pytest.raises(StageError, match='unknown figure'):
        figures_stage.make_jobs({'figures': ['nope']})
    assert figures_stage.run_job(jobs[0])['n_figures'] == len(pf.FIGURE_NAMES)


def test_stage_run_then_summarize_redraws_from_results(tmp_path: Path, rng, threads):
    results, out = tmp_path / 'results', tmp_path / 'figures'
    write_finite_n(results / 'finite_n' / 'finite_n_summary.csv')
    mapping = {'results_root': str(results), 'out_dir': str(out), 'figures': ['finite_n_costs']}
    report = runner.run_stage('figures', mapping, results_root=tmp_path / 'stage-results', workers=1)
    assert report.status == 'ok'
    summary = runner.summarize_stage('figures', mapping, results_root=tmp_path / 'stage-results', out_dir=tmp_path / 'summary')
    assert_embedded_pdf(out / 'finite_n_costs.pdf')
    assert (summary / figures_stage.MANIFEST_NAME).is_file()
    # The results change: the next summarize draws the bound figure as well.
    write_bound(results / 'bound' / 'bound_rows.csv', rng)
    mapping['figures'] = ['finite_n_costs', 'bound_validation']
    runner.summarize_stage('figures', mapping, results_root=tmp_path / 'stage-results', out_dir=tmp_path / 'summary')
    assert_embedded_pdf(out / 'bound_validation.pdf')


def test_stage_summarize_draws_the_rest_and_then_fails(tmp_path: Path):
    results, out = tmp_path / 'results', tmp_path / 'figures'
    write_finite_n(results / 'finite_n' / 'finite_n_summary.csv')
    summary_dir = tmp_path / 'summary'
    summary_dir.mkdir()
    with pytest.raises(StageError, match='bound_validation'):
        figures_stage.summarize([], {'results_root': str(results), 'out_dir': str(out),
                                     'figures': ['finite_n_costs', 'bound_validation']}, summary_dir)
    assert_embedded_pdf(out / 'finite_n_costs.pdf')
    assert not (out / 'bound_validation.pdf').exists()
    manifest = json.loads((summary_dir / figures_stage.MANIFEST_NAME).read_text(encoding='utf-8'))
    assert set(manifest['written']) == {'finite_n_costs'} and set(manifest['failed']) == {'bound_validation'}


def test_repository_config_is_valid_and_smoke_redirects_outputs():
    loaded = runner.load_config(REPO / 'configs' / 'figures.toml')
    full = runner.effective_config(loaded.raw, figures_stage.SMOKE, smoke=False)
    settings, _ = pf.options_from_mapping(full)
    assert settings.figures == pf.FIGURE_NAMES
    assert full['out_dir'] == 'papers/SIGMETRICS/figures/generated'
    smoke = runner.effective_config(loaded.raw, figures_stage.SMOKE, smoke=True)
    assert smoke['results_root'] == 'results/smoke'
    assert smoke['out_dir'] == 'results/smoke/figures'
    pf.options_from_mapping(smoke)
