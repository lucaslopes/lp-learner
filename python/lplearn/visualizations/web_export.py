'''
Data for the high-school web app (``web/``): the model, exact LP plans on a grid of budgets, the recorded results the
page quotes, and reference values that the app's TypeScript engine is tested against.

Two JSON files are written (stage ``web``, ``configs/web.toml``):

- ``model.json``: the base parameters; for each (alpha, beta) of the grid, the LP plan y[t][s][u], its
  prices lam[l][t] and value (method A, interior point); and ``results``: the numbers of the tutorial's tables (cost
  per booth of the coordinator and the selfish booths, social-dilemma tests, what-if scenarios, coordinators), read
  from the recorded summaries, plus the large-N lost-car curves. The app shows these as "results from our simulations";
  everything else it shows is computed live in the browser.
- ``reference.json``: Python outputs for the engine tests (kernels, backward induction for both tie-breaks
  and with LP prices, large-N evaluations under every mechanism, the LP conditional policy, and GDA from a zero start).

Conventions match ``lplearn.core``: actions u = 2a + b, P[u][s][s'], c[s][u], resources 0 = cars (admission) and
1 = attendants (high rate). The toll plaza is a pedagogical analogy.
'''


from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lplearn.commons.experiment import solve_method_a
from lplearn.commons.mechanisms import fluid_mechanism
from lplearn.commons.selfish import deterministic_policy, priced_solution, selfish_solution
from lplearn.core.gda import run_gda
from lplearn.core.instance import QueueParams, queue_wcmdp, transition_kernel
from lplearn.core.policy import conditional_policy
from lplearn.experiments.io import atomic_write_text
from lplearn.visualizations import tutorial_assets as ta
from lplearn.workflow import occupation_feasibility


DEFAULT_ALPHAS = (0.5, 0.625, 0.75, 0.875, 1.0)
DEFAULT_BETAS = (0.1, 0.25, 0.4, 0.5, 0.75, 1.0)
GDA_REFERENCE_ITERATIONS = 3000


def _r(x: Any, digits: int = 10) -> Any:
    '''Nested lists of floats rounded to `digits` significant digits (JSON-friendly; -0.0 becomes 0.0).'''
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        v = float(arr)
        if not math.isfinite(v):
            raise ValueError('non-finite value in web export input or output')
        out = float(f'{v:.{digits}g}')
        return 0.0 if out == 0 else out
    return [_r(v, digits) for v in arr]


def params_dict(p: QueueParams) -> dict[str, Any]:
    return {'K': p.K, 'T': p.T, 'p': p.p, 'bLow': p.b_low, 'bHigh': p.b_high, 'alpha': p.alpha, 'beta': p.beta,
            'gamma': p.gamma, 'gammaBar': p.gamma_bar}


def lp_grid(base: QueueParams, alphas, betas) -> list[dict[str, Any]]:
    out = []
    for alpha in alphas:
        for beta in betas:
            p = QueueParams(**{**base.to_dict(), 'alpha': float(alpha), 'beta': float(beta)})
            m = queue_wcmdp(p)
            lp = solve_method_a(m)
            raw = occupation_feasibility(lp['y'], m)
            if not raw['feasible']:
                raise ValueError(f'LP occupation infeasible before web export: {raw}')
            y = _r(np.maximum(lp['y'], 0.0), 8)
            exported = occupation_feasibility(y, m)
            if not exported['feasible']:
                raise ValueError(f'LP occupation infeasible after web export rounding: {exported}')
            if not np.isfinite(lp['objective']) or not np.isfinite(lp['lam_res']).all():
                raise ValueError('non-finite LP objective or multipliers')
            out.append({'alpha': float(alpha), 'beta': float(beta), 'value': _r(lp['objective']),
                        'y': y, 'lam': _r(np.maximum(lp['lam_res'], 0.0), 8),
                        'occupationValidation': {'raw': raw, 'exported': exported}})
    return out


def _ci(row) -> dict[str, float]:
    return {'mean': _r(row['cost_mean'], 8), 'ci': _r(row['cost_ci95'], 6)}


def results(tm: ta.TutorialModel, data: ta.TutorialData) -> dict[str, Any]:
    '''The recorded results the page quotes (all CJS, from the tutorial's summaries).'''
    p = tm.params
    main = []
    for N in (10, 100, 1000):
        row = {'N': N, 'coord': _ci(ta.pop_row(data, 'coord', 'M1', N))}
        for mech, name in (('M1', 'rationing'), ('M2', 'congestion')):
            sel = ta.pop_row(data, 'selfish', mech, N)
            pr = ta.paired_row(data, 'selfish_minus_coord', mech, N)
            row[name] = {**_ci(sel), 'diff': _r(pr['diff_mean'], 8), 'diffCi': _r(pr['diff_ci95'], 6),
                         'pct': _r(pr['pct_of_coord'], 6)}
        main.append(row)
    detail = {}
    for key, pop, mech in (('coord', 'coord', 'M1'), ('selfishRationing', 'selfish', 'M1'),
                           ('selfishCongestion', 'selfish', 'M2'), ('priceTaking', 'price', 'M1')):
        r = ta.pop_row(data, pop, mech, 1000)
        detail[key] = {**_ci(r), 'carRequestsPerCar': _r(r['requested_admit'] / p.alpha, 6),
                       'attRequestsPerAtt': _r(r['requested_fast'] / p.beta, 6),
                       'overloadCars': _r(r['overload_admit'], 6), 'overloadAtt': _r(r['overload_fast'], 6),
                       'fullLanes': _r(r['full_mean_per_stage'], 6), 'lostCars': _r(ta.lost_cars(r, p), 6)}
    dilemma = {}
    for mech, name in (('M1', 'rationing'), ('M2', 'congestion')):
        pr = ta.paired_row(data, 'selfish_minus_coord', mech, 1000)
        devs = {}
        for (label, test), key in zip(ta.DEVIATION_TESTS, ('neverAttendant', 'followCoordinator', 'neverGate')):
            row = ta.dilemma_row(data, test, mech, 1000)
            devs[key] = {'diff': _r(row['diff_mean'], 6), 'ci': _r(row['diff_ci95'], 6)}
        verdict = ta.dilemma_row(data, 'verdict: social dilemma (collective worse and individual rational)', mech, 1000)
        dilemma[name] = {'collective': {'diff': _r(pr['diff_mean'], 6), 'ci': _r(pr['diff_ci95'], 6)},
                         'deviations': devs, 'verdict': str(verdict['verdict'])}
    whatif = []
    for scenario, label, _ in ta.WHATIF:
        coord = ta.pop_row(data, 'coord', 'M1', 1000, scenario=scenario)
        entry = {'scenario': scenario, 'label': label, 'coord': _ci(coord)}
        for mech, name in (('M1', 'rationing'), ('M2', 'congestion')):
            entry[name] = _r(ta.paired_row(data, 'selfish_minus_coord', mech, 1000, scenario=scenario)['pct_of_coord'], 6)
        whatif.append(entry)
    coordinators = []
    lp_value = float(ta.finite_n_row(data, 'A', 'projection', 1000)['lp_objective'])
    for method, policy, label in ta.COORDINATORS:
        coordinators.append({'label': label, 'costs': {str(N): _r(ta.finite_n_row(data, method, policy, N)['cost_mean'], 8)
                                                       for N in (10, 100, 1000)}})
    lost = {key: _r(np.cumsum(ta.fluid_lost_cars(tm, key)), 8) for key in ('lp', 'selfish_M1', 'selfish_M2')}
    # learned coordinators (finite_n stage, 1000 booths): share of booths repaired per period, and the mass that the
    # learned occupation measure keeps at the last period
    learning = {name: {'repairsPerPeriod': _r(ta.finite_n_row(data, method, 'projection', 1000)['repairs_frac'], 6),
                       'massFinal': _r(ta.finite_n_row(data, method, 'projection', 1000)['source_mass_final'], 6)}
                for method, name in (('A', 'lp'), ('C', 'gda'), ('D', 'sgda'))}
    sensitivity = []
    for scenario in ('base', 'beta=0.10', 'beta=0.50'):
        pr = ta.paired_row(data, 'selfish_minus_coord', 'M1', 100, scenario=scenario)
        sensitivity.append({'scenario': scenario, 'N': 100, 'rationingPct': _r(pr['pct_of_coord'], 6)})
    for entry in whatif:
        rows = data.fluid[(data.fluid['scenario'] == entry['scenario']) & (data.fluid['case'] == 'CJS')]
        if rows.empty:
            raise ta.TutorialInputError(f"commons_fluid: no rows for scenario {entry['scenario']}")
        entry['lpValue'] = _r(rows['lp_value'].iloc[0], 8)  # the LP value does not depend on N, the rule or the tie-break
    # The original frontend used a hard-coded value of 2,000. Derive it from
    # the supplied tables instead; do not attribute synthetic or mixed inputs
    # to a fixed historical experiment. Verdict rows can carry zero reps.
    counts = set()
    for frame, column in ((data.summary, 'reps'), (data.paired, 'reps'),
                          (data.dilemma, 'reps'), (data.finite_n, 'n_reps')):
        if column not in frame:
            raise ta.TutorialInputError(f'missing input column {column!r} for replication provenance')
        for n in frame[column].dropna():
            value = float(n)
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ta.TutorialInputError(f'invalid replication count {n!r}')
            if value:
                counts.add(int(value))
    if len(counts) != 1:
        raise ta.TutorialInputError(f'expected one positive replication count across summaries; found {sorted(counts)}')
    return {'lpValue': _r(lp_value, 10), 'main': main, 'detail': detail, 'dilemma': dilemma, 'whatif': whatif,
            'coordinators': coordinators, 'fluidLostCumulative': lost, 'learning': learning, 'sensitivity': sensitivity,
            'unavoidableMissed': _r((1 - p.alpha) * p.T * p.gamma, 8), 'carsPerBooth': _r(p.alpha * p.T, 8),
            'replications': next(iter(counts)),
            'replicationCounts': sorted(counts),
            'source': 'Supplied recorded summaries; see input paths and hashes in inputs.'}


def reference(tm: ta.TutorialModel) -> dict[str, Any]:
    '''Python outputs that the TypeScript engine must reproduce.'''
    p, m = tm.params, tm.model
    lp_validation = occupation_feasibility(tm.lp['y'], m)
    if not lp_validation['feasible']:
        raise ValueError(f'LP reference occupation infeasible: {lp_validation}')
    out: dict[str, Any] = {'params': params_dict(p)}
    out['kernelLow'] = _r(transition_kernel('CJS', p.K, p.p, p.b_low), 14)
    out['kernelHigh'] = _r(transition_kernel('CJS', p.K, p.p, p.b_high), 14)
    out['kernelMid'] = {'q': 0.05, 'P': _r(transition_kernel('CJS', p.K, p.p, 0.05), 14)}
    out['P'] = _r(m.P, 14)
    out['c'] = _r(m.c, 14)
    for tie in ('no_request', 'request'):
        sol = selfish_solution(m, tie)
        out[f'selfish_{tie}'] = {'V': _r(sol.V, 12), 'u': sol.u.astype(int).tolist()}
    lam = np.asarray(tm.lp['lam_res'], dtype=float)
    out['lpLam'] = _r(lam, 12)
    out['lpY'] = _r(tm.lp['y'], 12)
    out['lpValue'] = _r(tm.lp['objective'], 12)
    out['lpOccupationValidation'] = lp_validation
    out['piLp'] = _r(conditional_policy(tm.lp['y']), 12)
    price = priced_solution(m, lam, 'no_request')
    out['priced_no_request'] = {'u': price.u.astype(int).tolist()}
    fluid = {}
    sol = selfish_solution(m, 'no_request')
    for mech in ('M0', 'M1', 'M2'):
        fl = fluid_mechanism(p, m, deterministic_policy(sol.u), mech)
        fluid[f'selfish_{mech}'] = {'cost': _r(fl['cost'], 12), 'mass': _r(fl['mass'], 12), 'grantA': _r(fl['grant_a'], 12),
                                    'grantB': _r(fl['grant_b'], 12), 'qHigh': _r(fl['q_high'], 12)}
    fl = fluid_mechanism(p, m, conditional_policy(tm.lp['y']), 'M1')
    fluid['lp_M1'] = {'cost': _r(fl['cost'], 12), 'mass': _r(fl['mass'], 12)}
    out['fluid'] = fluid
    gda = run_gda(m, 1.0, 2e-5, 2e-5, 0.9, 0.03, GDA_REFERENCE_ITERATIONS, 0, seed=0, avg_window=1, init='zeros')
    for name, value in gda.items():
        if isinstance(value, (np.ndarray, float, np.floating)) and not np.isfinite(value).all():
            raise ValueError(f'non-finite GDA reference result in {name}')
    gda_validation = occupation_feasibility(gda['y_last'], m)
    out['gda'] = {'nIter': GDA_REFERENCE_ITERATIONS, 'GammaY': 1.0, 'GammaLambda': 2e-5, 'GammaMu': 2e-5, 'etaY': 0.9,
                  'etaDual': 0.03, 'y': _r(gda['y_last'], 12), 'lamRes': _r(gda['lam_res_last'], 12),
                  'mu0': _r(gda['mu0_last'], 12), 'muf': _r(gda['muf_last'], 12),
                  'kind': 'iterate' if gda_validation['feasible'] else 'infeasible_iterate_diagnostic',
                  'occupationValidation': gda_validation}
    return out


def build_web_data(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    results_root = Path(cfg.get('results_root', 'results'))
    tm = ta.solve_tutorial_model(ta.base_params(cfg.get('model', {})))
    data = ta.load_data(results_root / str(cfg.get('commons_summary_dir', 'commons/summary_tutorial')),
                        results_root / str(cfg.get('finite_n_summary', 'finite_n/summary/finite_n_summary.csv')))
    recorded_value = float(ta.finite_n_row(data, 'A', 'projection', 1000)['lp_objective'])
    if not math.isfinite(recorded_value) or not math.isclose(recorded_value, float(tm.lp['objective']), rel_tol=1e-5, abs_tol=1e-4):
        raise ta.TutorialInputError('finite_n LP objective does not match the configured base model')
    alphas = tuple(cfg.get('alphas', DEFAULT_ALPHAS))
    betas = tuple(cfg.get('betas', DEFAULT_BETAS))
    if not alphas or not betas or any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in (*alphas, *betas)):
        raise ValueError('alpha and beta grids must be nonempty and contain only values in [0, 1]')
    if tm.params.alpha <= 0 or tm.params.beta <= 0:
        raise ValueError('base alpha and beta must be positive to compute request ratios')
    model = {
        'generatedBy': 'lplearn experiments summarize web (lplearn/visualizations/web_export.py); do not edit by hand',
        'params': params_dict(tm.params),
        'gda': {'GammaY': 1.0, 'GammaLambda': 2e-5, 'GammaMu': 2e-5, 'etaY': 0.9, 'etaDual': 0.03},
        'lpGrid': lp_grid(tm.params, alphas, betas),
        'results': results(tm, data),
        'inputs': data.files,
    }
    return model, reference(tm)


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    atomic_write_text(path, json.dumps(obj, separators=(',', ':'), sort_keys=False, allow_nan=False) + '\n')
    return path
