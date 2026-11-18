'''
Stage ``commons``: selfish queues against the LP-coordinated policy, with no coordinator.

A job is one chunk of replications (``chunk``, ``rep_start``, ``n_reps``) of one setting: scenario (base or a
sensitivity entry), case (CJS or SJS), queue count N, mechanism (M0 unenforced, M1 rationing, M2 congestion) and
tie-break for exact ties. Every job runs all populations of ``lplearn.commons.experiment`` on the same
replications (common random numbers), so the summaries are paired.

``summarize`` writes, into the stage's summary directory:

- commons_summary.csv    per population: cost per queue (mean, 95% CI), shadow-priced cost, overload frequency
                         and excess of admission and fast-mode requests, denials, degradations, queue statistics.
- commons_paired.csv     paired differences against the coordinated population, with 95% CIs.
- commons_dilemma.csv    collective test (all-selfish minus coordinated) and individual test (one deviating queue
                         minus a selfish queue, paired, and minus the selfish queues in the same run), plus the
                         fluid (N -> infinity) best-response check.
- commons_incentives.csv private advantages from the DP, the selfish and price-taking actions, and their ties.
- commons_fluid.csv      fluid (N -> infinity) costs, with cost terms and usage, for each setting.
- commons_table.tex      compact table for the manuscript (one panel per case, one row per policy and mechanism).

The shadow-priced cost is an accounting device: cost plus the LP shadow prices times the realised excess of usage
over each budget. It prices violations at the LP's marginal values and is not a cost the model charges. In the
N -> infinity limit, for flow-feasible occupation measures, LP duality makes it at least the LP value; at finite N
the realised occupation is only approximately flow-feasible, so the bound is not a finite-N statement.
'''


from __future__ import annotations

import csv
import itertools
import math
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from lplearn.commons.experiment import COST_TERMS, DETAILED, POPULATIONS, job_params, run_job as _run_job, solve_method_a
from lplearn.commons.selfish import incentive_rows, priced_solution, selfish_solution
from lplearn.core.instance import ADMIT, HIGH, queue_wcmdp
from lplearn.core.policy import conditional_policy
from lplearn.experiments.io import atomic_write_text


NAME = 'commons'
VERSION = '2'  # 2: fail on infeasible LP occupation and persist pre-conversion diagnostics
SMOKE = {'cases': ['CJS'], 'ns': [10], 'mechanisms': ['M0', 'M1', 'M2'], 'tie_breaks': ['no_request'], 'n_chunks': 1, 'reps_per_chunk': 20}

BASE_KEYS = ('K', 'T', 'p', 'b_low', 'b_high', 'alpha', 'beta', 'gamma', 'gamma_bar')
DEFAULTS: dict[str, Any] = {'K': 5, 'T': 10, 'p': 0.7, 'b_low': 0.01, 'b_high': 0.1, 'alpha': 0.75, 'beta': 0.25, 'gamma': 100.0, 'gamma_bar': 100.0}
INT_KEYS = ('K', 'T')
POP_LABELS = {
    'coord': 'coordinated (LP projection)',
    'indep': 'independent LP policy',
    'selfish': 'selfish (private DP)',
    'price': 'price-taking (LP prices)',
    'devnf': 'deviator never fast',
    'devlp': 'deviator LP conditional',
    'devna': 'deviator never admit',
}
DEVIATIONS = ('devnf', 'devlp', 'devna')
Z95 = 1.96


def _setting(entry: dict, cfg: dict, key: str) -> Any:
    value = entry.get(key, cfg.get(key, DEFAULTS[key]))
    return int(value) if key in INT_KEYS else float(value)


def make_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    '''
    Grid of scenarios x cases x N x mechanisms x tie-breaks x chunks. The base scenario uses the top-level keys;
    each entry of `sensitivity` overrides base values and may restrict cases, ns, mechanisms and tie_breaks.
    '''

    seed = int(cfg.get('seed', 2026))
    n_chunks = int(cfg.get('n_chunks', 10))
    per_chunk = int(cfg.get('reps_per_chunk', 200))
    scenarios = [{
        'name': 'base',
        'values': {k: _setting({}, cfg, k) for k in BASE_KEYS},
        'cases': cfg.get('cases', ['CJS', 'SJS']),
        'ns': cfg.get('ns', [10, 100, 1000]),
        'mechanisms': cfg.get('mechanisms', ['M0', 'M1', 'M2']),
        'tie_breaks': cfg.get('tie_breaks', ['no_request', 'request']),
    }]
    for entry in cfg.get('sensitivity', []):
        scenarios.append({
            'name': str(entry['name']),
            'values': {k: _setting(entry, cfg, k) for k in BASE_KEYS},
            'cases': entry.get('cases', cfg.get('cases', ['CJS', 'SJS'])),
            'ns': entry.get('ns', cfg.get('ns', [10, 100, 1000])),
            'mechanisms': entry.get('mechanisms', cfg.get('mechanisms', ['M0', 'M1', 'M2'])),
            'tie_breaks': entry.get('tie_breaks', cfg.get('tie_breaks', ['no_request', 'request'])),
        })
    names = [sc['name'] for sc in scenarios]
    if len(set(names)) != len(names):
        raise ValueError(f'scenario names must be unique, got {names}')

    jobs = []
    for sc in scenarios:
        for case, N, mech, tie, chunk in itertools.product(sc['cases'], sc['ns'], sc['mechanisms'], sc['tie_breaks'], range(n_chunks)):
            jobs.append({
                'scenario': sc['name'],
                'case': str(case),
                'N': int(N),
                'mechanism': str(mech),
                'tie_break': str(tie),
                'chunk': int(chunk),
                'rep_start': int(chunk) * per_chunk,
                'n_reps': per_chunk,
                'seed': seed,
                **sc['values'],
            })
    return jobs


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    '''
    Delegates to lplearn.commons.experiment.run_job (every population on the chunk's replications).
    '''

    return _run_job(job)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _ci(x: np.ndarray) -> tuple[float, float]:
    '''
    Mean and the half-width of its normal 95% confidence interval.
    '''

    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan, math.nan
    if x.size == 1:
        return float(x[0]), math.nan
    return float(x.mean()), float(Z95 * x.std(ddof=1) / math.sqrt(x.size))


def _group_records(records: list[tuple[dict, dict]]) -> dict[tuple, dict[str, Any]]:
    '''
    Concatenate the chunks of each (scenario, case, N, mechanism, tie-break) group along the replication axis.
    '''

    buckets: dict[tuple, list[tuple[int, dict, dict]]] = {}
    for job, arrays in records:
        key = (job['scenario'], job['case'], int(job['N']), job['mechanism'], job['tie_break'])
        buckets.setdefault(key, []).append((int(job['rep_start']), job, arrays))
    groups = {}
    for key, items in buckets.items():
        items.sort(key=lambda item: item[0])
        data: dict[str, Any] = {}
        for name in items[0][2]:
            if '__' in name:
                data[name] = np.concatenate([np.atleast_1d(np.asarray(arrays[name])) for _, _, arrays in items], axis=0)
            else:
                data[name] = items[0][2][name]
        groups[key] = {'job': items[0][1], 'data': data}
    return dict(sorted(groups.items()))


def _population_row(key: tuple, pop: str, data: dict[str, Any]) -> dict[str, Any]:
    scenario, case, N, mech, tie = key
    cost = data[f'{pop}__cost']
    shadow = data[f'{pop}__shadow']
    row: dict[str, Any] = {
        'scenario': scenario,
        'case': case,
        'N': N,
        'mechanism': mech,
        'tie_break': tie,
        'population': pop,
        'reps': int(cost.size),
    }
    row['cost_mean'], row['cost_ci95'] = _ci(cost)
    row['shadow_mean'], row['shadow_ci95'] = _ci(shadow)
    row['arm0_mean'], row['arm0_ci95'] = _ci(data[f'{pop}__arm0'])
    row['rest_mean'], row['rest_ci95'] = _ci(data[f'{pop}__rest'])
    if f'{pop}__over_a' in data:
        over_a, over_b = data[f'{pop}__over_a'], data[f'{pop}__over_b']
        exc_a, exc_b = data[f'{pop}__exc_a'], data[f'{pop}__exc_b']
        row.update({
            'overload_admit': float(np.mean(over_a)),
            'overload_fast': float(np.mean(over_b)),
            'excess_admit_mean': float(np.mean(exc_a)),
            'excess_admit_p95': float(np.quantile(exc_a, 0.95)),
            'excess_fast_mean': float(np.mean(exc_b)),
            'excess_fast_p95': float(np.quantile(exc_b, 0.95)),
            'requested_admit': float(np.mean(data[f'{pop}__req_a'])),
            'requested_fast': float(np.mean(data[f'{pop}__req_b'])),
            'used_admit': float(np.mean(data[f'{pop}__use_a'])),
            'used_fast': float(np.mean(data[f'{pop}__use_b'])),
            'denied_admit': float(np.mean(data[f'{pop}__den_a'])),
            'denied_fast': float(np.mean(data[f'{pop}__den_b'])),
            'degraded_fast': float(np.mean(data[f'{pop}__deg_b'])),
        })
    else:
        row.update({k: math.nan for k in (
            'overload_admit', 'overload_fast', 'excess_admit_mean', 'excess_admit_p95', 'excess_fast_mean', 'excess_fast_p95',
            'requested_admit', 'requested_fast', 'used_admit', 'used_fast', 'denied_admit', 'denied_fast', 'degraded_fast')})
    row['repairs_per_queue'] = float(np.mean(data[f'{pop}__repairs']))
    for name, col in (('qlen', 'qlen_mean_per_stage'), ('full', 'full_mean_per_stage'), ('nonadmit', 'not_admitted_per_queue_horizon'),
                      ('blocked', 'blocked_per_queue_horizon'), ('depart', 'departures_per_queue_horizon')):
        row[col] = float(np.mean(data[f'{pop}__{name}']))
    return row


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, str]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, ci = _ci(diff)
    if math.isnan(ci):
        verdict = 'n/a'
    elif mean - ci > 0:
        verdict = 'higher (95%)'
    elif mean + ci < 0:
        verdict = 'lower (95%)'
    else:
        verdict = 'not significant'
    return mean, ci, verdict


def _paired_rows(key: tuple, data: dict[str, Any]) -> list[dict[str, Any]]:
    scenario, case, N, mech, tie = key
    rows = []
    comparisons = [
        ('selfish_minus_coord', 'selfish__cost', 'coord__cost'),
        ('indep_minus_coord', 'indep__cost', 'coord__cost'),
        ('price_minus_coord', 'price__cost', 'coord__cost'),
        ('selfish_shadow_minus_coord', 'selfish__shadow', 'coord__shadow'),
        ('indep_shadow_minus_coord', 'indep__shadow', 'coord__shadow'),
        ('price_shadow_minus_coord', 'price__shadow', 'coord__shadow'),
        ('selfish_minus_indep', 'selfish__cost', 'indep__cost'),
    ]
    base_cost = float(np.mean(data['coord__cost']))
    for label, left, right in comparisons:
        mean, ci, verdict = _paired(data[left], data[right])
        rows.append({
            'scenario': scenario, 'case': case, 'N': N, 'mechanism': mech, 'tie_break': tie,
            'comparison': label, 'reps': int(np.size(data[left])), 'diff_mean': mean, 'diff_ci95': ci,
            'pct_of_coord': 100.0 * mean / base_cost if base_cost else math.nan, 'verdict': verdict,
        })
    return rows


def _dilemma_rows(key: tuple, data: dict[str, Any]) -> list[dict[str, Any]]:
    scenario, case, N, mech, tie = key
    reps = int(np.size(data['selfish__cost']))
    collective_mean, collective_ci, collective_verdict = _paired(data['selfish__cost'], data['coord__cost'])
    collectively_worse = collective_verdict == 'higher (95%)'
    shadow_mean, shadow_ci, shadow_verdict = _paired(data['selfish__shadow'], data['coord__shadow'])
    fluid_lp = float(data['fluid_lp'])
    fluid_selfish = float(data['fluid_selfish'])
    fluid_br = float(data['fluid_br'])
    deviation_rows = []
    profitable = []
    for dev in DEVIATIONS:
        gain_mean, gain_ci, gain_verdict = _paired(data[f'{dev}__arm0'], data['selfish__arm0'])
        within_mean, within_ci, within_verdict = _paired(data[f'{dev}__arm0'], data[f'{dev}__rest'])
        if not math.isnan(gain_ci) and gain_mean + gain_ci < 0:
            profitable.append(dev)
        for test, mean, ci, verdict in (
            (f'unilateral: {POP_LABELS[dev]} minus selfish', gain_mean, gain_ci, gain_verdict),
            (f'within run: {POP_LABELS[dev]} minus selfish queues', within_mean, within_ci, within_verdict),
        ):
            deviation_rows.append({
                'scenario': scenario, 'case': case, 'N': N, 'mechanism': mech, 'tie_break': tie,
                'test': test, 'reps': reps, 'diff_mean': mean, 'diff_ci95': ci, 'verdict': verdict,
            })
    individually_rational_sim = not profitable
    gap = fluid_selfish - fluid_br  # >= 0 when a best response to the selfish lotteries does better than the selfish rule
    rel_gap = gap / fluid_selfish if fluid_selfish else math.nan
    if abs(gap) <= 1e-6 * max(1.0, abs(fluid_selfish)):
        br_verdict = 'selfish rule is a best response'
        br_exact = True
    elif rel_gap <= 1e-3:
        br_verdict = f'approximately a best response (a single queue gains {100 * rel_gap:.3f}%)'
        br_exact = False
    else:
        br_verdict = f'not a best response (a single queue gains {100 * rel_gap:.2f}%)'
        br_exact = False
    if collectively_worse and individually_rational_sim and br_exact:
        dilemma = 'yes'
    elif collectively_worse and individually_rational_sim and rel_gap <= 1e-3:
        dilemma = 'approximately'
    else:
        dilemma = 'no'
    base = {'scenario': scenario, 'case': case, 'N': N, 'mechanism': mech, 'tie_break': tie, 'reps': reps}
    verdict_rows = [
        {**base, 'test': 'verdict: collective (all-selfish minus coordinated, simulated)',
         'diff_mean': collective_mean, 'diff_ci95': collective_ci, 'verdict': collective_verdict},
        {**base, 'test': 'verdict: individual (simulated unilateral deviations)', 'diff_mean': math.nan, 'diff_ci95': math.nan,
         'verdict': 'holds' if individually_rational_sim else f'violated by {", ".join(profitable)}'},
        {**base, 'test': 'verdict: individual (fluid best response to the selfish lotteries)', 'reps': 0,
         'diff_mean': gap, 'diff_ci95': math.nan, 'verdict': br_verdict},
        {**base, 'test': 'verdict: social dilemma (collective worse and individual rational)', 'diff_mean': math.nan,
         'diff_ci95': math.nan, 'verdict': dilemma},
        {**base, 'test': 'verdict: collective, shadow-priced accounting (device, not a model cost)',
         'diff_mean': shadow_mean, 'diff_ci95': shadow_ci, 'verdict': shadow_verdict},
        {**base, 'test': 'fluid: selfish minus LP (N -> infinity)', 'reps': 0,
         'diff_mean': fluid_selfish - fluid_lp, 'diff_ci95': math.nan, 'verdict': 'fluid'},
    ]
    return verdict_rows + deviation_rows


def _fluid_rows(groups: dict) -> list[dict[str, Any]]:
    rows = []
    for key, group in groups.items():
        scenario, case, N, mech, tie = key
        d = group['data']
        comp_s = np.asarray(d['fluid_selfish_components'], dtype=float)
        comp_l = np.asarray(d['fluid_lp_components'], dtype=float)
        comp_p = np.asarray(d['fluid_price_components'], dtype=float)
        usage_s = np.asarray(d['fluid_selfish_usage'], dtype=float)
        usage_l = np.asarray(d['fluid_lp_usage'], dtype=float)
        usage_p = np.asarray(d['fluid_price_usage'], dtype=float)
        row = {
            'scenario': scenario, 'case': case, 'mechanism': mech, 'tie_break': tie,
            'lp_value': float(d['lp_value']), 'fluid_lp': float(d['fluid_lp']),
            'fluid_selfish': float(d['fluid_selfish']), 'fluid_price': float(d['fluid_price']), 'fluid_br': float(d['fluid_br']),
            'selfish_admit_usage': float(usage_s[0].sum()), 'selfish_fast_usage': float(usage_s[1].sum()),
            'lp_admit_usage': float(usage_l[0].sum()), 'lp_fast_usage': float(usage_l[1].sum()),
            'price_admit_usage': float(usage_p[0].sum()), 'price_fast_usage': float(usage_p[1].sum()),
        }
        for name in ('lp_min_occupation', 'lp_nonnegativity_violation', 'lp_initial_residual',
                     'lp_flow_residual', 'lp_stage_mass_residual', 'lp_resource_violation',
                     'lp_feasibility_tolerance'):
            row[name] = float(d[name])
        for term, s_val, l_val, p_val in zip(COST_TERMS, comp_s, comp_l, comp_p):
            row[f'selfish_{term}'] = float(s_val)
            row[f'lp_{term}'] = float(l_val)
            row[f'price_{term}'] = float(p_val)
            row[f'selfish_minus_lp_{term}'] = float(s_val - l_val)
        rows.append(row)
    return rows


def _incentive_rows_for(records: list[tuple[dict, dict]]) -> list[dict[str, Any]]:
    '''
    Private advantages and the selfish and price-taking actions, once per (scenario, case, tie-break).
    '''

    seen: set[tuple] = set()
    rows = []
    for job, _ in records:
        key = (job['scenario'], job['case'], job['tie_break'])
        if key in seen:
            continue
        seen.add(key)
        params = job_params(job)
        model = queue_wcmdp(params)
        sol = selfish_solution(model, job['tie_break'])
        lp = solve_method_a(model)
        pi_lp = conditional_policy(lp['y'])
        price = priced_solution(model, lp['lam_res'], job['tie_break'])
        for row in incentive_rows(sol):
            t, s = row['t'], row['s']
            rows.append({
                'scenario': job['scenario'], 'case': job['case'], 'tie_break': job['tie_break'],
                'gamma': params.gamma, 'gamma_bar': params.gamma_bar, 'beta': params.beta,
                **row,
                'price_action_a': int(price.a[t, s]),
                'price_action_b': int(price.b[t, s]),
                'lp_prob_admit': float(pi_lp[t, s] @ ADMIT),
                'lp_prob_fast': float(pi_lp[t, s] @ HIGH),
                'lp_lam_admit': float(lp['lam_res'][0, t]),
                'lp_lam_fast': float(lp['lam_res'][1, t]),
            })
    return rows


STAGE_SERIES = {
    'ratio_admission': ('req_a', 'alpha'),
    'ratio_high_rate': ('req_b', 'beta'),
    'usage_admission': ('use_a', None),
    'usage_high_rate': ('use_b', None),
    'cost': ('cost_t', None),
}


def _stage_rows(groups: dict) -> list[dict[str, Any]]:
    '''
    Per-stage means with 95% t-intervals over replications, for the populations with stage arrays: requests divided by
    the budget (`ratio_*`), realised usage fractions (`usage_*`) and the cost of the stage.
    '''
    rows = []
    for key, group in groups.items():
        scenario, case, N, mech, tie = key
        job = group['job']
        for pop in DETAILED:
            for metric, (name, budget_key) in STAGE_SERIES.items():
                arr = group['data'].get(f'{pop}__{name}')
                if arr is None:
                    continue
                arr = np.asarray(arr, dtype=float)
                if budget_key is not None:
                    arr = arr / float(job[budget_key])
                for t in range(arr.shape[1]):
                    mean, half = _ci(arr[:, t])
                    rows.append({'scenario': scenario, 'case': case, 'N': N, 'mechanism': mech, 'tie_break': tie,
                                 'population': pop, 'stage': t, 'metric': metric, 'mean': mean,
                                 'ci_low': mean - half, 'ci_high': mean + half})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, '')
        return
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in row.items()})
    atomic_write_text(path, buffer.getvalue())


def _cell(value: float, ci: float | None = None, digits: int = 1, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '--'
    scale = 100.0 if pct else 1.0
    text = f'{scale * value:.{digits}f}'
    if ci is not None and not math.isnan(ci):
        text += f' $\\pm$ {scale * ci:.{digits}f}'
    return text


def _latex(groups: dict, table_n: int, scenario: str = 'base', tie: str = 'no_request') -> str:
    '''
    Compact LaTeX table with one panel per case, from the groups at N = table_n.
    '''

    header = [
        '% Generated by lplearn experiments summarize commons. Simulation output (common random numbers, 95% CIs), not a theorem.',
        '% Columns: overload = share of (replication, stage) pairs whose requests exceed floor(budget N); cost per queue is the',
        '% expected sum over stages of the one-stage cost; shadow = cost plus LP shadow prices times the realised excess of usage.',
        r'\begin{tabular}{l rr r rr}',
        r'\toprule',
        r'Policy (mechanism) & Overload adm. (\%) & Overload fast (\%) & Cost per queue & Shadow cost & Full (\%) \\',
        r'\midrule',
    ]
    lines = list(header)
    rows_spec = [
        ('coordinated (LP projection)', 'M1', 'coord'),
        ('independent LP policy (M1)', 'M1', 'indep'),
        ('independent LP policy (M0)', 'M0', 'indep'),
        ('selfish (M0)', 'M0', 'selfish'),
        ('selfish (M1)', 'M1', 'selfish'),
        ('selfish (M2)', 'M2', 'selfish'),
        ('price-taking (M1)', 'M1', 'price'),
    ]
    tie_tex = tie.replace('_', r'\_')
    for case in ('CJS', 'SJS'):
        lines.append(rf'\multicolumn{{6}}{{l}}{{\textit{{Case {case}, $N={table_n}$, tie-break \texttt{{{tie_tex}}}}}}} \\')
        for label, mech, pop in rows_spec:
            key = (scenario, case, table_n, mech, tie)
            if key not in groups:
                continue
            data = groups[key]['data']
            cost_mean, cost_ci = _ci(data[f'{pop}__cost'])
            shadow_mean, shadow_ci = _ci(data[f'{pop}__shadow'])
            over_a = float(np.mean(data[f'{pop}__over_a'])) if f'{pop}__over_a' in data else math.nan
            over_b = float(np.mean(data[f'{pop}__over_b'])) if f'{pop}__over_b' in data else math.nan
            full = float(np.mean(data[f'{pop}__full']))
            if pop == 'coord':
                over_a, over_b = float(np.mean(data['coord__over_a'])), float(np.mean(data['coord__over_b']))
            lines.append(' & '.join([
                label,
                _cell(over_a, pct=True),
                _cell(over_b, pct=True),
                _cell(cost_mean, cost_ci),
                _cell(shadow_mean, shadow_ci),
                _cell(full, pct=True),
            ]) + r' \\')
        if case == 'CJS':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def summarize(records: list[tuple[dict[str, Any], dict[str, Any]]], cfg: dict[str, Any], out_dir: Path) -> None:
    '''
    Write the stage tables for the complete records (see the module docstring).
    '''

    groups = _group_records(records)
    summary_rows, paired_rows, dilemma_rows = [], [], []
    for key, group in groups.items():
        for pop in POPULATIONS:
            summary_rows.append(_population_row(key, pop, group['data']))
        paired_rows.extend(_paired_rows(key, group['data']))
        dilemma_rows.extend(_dilemma_rows(key, group['data']))

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / 'commons_summary.csv', summary_rows)
    _write_csv(out_dir / 'commons_paired.csv', paired_rows)
    _write_csv(out_dir / 'commons_dilemma.csv', dilemma_rows)
    _write_csv(out_dir / 'commons_incentives.csv', _incentive_rows_for(records))
    _write_csv(out_dir / 'commons_fluid.csv', _fluid_rows(groups))
    _write_csv(out_dir / 'commons_stages.csv', _stage_rows(groups))
    available = sorted({key[2] for key in groups})
    table_n = int(cfg.get('table_n', 100))
    if available and table_n not in available:
        table_n = available[-1]  # e.g. a smoke run without N = 100: use the largest N that was run
    atomic_write_text(out_dir / 'commons_table.tex', _latex(groups, table_n))
