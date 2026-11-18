'''
Finite-N evaluation stage (``finite_n``): feasible N-queue policies built from LP occupation measures,
simulated with sample-path budgets and common random numbers. Construction and metrics: reports/finite_n/README.md.

A job simulates one (case, method, policy, N, replication chunk). The policy of a method is the conditional
policy pi = y / m of its occupation measure y on the detailed grid. The sources are the detailed or aggregate
LP (``exact_lp``, method A and E-LP..K-LP), the LP with the primal regularization (``exact_lp_reg``, method B),
the fixed point of the regularized saddle operator G (``exact_saddle``, the point GDA and SGDA converge to,
for methods C..K), or the averaged iterate of the learners stage (``learned``, from `learned_file`).

Policies: ``projection`` (open-loop rounding and repair of pi, StrictBudget), ``lp_update`` (closed-loop LP
re-solve, StrictBudget, only for LP-based sources) and ``independent`` (uncoordinated sampling of pi, Unenforced).

A job stores the per-replication scalars exactly (rep_index, cost, violation and overload maxima) and, for every
per-stage quantity, its sum over the replications and the sum of squares, so chunks combine exactly. Each
per-stage quantity is in arm fractions unless the name says otherwise.
'''

import csv
import hashlib
import itertools
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from lplearn.analysis.metrics import feasibility
from lplearn.core import PARTITIONS_K5, build_lp, conditional_policy, queue_wcmdp
from lplearn.experiments.io import atomic_write_text, load_arrays
from lplearn.experiments.methods import LEARNER_DEFAULTS, exact_solution, model_fields, params_from_job, partition_for
from lplearn.experiments.runner import ConfigError
from lplearn.finite_n import (
    IndependentRandomizedController, LPProjectionController, LPUpdateController, integer_budgets, simulate,
)
from lplearn.finite_n.mechanisms import MECHANISMS


NAME = 'finite_n'
VERSION = '4'  # 4: record diagnostics of learned occupations before policy conversion
POLICY_ORDER = ('projection', 'lp_update', 'independent')
MECHANISM_OF_POLICY = {'projection': 'StrictBudget', 'lp_update': 'StrictBudget', 'independent': 'Unenforced'}
SADDLE_SOURCES = ('exact_saddle', 'learned')
# Partition labels: Table tab:methods for K = 5, and the boundary-aware (K11) and pairs (P2) partitions for K = 11.
AGGREGATE_LABELS = tuple(PARTITIONS_K5) + ('K11', 'P2')
PER_REPLICATION = ('rep_index', 'cost', 'viol_max', 'viol_any', 'over_max', 'over_any')
STAGE_METRICS = (
    'cost_t', 'req_adm_t', 'adm_t', 'req_high_t', 'high_t', 'viol_adm_t', 'viol_high_t', 'over_adm_t',
    'over_high_t', 'queue_t', 'full_t', 'nonadmit_t', 'blocked_t', 'depart_t', 'repairs_t', 'repairs_adm_t',
    'repairs_high_t', 'fallbacks_t', 'denied_adm_t', 'denied_high_t',
)


def method_source(method: str, saddle_source: str = 'exact_saddle') -> dict[str, Any]:
    '''How the occupation measure of a method is obtained: solve kind, partition label and source tag.'''
    if method == 'A':
        return {'kind': 'lp', 'partition': None, 'source': 'exact_lp'}
    if method == 'B':
        return {'kind': 'lp_reg', 'partition': None, 'source': 'exact_lp_reg'}
    if method in ('C', 'D'):
        return {'kind': 'saddle', 'partition': None, 'source': saddle_source}
    if method in AGGREGATE_LABELS:
        return {'kind': 'saddle', 'partition': method, 'source': saddle_source}
    if method.endswith('-LP') and method[:-3] in AGGREGATE_LABELS:
        return {'kind': 'lp', 'partition': method[:-3], 'source': 'exact_lp'}
    raise ConfigError(f'unknown method {method!r}; use A, B, C, D, a partition label or <label>-LP')


def make_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    '''
    Jobs over case x method x policy x N x replication chunk. Every method gets a projection job; `lp_update`
    and `independent` jobs are added for the methods listed in the config. All jobs of one configuration share
    the same seed, so the replications are paired across methods and policies (common random numbers).
    '''
    K, T = int(cfg.get('K', 5)), int(cfg.get('T', 10))
    cases = [str(case) for case in cfg.get('cases', ['CJS', 'SJS'])]
    methods = [str(method) for method in cfg['methods']]
    lp_update = [str(method) for method in cfg.get('lp_update_methods', [])]
    independent = [str(method) for method in cfg.get('independent_methods', [])]
    saddle_source = str(cfg.get('saddle_source', 'exact_saddle'))
    learned_file = str(cfg.get('learned_file', ''))
    learned_seeds = [int(x) for x in cfg.get('learned_seeds', [cfg.get('learned_seed', 0)])]
    N_values = [int(n) for n in cfg['N_values']]
    n_chunks, reps = int(cfg['n_chunks']), int(cfg['reps_per_chunk'])
    seed = int(cfg['seed'])
    model = model_fields(cfg)
    instance = f'K{K}T{T}'
    defaults = LEARNER_DEFAULTS.get(instance, LEARNER_DEFAULTS['K5T10'])
    saddle_gamma = {key: float(cfg.get(key, defaults[key])) for key in ('Gamma_y', 'Gamma_lambda', 'Gamma_mu')}

    if saddle_source not in SADDLE_SOURCES:
        raise ConfigError(f'saddle_source must be one of {SADDLE_SOURCES}, got {saddle_source!r}')
    if any(case not in ('CJS', 'SJS') for case in cases):
        raise ConfigError(f'cases must be CJS or SJS, got {cases}')
    for label in lp_update:
        if label not in methods or method_source(label, saddle_source)['source'] != 'exact_lp':
            raise ConfigError(f'lp_update applies only to LP-based methods listed in methods, got {label!r}')
    for label in independent:
        if label not in methods:
            raise ConfigError(f'independent_methods must be listed in methods, got {label!r}')

    jobs = []
    for case, method in itertools.product(cases, methods):
        spec = method_source(method, saddle_source)
        if spec['source'] == 'learned' and not learned_file:
            raise ConfigError('saddle_source = "learned" needs learned_file')
        policies = ['projection']
        policies += ['lp_update'] if method in lp_update else []
        policies += ['independent'] if method in independent else []
        seeds = _available_seeds(learned_file, case, method, learned_seeds) if spec['source'] == 'learned' else [None]
        for policy, N, chunk, lseed in itertools.product(policies, N_values, range(n_chunks), seeds):
            job = {
                'case': case, 'K': K, 'T': T, 'method': method, 'kind': spec['kind'],
                'partition': spec['partition'], 'source': spec['source'], 'policy': policy,
                'mechanism': MECHANISM_OF_POLICY[policy], 'N': N, 'chunk': chunk,
                'rep_offset': chunk * reps, 'n_reps': reps, 'seed': seed, **model,
            }
            if spec['kind'] == 'saddle' and spec['source'] == 'exact_saddle':
                job.update(saddle_gamma, convention='consistent')
            if policy == 'lp_update':
                job['Gamma_y'] = 0.0
            if spec['source'] == 'learned':
                job['learned_file'] = learned_file
                job['learned_seed'] = lseed
                job['learned_digest'] = _learned_digest(learned_file, case, method, lseed)
            jobs.append(job)
    return jobs


def _available_seeds(path: str, case: str, method: str, seeds: list[int]) -> list[int]:
    '''The configured learned seeds that the learners stage recorded for this case and method (GDA has fewer seeds).'''
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f'learned solutions {file} not found; run and summarize the learners stage first')
    with np.load(file, allow_pickle=False) as arrays:
        present = [s for s in seeds if f'{case}/{method}/seed{int(s)}/y_avg' in arrays.files]
    if not present:
        raise ConfigError(f'no learned seed of {case}/{method} in {file}')
    return present


def _learned_digest(path: str, case: str, method: str, seed: int) -> str:
    '''SHA-256 prefix of a learned occupation measure, so cached jobs are invalidated when the learners rerun.'''
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f'learned solutions {file} not found; run and summarize the learners stage first')
    key = f'{case}/{method}/seed{int(seed)}/y_avg'
    with np.load(file, allow_pickle=False) as arrays:
        if key not in arrays.files:
            raise ConfigError(f'{key} is not in {file}')
        y = np.ascontiguousarray(arrays[key], dtype=float)
    return hashlib.sha256(y.tobytes()).hexdigest()[:16]


def _learned_occupation(job: dict[str, Any], model) -> np.ndarray:
    path = Path(job['learned_file'])
    if not path.is_file():
        raise FileNotFoundError(
            f'learned solutions {path} not found; run the learners stage or use saddle_source = "exact_saddle"'
        )
    arrays = load_arrays(path)
    key = f"{job['case']}/{job['method']}/seed{int(job['learned_seed'])}/y_avg"
    if key not in arrays:
        raise KeyError(f'{key} is not in {path}')
    y = np.asarray(arrays[key], dtype=float)
    if y.shape != (model.T, model.S, model.U):
        raise ValueError(f'{key} has shape {y.shape}, expected {(model.T, model.S, model.U)} on the detailed grid')
    if not np.isfinite(y).all():
        raise ValueError(f'{key} contains non-finite occupation values')
    digest = hashlib.sha256(np.ascontiguousarray(y, dtype=float).tobytes()).hexdigest()[:16]
    if 'learned_digest' in job and digest != job['learned_digest']:
        raise ValueError(f'{key} changed after job planning')
    return y


def _learned_diagnostics(model, y: np.ndarray) -> dict[str, Any]:
    '''Describe the raw learned iterate before clipping or policy fallback.'''
    with np.errstate(over='ignore', invalid='ignore'):
        diagnostics = feasibility(build_lp(model), y)
        clipped_mass = np.maximum(y, 0.0).sum(axis=2)
        raw_objective = float(np.sum(model.c[None] * y))
    if (not np.isfinite(clipped_mass).all() or not np.isfinite(raw_objective)
            or any(not np.isfinite(value).all() for value in diagnostics.values())):
        raise ValueError('learned occupation diagnostics are non-finite')
    return {
        'learned_raw_min': float(np.min(y)),
        'learned_raw_objective': raw_objective,
        'learned_raw_mass': np.asarray(diagnostics['mass'], dtype=float),
        'learned_flow_violation': float(diagnostics['flow_violation']),
        'learned_budget_violation': float(diagnostics['budget_violation']),
        'learned_nonneg_violation': float(diagnostics['nonneg_violation']),
        'learned_fallback_rows': int(np.count_nonzero(clipped_mass <= 1e-9)),
        'learned_empty_stages': int(np.count_nonzero(np.all(clipped_mass <= 1e-9, axis=1))),
    }


def _reduce(result: dict[str, Any]) -> dict[str, Any]:
    '''Keep the per-replication arrays exactly, and the sums and sums of squares of the per-stage arrays.'''
    out = {key: np.asarray(result[key]) for key in PER_REPLICATION}
    for key, value in result.items():
        if key in PER_REPLICATION or key == 'state_hist' or np.ndim(value) != 2:
            continue
        value = np.asarray(value, dtype=float)
        out[f'{key}__sum'] = value.sum(axis=0)
        out[f'{key}__sumsq'] = (value ** 2).sum(axis=0)
    out['state_hist__sum'] = np.asarray(result['state_hist'], dtype=float).sum(axis=0)  # (T, S)
    return out


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    '''
    Simulate one job. Returns the reduced arrays of ``simulate`` (see ``_reduce``) and scalar metadata: the LP
    objective of the case (method A), the objective and final mass of the job's source, and the integer budgets.
    '''
    K, T = int(job['K']), int(job['T'])
    params = params_from_job(job)
    model = queue_wcmdp(params)
    partition = partition_for(job['partition'], K)
    learned_diagnostics = {}
    if job['source'] == 'learned':
        y = _learned_occupation(job, model)
        learned_diagnostics = _learned_diagnostics(model, y)
    else:
        gamma = {key: float(job[key]) for key in ('Gamma_y', 'Gamma_lambda', 'Gamma_mu') if key in job}
        if 'convention' in job:
            gamma['convention'] = str(job['convention'])
        y = np.array(exact_solution(params, job['kind'], partition, **gamma)['y'], dtype=float)
    pi = conditional_policy(y)

    policy = job['policy']
    if policy == 'projection':
        def make_controller():
            return LPProjectionController(pi, model.budgets)
    elif policy == 'lp_update':
        def make_controller():
            return LPUpdateController(model, partition=partition, Gamma_y=float(job['Gamma_y']))
    elif policy == 'independent':
        def make_controller():
            return IndependentRandomizedController(pi)
    else:
        raise ValueError(f'unknown policy {policy!r}')
    mechanism = MECHANISMS[job['mechanism']]()

    N = int(job['N'])
    result = simulate(model, make_controller, N, int(job['n_reps']), int(job['seed']), mechanism,
                      rep_offset=int(job['rep_offset']))
    budgets_int = integer_budgets(model.budgets, N)
    out = _reduce(result)
    out.update(learned_diagnostics)
    out.update({
        'lp_objective': float(exact_solution(params, 'lp')['objective']),
        'source_objective': float(np.sum(model.c[None] * y)),
        'source_mass_final': float(y[-1].sum()),
        'budget_adm': int(budgets_int[0]),
        'budget_high': int(budgets_int[1]),
        'N': N,
        'n_reps': int(job['n_reps']),
        'rep_offset': int(job['rep_offset']),
        'case': str(job['case']),
        'method': str(job['method']),
        'policy': policy,
        'mechanism': mechanism.name,
        'source': str(job['source']),
        'partition': str(job['partition'] or ''),
    })
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _merge_seeds(items: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    '''
    Combine one group whose policy comes from several learned seeds. Each seed is merged over its chunks; the
    per-replication cost is then averaged over seeds replication by replication (the seeds share the common
    random numbers), so the interval is over replications of the seed-averaged cost. Per-stage sums are averaged
    over seeds (their sums of squares give the mean within-seed variance), violation and overload indicators
    are maxima over seeds, and the spread of the per-seed mean costs is kept as `seed_means`.
    '''
    by_seed: dict[int, list] = defaultdict(list)
    for job, result in items:
        by_seed[int(job.get('learned_seed', -1))].append((job, result))
    if len(by_seed) == 1:
        return _merge(items)
    merged_seeds = [_merge(by_seed[seed]) for seed in sorted(by_seed)]
    first = merged_seeds[0]
    for other in merged_seeds[1:]:
        if not np.array_equal(other['rep_index'], first['rep_index']):
            raise ValueError('learned seeds of one group must share the same replications')
    merged = dict(first)
    merged['cost'] = np.mean([m['cost'] for m in merged_seeds], axis=0)
    for key in ('viol_max', 'over_max'):
        merged[key] = np.max([m[key] for m in merged_seeds], axis=0)
    for key in ('viol_any', 'over_any'):
        merged[key] = np.max([np.asarray(m[key], dtype=float) for m in merged_seeds], axis=0)
    for key in first:
        if key.endswith(('__sum', '__sumsq')):
            merged[key] = np.mean([m[key] for m in merged_seeds], axis=0)
    merged['seed_means'] = np.array([float(m['cost'].mean()) for m in merged_seeds])
    merged['source_objective'] = float(np.mean([m['source_objective'] for m in merged_seeds]))
    merged['source_mass_final'] = float(np.mean([m['source_mass_final'] for m in merged_seeds]))
    return merged


def _merge(items: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    '''Combine the chunks of one group: per-replication arrays are concatenated, per-stage sums are added.'''
    items = sorted(items, key=lambda item: int(item[0]['rep_offset']))
    first = items[0][1]
    merged: dict[str, Any] = {}
    for key in PER_REPLICATION:
        merged[key] = np.concatenate([np.asarray(result[key]) for _, result in items])
    for key in first:
        if key in PER_REPLICATION:
            continue
        if key.endswith(('__sum', '__sumsq')):
            merged[key] = np.sum([np.asarray(result[key]) for _, result in items], axis=0)
        else:
            merged[key] = first[key]  # scalars and strings are the same for all chunks
    merged['n'] = int(merged['cost'].size)
    merged['T'] = int(items[0][0]['T'])
    return merged


def _seed_ci(group: dict[str, Any]) -> float:
    '''
    95% t-interval half-width of the expected cost over solver seeds: computed from the per-seed mean costs when the
    policy comes from several learned seeds (each mean already averages the replications), otherwise the
    replication interval.
    '''
    means = np.asarray(group.get('seed_means', []), dtype=float)
    if means.size > 1:
        return float(stats.t.ppf(0.975, means.size - 1) * means.std(ddof=1) / np.sqrt(means.size))
    return _mean_sd_ci(group['cost'])[2]


def _mean_sd_ci(values: np.ndarray) -> tuple[float, float, float]:
    '''Mean, sample standard deviation and 95% t-interval half-width.'''
    values = np.asarray(values, dtype=float)
    n = values.size
    mean = float(values.mean())
    if n < 2:
        return mean, float('nan'), float('nan')
    sd = float(values.std(ddof=1))
    return mean, sd, float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))


def _overall(group: dict[str, Any], name: str) -> float:
    '''Mean of a per-stage quantity over replications and stages.'''
    return float(group[f'{name}__sum'].sum()) / (group['n'] * group['T'])


def _stage_mean_sd(group: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    n = group['n']
    mean = group[f'{name}__sum'] / n
    if n < 2:
        return mean, np.full_like(mean, np.nan)
    var = np.maximum(group[f'{name}__sumsq'] - n * mean ** 2, 0.0) / (n - 1)
    return mean, np.sqrt(var)


def _summary_row(group: dict[str, Any]) -> dict[str, Any]:
    cost = group['cost']
    mean, sd, half = _mean_sd_ci(cost)
    lp_objective = float(group['lp_objective'])
    return {
        'case': group['case'], 'method': group['method'], 'policy': group['policy'],
        'mechanism': group['mechanism'], 'source': group['source'], 'partition': group['partition'],
        'N': int(group['N']), 'n_reps': int(group['n']),
        'n_seeds': int(len(group.get('seed_means', [0]))),
        'cost_ci95_seed': _seed_ci(group),
        'seed_sd': float(np.std(group['seed_means'], ddof=1)) if len(group.get('seed_means', [])) > 1 else float('nan'),
        'budget_adm': int(group['budget_adm']), 'budget_high': int(group['budget_high']),
        'cost_mean': mean, 'cost_sd': sd, 'cost_ci95': half,
        'cost_ci_lo': mean - half, 'cost_ci_hi': mean + half,
        'cost_per_stage': mean / group['T'],
        'lp_objective': lp_objective, 'gap_to_lp': mean - lp_objective,
        'source_objective': float(group['source_objective']), 'source_mass_final': float(group['source_mass_final']),
        'queue_mean': _overall(group, 'queue_t'),
        'full_frac': _overall(group, 'full_t'),
        'nonadmit_frac': _overall(group, 'nonadmit_t'),
        'blocked_frac': _overall(group, 'blocked_t'),
        'depart_frac': _overall(group, 'depart_t'),
        'req_adm_frac': _overall(group, 'req_adm_t'),
        'adm_frac': _overall(group, 'adm_t'),
        'req_high_frac': _overall(group, 'req_high_t'),
        'high_frac': _overall(group, 'high_t'),
        'viol_max': float(group['viol_max'].max()),
        'viol_rep_frac': float(group['viol_any'].mean()),
        'over_max': float(group['over_max'].max()),
        'over_rep_frac': float(group['over_any'].mean()),
        'over_adm_frac': _overall(group, 'over_adm_t'),
        'over_high_frac': _overall(group, 'over_high_t'),
        'repairs_frac': _overall(group, 'repairs_t'),
        'repairs_adm_frac': _overall(group, 'repairs_adm_t'),
        'repairs_high_frac': _overall(group, 'repairs_high_t'),
        'fallbacks_total': int(group['fallbacks_t__sum'].sum()),
        'denied_adm_frac': _overall(group, 'denied_adm_t'),
        'denied_high_frac': _overall(group, 'denied_high_t'),
    }


def _paired_rows(groups: dict[tuple, dict[str, Any]]) -> list[dict[str, Any]]:
    '''CRN paired differences of cost against method A with the projection policy, at the same N and replication.'''
    rows = []
    for key, group in groups.items():
        case, method, policy, N = key
        ref = groups.get((case, 'A', 'projection', N))
        if ref is None or (method, policy) == ('A', 'projection'):
            continue
        cost = dict(zip(group['rep_index'].tolist(), group['cost'].tolist()))
        cost_ref = dict(zip(ref['rep_index'].tolist(), ref['cost'].tolist()))
        common = sorted(set(cost) & set(cost_ref))
        diff = np.array([cost[r] - cost_ref[r] for r in common])
        mean, sd, half = _mean_sd_ci(diff)
        rows.append({
            'case': case, 'N': N, 'method': method, 'policy': policy, 'ref_method': 'A', 'ref_policy': 'projection',
            'n_pairs': len(common), 'diff_mean': mean, 'diff_sd': sd, 'diff_ci95': half,
            'diff_ci_lo': mean - half, 'diff_ci_hi': mean + half,
        })
    return rows


def _stage_rows(groups: dict[tuple, dict[str, Any]], keys: list[tuple]) -> list[dict[str, Any]]:
    '''Per-stage means of the metrics, the per-stage standard deviation of the cost, and the state distribution.'''
    rows = []
    for key in keys:
        group = groups[key]
        case, method, policy, N = key
        cost_mean, cost_sd = _stage_mean_sd(group, 'cost_t')
        means = {name: group[f'{name}__sum'] / group['n'] for name in STAGE_METRICS}
        states = group['state_hist__sum'] / group['n']
        for t in range(group['T']):
            row = {'case': case, 'method': method, 'policy': policy, 'N': N, 'stage': t,
                   'cost_t_sd': cost_sd[t]}
            row.update({name: means[name][t] for name in STAGE_METRICS})
            row.update({f'p_s{s}': states[t, s] for s in range(states.shape[1])})
            rows.append(row)
    return rows


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: (f'{value:.6g}' if isinstance(value, (float, np.floating)) else value)
                         for key, value in row.items()})
    return buffer.getvalue()


def _latex_table(groups: dict[tuple, dict[str, Any]], cfg: dict[str, Any]) -> str:
    cases = [str(case) for case in cfg.get('cases', ['CJS', 'SJS'])]
    methods = [str(method) for method in cfg['methods']]
    Ns = [int(n) for n in cfg['N_values']]
    lp_lines = []
    for case in cases:
        ref = next((g for (c, m, p, _), g in groups.items() if c == case and (m, p) == ('A', 'projection')), None)
        if ref is not None:
            lp_lines.append(f'{case} = {float(ref["lp_objective"]):.3f}')
    lines = [
        '% Generated by `lplearn experiments summarize finite_n`. Requires \\usepackage{booktabs}.',
        '% Entries: cost per arm, summed over the T stages, as mean $\\pm$ 95\\% t-interval half-width over replications.',
        '% Reference lower bound: the LP value of method (A), unregularized relaxed LP, per arm: ' + ', '.join(lp_lines) + '.',
        '% The LP value is a lower bound in expectation for any policy that meets the budgets in expectation.',
        '% Projection and lp_update rows run under StrictBudget (zero sample-path violations, see the CSV).',
        '% The independent row is an uncoordinated diagnostic: its requests exceed the budgets on sample paths.',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Finite-$N$ cost per arm for feasible policies built from the LP occupation measures.}',
        '\\label{tab:finite-n}',
        f'\\begin{{tabular}}{{l{"r" * (len(cases) * len(Ns))}}}',
        '\\toprule',
        ' & ' + ' & '.join(f'\\multicolumn{{{len(Ns)}}}{{c}}{{{case}}}' for case in cases) + ' \\\\',
        ''.join(f'\\cmidrule(lr){{{2 + i * len(Ns)}-{1 + (i + 1) * len(Ns)}}}' for i in range(len(cases))),
        'Method & ' + ' & '.join(f'$N{{=}}{n}$' for _ in cases for n in Ns) + ' \\\\',
        '\\midrule',
    ]
    row_labels = []
    for method in methods:
        for policy in POLICY_ORDER:
            if any((case, method, policy, n) in groups for case in cases for n in Ns):
                row_labels.append((method, policy))
    for method, policy in row_labels:
        label = {'projection': '', 'lp_update': ' (LP update)', 'independent': ' (independent, diagnostic)'}[policy]
        cells = []
        for case in cases:
            for n in Ns:
                g = groups.get((case, method, policy, n))
                if g is None:
                    cells.append('--')
                else:
                    _, _, half = _mean_sd_ci(g['cost'])
                    cells.append(f'{float(g["cost"].mean()):.2f} $\\pm$ {half:.2f}')
        lines.append(f'{method}{label} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(lines)


def _check_complete(grouped: dict, groups: dict, cfg: dict[str, Any]) -> None:
    '''
    Refuse to summarize incomplete groups: every (case, method, policy, N) needs n_chunks x reps_per_chunk replications
    and, for learned policies, every configured seed. A partial run would otherwise give silently narrower tables.
    '''
    expected_reps = int(cfg['n_chunks']) * int(cfg['reps_per_chunk'])
    configured = [int(x) for x in cfg.get('learned_seeds', [cfg.get('learned_seed', 0)])]
    problems = []
    for key, items in grouped.items():
        learned = any(job.get('source') == 'learned' for job, _ in items)
        seeds = {int(job.get('learned_seed', -1)) for job, _ in items}
        if int(groups[key]['n']) != expected_reps:
            problems.append(f'{key}: {groups[key]["n"]} replications, expected {expected_reps}')
        if learned:
            expected_seeds = _available_seeds(str(cfg['learned_file']), key[0], key[1], configured)
            if seeds != set(expected_seeds):
                problems.append(f'{key}: learned seeds {sorted(seeds)}, expected {expected_seeds}')
    if problems:
        raise ConfigError('finite_n: incomplete groups (run the missing jobs first): ' + '; '.join(problems[:5]))


def summarize(records: list[tuple[dict[str, Any], dict[str, Any]]], cfg: dict[str, Any], out_dir: Path) -> None:
    '''
    Write finite_n_summary.csv (one row per case, method, policy and N), finite_n_paired.csv (CRN paired cost
    differences against method A with projection), finite_n_stages.csv (per-stage means and state distribution)
    and finite_n_table.tex (booktabs table of the cost per arm).
    '''
    out_dir = Path(out_dir)
    grouped: dict[tuple, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for job, result in records:
        grouped[(str(job['case']), str(job['method']), str(job['policy']), int(job['N']))].append((job, result))
    groups = {key: _merge_seeds(items) for key, items in grouped.items()}
    _check_complete(grouped, groups, cfg)

    method_order = {str(method): i for i, method in enumerate(cfg['methods'])}
    case_order = {case: i for i, case in enumerate(cfg.get('cases', ['CJS', 'SJS']))}
    keys = sorted(groups, key=lambda k: (case_order.get(k[0], 99), method_order.get(k[1], 99),
                                         POLICY_ORDER.index(k[2]), k[3]))
    summary = [_summary_row(groups[k]) for k in keys]
    paired = _paired_rows({key: groups[key] for key in keys})
    stages = _stage_rows(groups, keys)

    atomic_write_text(out_dir / 'finite_n_summary.csv', _csv_text(summary) if summary else '')
    atomic_write_text(out_dir / 'finite_n_paired.csv', _csv_text(paired) if paired else '')
    atomic_write_text(out_dir / 'finite_n_stages.csv', _csv_text(stages) if stages else '')
    atomic_write_text(out_dir / 'finite_n_table.tex', _latex_table(groups, cfg))
