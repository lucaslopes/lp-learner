'''
Quantitative comparison metrics of the numerical evaluation (Section 8, methods (A)-(K)).\n
Every metric is a pure function of detailed occupation measures y[t, s, u] (u = 2a + b), resource
prices lam[l, t] and the model. Aggregate solutions are lifted to the detailed grid before they are
compared. Formulas and conventions are in reports/metrics/README.md.\n
The LP objective and the feasibility checks are properties of y itself. The fluid evaluation is
the mean-field cost of the conditional policy pi = y / m in the true dynamics, which is the
quantity that is comparable across methods when y leaks mass.
'''


from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from lplearn.core import LPStructure, WCMDP, conditional_policy, fluid_evaluation


NAN = float('nan')


# ---------------------------------------------------------------------------
# Elementary distances and statistics
# ---------------------------------------------------------------------------


def frobenius(a, b) -> float:
    '''
    ||a - b||_F for two arrays of the same shape.
    '''

    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def pairwise_frobenius(arrays: Sequence[np.ndarray]) -> np.ndarray:
    '''
    Symmetric matrix D[i, j] = ||arrays[i] - arrays[j]||_F with an exact zero diagonal.
    '''

    if len(arrays) == 0:
        return np.zeros((0, 0))
    flat = np.stack([np.ravel(np.asarray(a, dtype=float)) for a in arrays])
    diff = flat[:, None, :] - flat[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def mean_sd(values) -> dict:
    '''
    Mean, sample standard deviation (ddof = 1) and count of the finite entries. NaN where undefined.
    '''

    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = int(x.size)
    mean = float(x.mean()) if n else NAN
    sd = float(x.std(ddof=1)) if n >= 2 else NAN
    return {'mean': mean, 'sd': sd, 'n': n}


def summarize_runs(records: Sequence[Mapping[str, float]]) -> dict:
    '''
    mean_sd of every key of the per-run metric dictionaries.
    '''

    if not records:
        return {}
    return {key: mean_sd([record[key] for record in records]) for key in records[0]}


def run_to_run(arrays: Sequence[np.ndarray]) -> dict:
    '''
    Mean and sd over the pairs i < j of ||arrays[i] - arrays[j]||_F: the variation between independent runs
    of one configuration (solver noise, since the configuration is the same).
    '''

    n = len(arrays)
    if n < 2:
        return {'mean': NAN, 'sd': NAN, 'n': 0}
    D = pairwise_frobenius(arrays)
    return mean_sd(D[np.triu_indices(n, k=1)])


# ---------------------------------------------------------------------------
# Metrics of one solution
# ---------------------------------------------------------------------------


def feasibility(struct: LPStructure, y) -> dict:
    '''
    Constraint violations of y on the detailed grid:\n
    - flow_violation = max |A y - b| over the initial and flow rows;\n
    - budget_violation = max_{l,t} (sum_{s,u} d_l(s,u) y[t,s,u] - alpha_l)_+ (zero when feasible);\n
    - nonneg_violation = max (-y)_+;\n
    and the total mass per stage, mass[t] = sum_{s,u} y[t,s,u].
    '''

    yv = np.asarray(y, dtype=float)
    flat = yv.ravel()
    n_res = struct.L * struct.T
    flow = float(np.max(np.abs(struct.A @ flat - struct.b)))
    usage_excess = struct.D[:n_res] @ flat - struct.r[:n_res]
    return {
        'flow_violation': flow,
        'budget_violation': float(max(float(np.max(usage_excess)), 0.0)),
        'nonneg_violation': float(max(-float(np.min(flat)), 0.0)),
        'mass': yv.sum(axis=(1, 2)),
    }


def fluid_evaluate(model: WCMDP, y) -> dict:
    '''
    Mean-field evaluation of the conditional policy pi = y / m (see `conditional_policy`) in the true
    dynamics: total cost and the largest budget excess max_{l,t} (expected usage - alpha_l)_+.
    '''

    pi = conditional_policy(y)
    out = fluid_evaluation(model, pi)
    return {'fluid_cost': float(out['cost']), 'fluid_excess': float(out['excess'].max()), 'policy': pi}


def policy_distance(y_ref, y) -> dict:
    '''
    Mass-weighted policy distance between two occupation measures, insensitive to the mass leak of the
    dual-regularized solutions:\n
    D_pi(y_ref, y) = sum_{t,s} m^ref_s(t) * || pi_t(. | s) - pi^ref_t(. | s) ||_1, with
    m^ref_s(t) = sum_u max(y_ref[t,s,u], 0), pi = conditional_policy(y) and pi^ref = conditional_policy(y_ref).\n
    `policy_dist_per_mass` = D_pi / sum_{t,s} m^ref_s(t) is the average L1 distance per unit of reference mass
    (in [0, 2]). Both vanish when y is a positive multiple of y_ref at every stage.
    '''

    weight = np.maximum(np.asarray(y_ref, dtype=float), 0.0).sum(axis=2)  # (T, S)
    l1 = np.abs(conditional_policy(y) - conditional_policy(y_ref)).sum(axis=2)  # (T, S)
    total = float(np.sum(weight * l1))
    mass = float(weight.sum())
    return {'policy_dist': total, 'policy_dist_per_mass': total / mass if mass > 0 else NAN}


def price_distance(lam, lam_ref) -> dict:
    '''
    Resource-price distances: ||lam - lam_ref||_F over all resources and stages, and per resource l the
    distance ||lam[l, :] - lam_ref[l, :]||_2 over the stages.
    '''

    diff = np.asarray(lam, dtype=float) - np.asarray(lam_ref, dtype=float)
    per_resource = np.sqrt(np.sum(diff * diff, axis=1))
    return {'lam_dist': float(np.sqrt(np.sum(diff * diff))), 'lam_dist_per_resource': per_resource}


def legacy_to_detailed(stored) -> np.ndarray:
    '''
    Convert a stored legacy array indexed [s, a, b, t] (shape (S, 2, 2, T)) to the detailed grid y[t, s, u]
    with u = 2a + b.
    '''

    arr = np.asarray(stored, dtype=float)
    if arr.ndim != 4 or arr.shape[1:3] != (2, 2):
        raise ValueError(f'legacy arrays must have shape (S, 2, 2, T), got {arr.shape}')
    S, _, _, T = arr.shape
    return np.transpose(arr, (3, 0, 1, 2)).reshape(T, S, 4)


def solution_metrics(problem: Problem, y, lam, *, target: str | None = None, legacy=None) -> dict:
    '''
    All scalar metrics of one solution (occupation measures y and resource prices lam).\n
    `target` names the exact reference whose saddle point the solution approximates (solver noise), and
    `legacy` is a stored legacy array on the detailed grid. Keys that do not apply are NaN.
    '''

    A, B = problem.refs['A'], problem.refs['B']
    yv = np.asarray(y, dtype=float)
    lv = np.asarray(lam, dtype=float)
    obj = problem.struct.objective(yv)
    feas = feasibility(problem.struct, yv)
    fluid = fluid_evaluate(problem.model, yv)
    policy = policy_distance(B.y, yv)
    to_B = price_distance(lv, B.lam)
    out = {
        'objective': obj,
        'gap_A': obj - A.objective,
        'gap_A_pct': 100.0 * (obj - A.objective) / abs(A.objective),
        'gap_B': obj - B.objective,
        'gap_B_pct': 100.0 * (obj - B.objective) / abs(B.objective),
        'mass_last': float(feas['mass'][-1]),
        'flow_violation': feas['flow_violation'],
        'budget_violation': feas['budget_violation'],
        'nonneg_violation': feas['nonneg_violation'],
        'fluid_cost': fluid['fluid_cost'],
        'fluid_excess': fluid['fluid_excess'],
        'dist_A': frobenius(yv, A.y),
        'dist_B': frobenius(yv, B.y),
        'policy_dist': policy['policy_dist'],
        'policy_dist_per_mass': policy['policy_dist_per_mass'],
        'lam_dist_B': to_B['lam_dist'],
        **{f'lam_dist_B_res{index}': float(v) for index, v in enumerate(to_B['lam_dist_per_resource'])},
        'dist_star': NAN,
        'lam_dist_star': NAN,
        'legacy_dist': NAN,
    }
    if target is not None:
        ref = problem.refs[target]
        out['dist_star'] = frobenius(yv, ref.y)
        out['lam_dist_star'] = price_distance(lv, ref.lam)['lam_dist']
    if legacy is not None:
        out['legacy_dist'] = frobenius(yv, legacy)
    return out


# ---------------------------------------------------------------------------
# Problems, runs and the comparison table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Reference:
    '''
    An exact solution used as a target: occupation measures, resource prices and the objective c^T y.
    '''

    y: np.ndarray
    lam: np.ndarray
    objective: float


@dataclass(frozen=True, eq=False)
class Problem:
    '''
    One (instance, case). `refs` holds 'A' (unregularized LP), 'B' (regularized interior point) and
    'star:<partition>' (exact saddle point of the detailed, 'none', or aggregate problem). `legacy` holds the
    stored legacy arrays by method label, on the detailed grid.
    '''

    struct: LPStructure
    model: WCMDP
    refs: Mapping[str, Reference]
    legacy: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class MethodRuns:
    '''
    The runs of one learned method: seeds, and for each seed the tail average and the last iterate.
    '''

    label: str
    solver: str
    partition: str
    seeds: tuple[int, ...]
    y: tuple[np.ndarray, ...]
    lam: tuple[np.ndarray, ...]
    y_last: tuple[np.ndarray, ...]
    lam_last: tuple[np.ndarray, ...]

    @property
    def target(self) -> str:
        return f'star:{self.partition}'


def reference_metrics(problem: Problem) -> dict[str, dict]:
    '''
    Metrics of A, B and every exact saddle point of the problem (the saddle points are compared with themselves).
    '''

    out = {}
    for name, ref in problem.refs.items():
        target = name if name.startswith('star:') else None
        out[name] = solution_metrics(problem, ref.y, ref.lam, target=target, legacy=problem.legacy.get(name))
    return out


def comparison_rows(problem: Problem, methods: Sequence[MethodRuns]) -> list[dict]:
    '''
    One row per reference, (A) and (B), and per learned method, in the order given.\n
    A learned row aggregates its seeds with mean, sd and count (for the tail average `metrics` and for the
    last iterate `last`). It also reports the partition effect ||y* - y^B||_F of the method's exact target (no
    solver noise), the solver noise in `metrics['dist_star']`, and the run-to-run variation of y and lam over
    seeds. (A) and (B) are single deterministic solutions, so their sd is NaN.
    '''

    rows = []
    for label in ('A', 'B'):
        ref = problem.refs[label]
        met = solution_metrics(problem, ref.y, ref.lam, legacy=problem.legacy.get(label))
        rows.append({
            'label': label,
            'solver': {'A': 'lp', 'B': 'lp_reg'}[label],
            'partition': 'none',
            'target': None,
            'seeds': [],
            'n_seeds': 1,
            'metrics': {key: {'mean': value, 'sd': NAN, 'n': 1} for key, value in met.items()},
            'last': {},
            'per_seed': [met],
            'partition_effect': NAN,
            'partition_effect_lam': NAN,
            'aggregation_effect': NAN,
            'aggregation_effect_lam': NAN,
            'run_to_run_y': {'mean': NAN, 'sd': NAN, 'n': 0},
            'run_to_run_lam': {'mean': NAN, 'sd': NAN, 'n': 0},
            'legacy_dist_of_mean': met['legacy_dist'],
        })
    for m in methods:
        legacy = problem.legacy.get(m.label)
        per_seed = [solution_metrics(problem, y, lam, target=m.target, legacy=legacy) for y, lam in zip(m.y, m.lam)]
        per_seed_last = [solution_metrics(problem, y, lam, target=m.target) for y, lam in zip(m.y_last, m.lam_last)]
        target = problem.refs[m.target]
        detailed = problem.refs.get('star:none')
        rows.append({
            'label': m.label,
            'solver': m.solver,
            'partition': m.partition,
            'target': m.target,
            'seeds': list(m.seeds),
            'n_seeds': len(m.seeds),
            'metrics': summarize_runs(per_seed),
            'last': summarize_runs(per_seed_last),
            'per_seed': per_seed,
            'partition_effect': frobenius(target.y, problem.refs['B'].y),
            'partition_effect_lam': price_distance(target.lam, problem.refs['B'].lam)['lam_dist'],
            # Same regularization, so this difference is the aggregation error alone (zero for C and D).
            'aggregation_effect': frobenius(target.y, detailed.y) if detailed is not None else NAN,
            'aggregation_effect_lam': price_distance(target.lam, detailed.lam)['lam_dist'] if detailed is not None else NAN,
            'run_to_run_y': run_to_run(m.y),
            'run_to_run_lam': run_to_run(m.lam),
            'legacy_dist_of_mean': frobenius(np.mean(np.stack(m.y), axis=0), legacy) if legacy is not None else NAN,
        })
    return rows


def pairwise_matrices(problem: Problem, methods: Sequence[MethodRuns]) -> dict:
    '''
    Pairwise Frobenius distances among (A), (B) and the learned methods. `seed_mean` uses the seed-average
    of the tail averages, `seed0` the run with seed 0 (or the first seed if 0 is absent).
    '''

    names = ['A', 'B'] + [m.label for m in methods]
    mean_arrays = [problem.refs['A'].y, problem.refs['B'].y]
    seed0_arrays = [problem.refs['A'].y, problem.refs['B'].y]
    for m in methods:
        mean_arrays.append(np.mean(np.stack(m.y), axis=0))
        index = m.seeds.index(0) if 0 in m.seeds else 0
        seed0_arrays.append(m.y[index])
    return {
        'labels': names,
        'seed_mean': pairwise_frobenius(mean_arrays),
        'seed0': pairwise_frobenius(seed0_arrays),
    }


# ---------------------------------------------------------------------------
# Serialization and the manuscript table
# ---------------------------------------------------------------------------


def to_jsonable(obj):
    '''
    Plain Python values for json: NumPy types become Python types, arrays become lists, NaN and infinities become null.
    '''

    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.str_):
        return str(obj)
    return obj


def _csv_value(value):
    if value is None:
        return ''
    if isinstance(value, (float, np.floating)):
        return '' if not math.isfinite(float(value)) else repr(float(value))
    return value


def rows_to_csv(rows: Sequence[dict]) -> str:
    '''
    CSV with one line per row of `comparison_rows`: mean and sd of every metric, then the partition effect,
    run-to-run variation and the legacy distance.
    '''

    if not rows:
        return ''
    metric_keys = list(rows[0]['metrics'])
    header = ['label', 'solver', 'partition', 'n_seeds']
    for key in metric_keys:
        header += [f'{key}_mean', f'{key}_sd']
    header += [
        'partition_effect', 'partition_effect_lam', 'aggregation_effect', 'aggregation_effect_lam',
        'run_to_run_y_mean', 'run_to_run_y_sd', 'run_to_run_y_pairs', 'run_to_run_lam_mean', 'run_to_run_lam_sd',
        'legacy_dist_of_mean',
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    for row in rows:
        values = [row['label'], row['solver'], row['partition'], row['n_seeds']]
        for key in metric_keys:
            values += [row['metrics'][key]['mean'], row['metrics'][key]['sd']]
        values += [
            row['partition_effect'], row['partition_effect_lam'], row['aggregation_effect'], row['aggregation_effect_lam'],
            row['run_to_run_y']['mean'], row['run_to_run_y']['sd'], row['run_to_run_y']['n'], row['run_to_run_lam']['mean'],
            row['run_to_run_lam']['sd'], row['legacy_dist_of_mean'],
        ]
        writer.writerow([_csv_value(value) for value in values])
    return buffer.getvalue()


def matrix_to_csv(labels: Sequence[str], matrix: np.ndarray) -> str:
    '''
    A labelled square matrix as CSV.
    '''

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([''] + list(labels))
    for label, row in zip(labels, np.asarray(matrix, dtype=float)):
        writer.writerow([label] + [repr(float(value)) for value in row])
    return buffer.getvalue()


def _fmt(value, digits: int) -> str:
    if value is None or not math.isfinite(value):
        return '--'
    return f'{value:.{digits}f}'


def _fmt_stat(stat: Mapping[str, float], digits: int) -> str:
    mean, sd, n = stat.get('mean', NAN), stat.get('sd', NAN), stat.get('n', 0)
    if n == 0 or not math.isfinite(mean):
        return '--'
    if n < 2 or not math.isfinite(sd):
        return _fmt(mean, digits)
    return f'{mean:.{digits}f} $\\pm$ {sd:.{digits}f}'


def _seed_text(rows: Sequence[dict]) -> str:
    groups: list[tuple[str, str, int]] = []  # (first label, last label, n_seeds)
    for row in rows:
        if row['n_seeds'] == 0 or row['label'] in ('A', 'B'):
            continue
        if groups and groups[-1][2] == row['n_seeds']:
            groups[-1] = (groups[-1][0], row['label'], row['n_seeds'])
        else:
            groups.append((row['label'], row['label'], row['n_seeds']))
    parts = []
    for first, last, n in groups:
        label = first if first == last else f'{first}--{last}'
        parts.append(f'{label}: {n}')
    return ', '.join(parts)


def comparison_latex(case: str, rows: Sequence[dict], *, instance: str, n_iter: int) -> str:
    '''
    A booktabs table of the comparison for one case (rows of `comparison_rows`), sized for one column.
    '''

    header = (
        'Method & $c^\\top\\bar y$ & Gap to (A) [\\%] & $\\|\\bar y-y^B\\|_F$ & Partition effect & '
        '$\\|\\lambda-\\lambda^B\\|_2$ & Run-to-run & $m_{T-1}$ & Fluid cost \\\\'
    )
    body = []
    for row in rows:
        met, run = row['metrics'], row['run_to_run_y']
        partition_effect = _fmt(row['partition_effect'], 3)
        cells = [
            f'({row["label"]})',
            _fmt_stat(met['objective'], 2),
            _fmt_stat(met['gap_A_pct'], 2),
            _fmt_stat(met['dist_B'], 3),
            partition_effect,
            _fmt_stat(met['lam_dist_B'], 2),
            _fmt_stat(run, 3) if row['label'] not in ('A', 'B') else '--',
            _fmt_stat(met['mass_last'], 3),
            _fmt_stat(met['fluid_cost'], 2),
        ]
        body.append(' & '.join(cells) + ' \\\\')
    seeds = _seed_text(rows)
    seed_clause = f' ({seeds})' if seeds else ''
    caption = (
        f'Quantitative comparison for {case} ({instance}, $n_{{\\mathrm{{iter}}}}={n_iter:,}$). '
        'Entries use the tail averages $\\bar y$ and $\\bar\\lambda$. Learned methods show mean $\\pm$ sd over seeds'
        f'{seed_clause}. $y^B$ is the regularized interior-point solution, and $y^\\star$ is the exact '
        'saddle point of the method\'s (aggregate) problem. The partition effect $\\|y^\\star-y^B\\|_F$ contains no solver noise, '
        'but for (C) and (D) it is only the regularization bias. Run-to-run is the mean pairwise distance between '
        'independent seeds, which measures solver noise. '
        'Fluid cost evaluates the conditional policy in the true dynamics.'
    )
    lines = [
        f'% Generated by lplearn.analysis.metrics from the {instance} learner results (case {case}, n_iter={n_iter}).',
        '\\begin{table}[t]',
        '\\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{tab:metrics-{case.lower()}}}',
        '\\small',
        '\\setlength{\\tabcolsep}{2.5pt}',
        '\\resizebox{\\columnwidth}{!}{%',
        '\\begin{tabular}{l' + 'c' * 8 + '}',
        '\\toprule',
        header,
        '\\midrule',
        *body,
        '\\bottomrule',
        '\\end{tabular}}',
        '\\end{table}',
        '',
    ]
    return '\n'.join(lines)


__all__ = [
    'NAN', 'Problem', 'Reference', 'MethodRuns', 'comparison_latex', 'comparison_rows', 'feasibility',
    'fluid_evaluate', 'frobenius', 'legacy_to_detailed', 'matrix_to_csv', 'mean_sd', 'pairwise_frobenius',
    'pairwise_matrices', 'policy_distance', 'price_distance', 'reference_metrics', 'rows_to_csv', 'run_to_run',
    'solution_metrics', 'summarize_runs', 'to_jsonable',
]
