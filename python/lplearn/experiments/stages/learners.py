'''
Learners (methods (C)-(K)) on the detailed and the aggregate grid (stage `learners`).

A job runs one learner with one seed: GDA (method C, exact kernel; the seed only changes the random
initialization) or SGDA (methods D-K, estimated kernel with `samples_per_pair` draws per (u, s) pair). The job
stores the tail averages and the last iterates of y and of the resource prices lam, the total mass per stage,
and a trace of at most 200 records. Aggregate solutions are lifted to the detailed grid.\n
`summarize` writes three kinds of files into the summary directory:\n
- solutions_{instance}.npz: the consolidated solutions with keys '{case}/{method}/seed{seed}/{field}', where field
  is one of y_avg, y_last, lam_res_avg, lam_res_last (this is the file the finite-N agent loads);\n
- solutions_index.csv: one line per job, with its job id and hyperparameters;\n
- the comparison metrics against the exact references (recomputed with `exact_solution`, the same function as
  stage `lp_solutions`): comparison_{case}.csv, comparison_table.tex (manuscript table, K5T10), metrics_{instance}.json,
  pairwise_{case}.npz and lambda_{case}.npz, plus the CSV versions of the pairwise matrices.
'''


from __future__ import annotations

import csv
import hashlib
import io
import math
from importlib import metadata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import os

import numpy as np

from lplearn.analysis.metrics import (
    MethodRuns, Problem, Reference, comparison_latex, comparison_rows, feasibility, frobenius, legacy_to_detailed, matrix_to_csv,
    pairwise_matrices, reference_metrics, rows_to_csv, to_jsonable,
)
from lplearn.core import build_lp, queue_wcmdp
from lplearn.experiments.io import atomic_savez, atomic_write_json, atomic_write_text
from lplearn.experiments.methods import (
    INSTANCE_SIZES, LEARNER_DEFAULTS, LEGACY_EXTRA, LEGACY_FILES, METHODS_K5, METHODS_K11, NO_PARTITION, exact_solution,
    instance_params, method_spec, model_fields, params_from_job, partition_for, run_learner,
)
from lplearn.experiments.runner import ConfigError, job_id


NAME = 'learners'
VERSION = '4'  # 4: backend and raw occupation diagnostics in every job
TRACE_RECORDS = 200
FIELDS = ('y_avg', 'y_last', 'lam_res_avg', 'lam_res_last')
HYPER_KEYS = ('Gamma_y', 'Gamma_lambda', 'Gamma_mu', 'eta_y', 'eta_dual', 'n_iter', 'samples_per_pair', 'avg_window')
CASE_ORDER = ('CJS', 'SJS')


# ---------------------------------------------------------------------------
# Configuration and jobs
# ---------------------------------------------------------------------------


def _hyper(cfg: Mapping[str, Any], instance: str) -> dict[str, Any]:
    '''
    LEARNER_DEFAULTS for `instance`, overridden by the instance's [hyper] table in the config.
    '''

    block = cfg.get(instance) or {}
    hyper = dict(LEARNER_DEFAULTS[instance])
    override = block.get('hyper') or {}
    unknown = sorted(set(override) - set(HYPER_KEYS))
    if unknown:
        raise ConfigError(f'{instance}.hyper has unknown keys: {", ".join(unknown)}')
    hyper.update(override)
    return hyper


def make_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    '''
    For each instance table ([K5T10], [K11T30]) and case: the GDA seeds of method C (`gda_seeds`), and the SGDA
    seeds (`sgda_seeds`) of the methods in `sgda_methods`. Hyperparameters come from [INSTANCE.hyper].
    '''

    backend = cfg.get('backend', 'numba')
    if backend not in ('numba', 'rust'):
        raise ConfigError(f'learners: backend must be numba or rust, got {backend!r}')
    timeout = cfg.get('rust_timeout', 3600.0)
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout <= 0):
        raise ConfigError('learners: rust_timeout must be positive and finite')
    if backend == 'rust':
        try:
            from lplearn.rust import _executable
            binary_hash = hashlib.sha256(Path(_executable()).read_bytes()).hexdigest()
        except (OSError, RuntimeError) as exc:
            raise ConfigError(f'learners: Rust backend unavailable: {exc}') from exc
        backend_identity = {'backend_binary_sha256': binary_hash}
    else:
        backend_identity = {'backend_version': metadata.version('numba')}
    jobs = []
    for instance, (K, T) in INSTANCE_SIZES.items():
        if instance not in cfg:
            continue
        block = cfg[instance]
        hyper = _hyper(cfg, instance)
        n_iter = int(hyper['n_iter'])
        if n_iter < 1:
            raise ConfigError(f'{instance}.hyper.n_iter must be positive')
        avg_window = min(int(hyper['avg_window']), n_iter)
        record_every = max(1, n_iter // TRACE_RECORDS)
        base = {
            'instance': instance, 'K': K, 'T': T,
            'Gamma_y': float(hyper['Gamma_y']), 'Gamma_lambda': float(hyper['Gamma_lambda']), 'Gamma_mu': float(hyper['Gamma_mu']),
            'eta_y': float(hyper['eta_y']), 'eta_dual': float(hyper['eta_dual']), 'n_iter': n_iter,
            'avg_window': avg_window, 'record_every': record_every, **model_fields(cfg),
            'aggregate_regularization': str(cfg.get('aggregate_regularization', 'inherited')),
            'backend': backend,
            **backend_identity,
        }
        if backend == 'rust':
            base['rust_timeout'] = float(timeout)
        for case in block.get('cases', ['CJS']):
            for seed in block.get('gda_seeds', []):
                jobs.append(_job(base, case, 'C', int(seed), samples=0, spec=method_spec(instance, 'C')))
            for label in block.get('sgda_methods', []):
                spec = method_spec(instance, label)
                if spec['solver'] != 'sgda':
                    raise ConfigError(f'{instance}: method {label} is not an SGDA method; list it under gda_seeds instead')
                for seed in block.get('sgda_seeds', []):
                    jobs.append(_job(base, case, label, int(seed), samples=int(hyper['samples_per_pair']), spec=spec))
    if not jobs:
        raise ConfigError('learners: the config selects no jobs')
    return jobs


def _job(base: Mapping[str, Any], case: str, label: str, seed: int, *, samples: int, spec: Mapping[str, str]) -> dict[str, Any]:
    if seed < 0:
        raise ConfigError('seeds must be non-negative')
    return {**base, 'case': case, 'method': label, 'solver': spec['solver'], 'partition': spec['partition'], 'seed': seed,
            'samples_per_pair': samples}


# ---------------------------------------------------------------------------
# Running one learner
# ---------------------------------------------------------------------------


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    '''
    One GDA or SGDA run. Returns the averages and last iterates of y and lam_res on the detailed grid, the mass per
    stage, the trace (at most 200 records) and the wall time.
    '''

    K, T = int(job['K']), int(job['T'])
    params = params_from_job(job)
    partition = partition_for(job['partition'], K)
    res = run_learner(
        params, str(job['solver']), partition, int(job['seed']),
        Gamma_y=float(job['Gamma_y']), Gamma_lambda=float(job['Gamma_lambda']), Gamma_mu=float(job['Gamma_mu']),
        eta_y=float(job['eta_y']), eta_dual=float(job['eta_dual']), n_iter=int(job['n_iter']),
        samples_per_pair=int(job['samples_per_pair']), avg_window=int(job['avg_window']), record_every=int(job['record_every']),
        aggregate_regularization=str(job.get('aggregate_regularization', 'inherited')),
        backend=str(job.get('backend', 'numba')), rust_timeout=float(job.get('rust_timeout', 3600.0)),
    )
    for name, value in res.items():
        if isinstance(value, (np.ndarray, float, np.floating)) and not np.isfinite(value).all():
            raise RuntimeError(f'learner returned non-finite values in {name}')
    y_avg = np.asarray(res['y_avg'], dtype=float)
    y_last = np.asarray(res['y_last'], dtype=float)
    diagnostics = feasibility(build_lp(queue_wcmdp(params)), y_avg)
    if any(not np.isfinite(value).all() for value in diagnostics.values()):
        raise RuntimeError('learner occupation diagnostics are non-finite')
    keep = slice(0, len(res['trace_iter']), max(1, math.ceil(len(res['trace_iter']) / TRACE_RECORDS)))
    backend = str(job.get('backend', 'numba'))
    version = str(res['backend_version']) if backend == 'rust' else metadata.version('numba')
    if backend == 'rust' and job.get('backend_binary_sha256', res['backend_binary_sha256']) != res['backend_binary_sha256']:
        raise RuntimeError('Rust backend binary differs from planned job')
    if backend == 'numba' and job.get('backend_version', version) != version:
        raise RuntimeError('Numba version differs from planned job')
    result = {
        'y_avg': y_avg,
        'y_last': y_last,
        'lam_res_avg': np.asarray(res['lam_res_avg'], dtype=float),
        'lam_res_last': np.asarray(res['lam_res_last'], dtype=float),
        'mass_avg': y_avg.sum(axis=(1, 2)),
        'mass_last': y_last.sum(axis=(1, 2)),
        'trace_iter': np.asarray(res['trace_iter'])[keep],
        'trace_y': np.asarray(res['trace_y'])[keep],
        'trace_lam_res': np.asarray(res['trace_lam_res'])[keep],
        'trace_mu_norm': np.asarray(res['trace_mu_norm'])[keep],
        'n_avg': int(res['n_avg']),
        'n_iter': int(res['n_iter']),
        'wall_time': float(res['wall_time']),
        'wall_time_scope': 'rust-process-json' if backend == 'rust' else 'numba-run-including-jit',
        'backend': backend,
        'backend_version': version,
        'occupation_min': float(y_avg.min()),
        'flow_violation': float(diagnostics['flow_violation']),
        'budget_violation': float(diagnostics['budget_violation']),
        'nonneg_violation': float(diagnostics['nonneg_violation']),
    }
    if backend == 'rust':
        result.update(backend_binary_sha256=res['backend_binary_sha256'],
                      backend_debug_assertions=res['debug_assertions'],
                      backend_execute_seconds=float(res['execute_seconds']))
    return result


# ---------------------------------------------------------------------------
# Summaries: consolidated solutions, index, comparison metrics
# ---------------------------------------------------------------------------


def solution_key(job: Mapping[str, Any], field: str) -> str:
    '''
    Key of one array in solutions_{instance}.npz.
    '''

    return f'{job["case"]}/{job["method"]}/seed{int(job["seed"])}/{field}'


def consolidate_solutions(records: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, np.ndarray]:
    '''
    The four solution fields of every record under the keys of `solution_key`. Duplicate keys are refused.
    '''

    arrays: dict[str, np.ndarray] = {}
    for job, res in sorted(records, key=lambda record: (record[0]['case'], record[0]['method'], int(record[0]['seed']))):
        for field in FIELDS:
            key = solution_key(job, field)
            if key in arrays:
                raise ConfigError(f'two jobs produce the same solution {key!r}; use one hyperparameter set per summary')
            arrays[key] = np.asarray(res[field])
    return arrays


def _csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, '')
            if isinstance(value, (float, np.floating)):
                value = '' if not math.isfinite(float(value)) else repr(float(value))
            values.append(value)
        writer.writerow(values)
    return buffer.getvalue()


INDEX_COLUMNS = (
    'instance', 'case', 'method', 'solver', 'backend', 'backend_version', 'backend_binary_sha256', 'partition', 'seed', 'job_id', 'n_iter', 'samples_per_pair', 'avg_window',
    'eta_y', 'eta_dual', 'Gamma_y', 'Gamma_lambda', 'Gamma_mu', 'n_avg', 'mass_last', 'occupation_min', 'flow_violation', 'budget_violation', 'nonneg_violation', 'wall_time_s', 'wall_time_scope',
)


def _index_row(job: Mapping[str, Any], res: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'instance': job['instance'], 'case': job['case'], 'method': job['method'], 'solver': job['solver'],
        'backend': job.get('backend', 'numba'), 'backend_version': res['backend_version'],
        'backend_binary_sha256': res.get('backend_binary_sha256', ''),
        'partition': job['partition'], 'seed': int(job['seed']), 'job_id': job_id(NAME, VERSION, dict(job)),
        'n_iter': int(job['n_iter']), 'samples_per_pair': int(job['samples_per_pair']), 'avg_window': int(job['avg_window']),
        'eta_y': float(job['eta_y']), 'eta_dual': float(job['eta_dual']), 'Gamma_y': float(job['Gamma_y']),
        'Gamma_lambda': float(job['Gamma_lambda']), 'Gamma_mu': float(job['Gamma_mu']), 'n_avg': int(res['n_avg']),
        'mass_last': float(np.asarray(res['mass_last'])[-1]), 'occupation_min': float(res['occupation_min']),
        'flow_violation': float(res['flow_violation']), 'budget_violation': float(res['budget_violation']),
        'nonneg_violation': float(res['nonneg_violation']), 'wall_time_s': float(res['wall_time']),
        'wall_time_scope': res['wall_time_scope'],
    }


def _method_order(instance: str) -> tuple[str, ...]:
    table = METHODS_K5 if INSTANCE_SIZES[instance][0] == 5 else METHODS_K11
    return tuple(table)


def build_method_runs(instance: str, case: str, records: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> list[MethodRuns]:
    '''
    The learned methods of one case, in table order, with their seeds sorted.
    '''

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for job, res in records:
        if job['case'] == case:
            grouped[job['method']].append((int(job['seed']), res))
    methods = []
    for label in _method_order(instance):
        if label not in grouped:
            continue
        items = sorted(grouped[label], key=lambda item: item[0])
        spec = method_spec(instance, label)
        methods.append(MethodRuns(
            label=label, solver=spec['solver'], partition=spec['partition'],
            seeds=tuple(seed for seed, _ in items),
            y=tuple(np.asarray(res['y_avg'], dtype=float) for _, res in items),
            lam=tuple(np.asarray(res['lam_res_avg'], dtype=float) for _, res in items),
            y_last=tuple(np.asarray(res['y_last'], dtype=float) for _, res in items),
            lam_last=tuple(np.asarray(res['lam_res_last'], dtype=float) for _, res in items),
        ))
    return methods


def _legacy_dir(cfg: Mapping[str, Any], instance: str) -> Path | None:
    legacy = cfg.get('legacy', {})
    # The stored legacy arrays live outside the repository: LPL_LEGACY_DIR overrides the config's root.
    root, name = os.environ.get('LPL_LEGACY_DIR') or legacy.get('root'), legacy.get(instance)
    if not root or not name:
        return None
    return Path(root) / name


def _load_legacy(instance: str, case: str, legacy_dir: Path | None) -> dict[str, np.ndarray]:
    '''
    Stored legacy arrays of one case by method label (the detailed grid). Missing files are skipped.
    '''

    out: dict[str, np.ndarray] = {}
    if legacy_dir is None or not legacy_dir.is_dir():
        return out
    K, T = INSTANCE_SIZES[instance]
    for label, stems in LEGACY_FILES.get(instance, {}).items():
        stem = stems.get(case)
        path = legacy_dir / f'{stem}.npy' if stem else None
        if path is not None and path.is_file():
            arr = legacy_to_detailed(np.load(path, allow_pickle=False))
            if arr.shape != (T, K + 1, 4):
                raise ValueError(f'{path} has shape {arr.shape}, expected {(T, K + 1, 4)} on the detailed grid')
            out[label] = arr
    return out


def _load_legacy_extra(instance: str, case: str, legacy_dir: Path | None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if legacy_dir is None or not legacy_dir.is_dir():
        return out
    for name, stems in LEGACY_EXTRA.get(instance, {}).items():
        stem = stems.get(case)
        path = legacy_dir / f'{stem}.npy' if stem else None
        if path is not None and path.is_file():
            out[name] = legacy_to_detailed(np.load(path, allow_pickle=False))
    return out


def build_problem(instance: str, case: str, hyper: Mapping[str, Any], methods: Sequence[MethodRuns], legacy: Mapping[str, np.ndarray],
                  job: Mapping[str, Any] | None = None) -> Problem:
    '''
    The exact references of one case: (A) the LP, (B) the regularized LP, and the exact saddle point of the
    detailed problem and of every partition used by `methods` (the target of their solver). The references come
    from `exact_solution`, so they do not depend on any stored results.
    '''

    K, T = INSTANCE_SIZES[instance]
    params = params_from_job(job) if job is not None else instance_params(K, T, case)  # the model of the recorded jobs
    gamma = {key: float(hyper[key]) for key in ('Gamma_y', 'Gamma_lambda', 'Gamma_mu')}
    refs: dict[str, Reference] = {}
    lp = exact_solution(params, 'lp')
    refs['A'] = Reference(np.array(lp['y']), np.array(lp['lam_res']), float(lp['objective']))
    reg = exact_solution(params, 'lp_reg', Gamma_y=gamma['Gamma_y'])
    refs['B'] = Reference(np.array(reg['y']), np.array(reg['lam_res']), float(reg['objective']))
    for partition in sorted({m.partition for m in methods} | {NO_PARTITION}):  # the detailed saddle point is always needed
        sol = exact_solution(params, 'saddle', partition_for(partition, K), **gamma)
        refs[f'star:{partition}'] = Reference(np.array(sol['y']), np.array(sol['lam_res']), float(sol['objective']))
    model = queue_wcmdp(params)
    return Problem(struct=build_lp(model), model=model, refs=refs, legacy=dict(legacy))


def _lambda_arrays(problem: Problem, methods: Sequence[MethodRuns]) -> dict[str, np.ndarray]:
    labels = ['A', 'B'] + [m.label for m in methods]
    shape = problem.refs['B'].lam.shape
    nan = np.full(shape, np.nan)
    means = [problem.refs['A'].lam, problem.refs['B'].lam]
    sds = [nan, nan]
    stars = [nan, nan]
    for m in methods:
        stack = np.stack(m.lam)
        means.append(stack.mean(axis=0))
        sds.append(stack.std(axis=0, ddof=1) if len(m.lam) > 1 else nan)
        stars.append(problem.refs[m.target].lam)
    return {
        'labels': np.array(labels),
        'lam_mean': np.stack(means),
        'lam_sd': np.stack(sds),
        'lam_star': np.stack(stars),
        'lam_B': problem.refs['B'].lam,
    }


def _suffix(instance: str, case: str) -> str:
    return case if instance == 'K5T10' else f'{instance}_{case}'


def write_metrics(instance: str, records: Sequence[tuple[dict[str, Any], dict[str, Any]]], cfg: Mapping[str, Any], out_dir: Path) -> str:
    '''
    Comparison metrics of every case of `instance`. Writes the CSV, npz and JSON files, and returns the LaTeX
    table block of each case (the caller assembles comparison_table.tex).
    '''

    hyper = _hyper(cfg, instance)
    legacy_dir = _legacy_dir(cfg, instance)
    report: dict[str, Any] = {
        'instance': instance,
        'hyper': {key: hyper[key] for key in HYPER_KEYS},
        'legacy_dir': str(legacy_dir) if legacy_dir is not None else None,
        'legacy_found': bool(legacy_dir is not None and legacy_dir.is_dir()),
        'cases': {},
    }
    latex_blocks: list[str] = []
    for case in CASE_ORDER:
        case_records = [(job, res) for job, res in records if job['case'] == case]
        if not case_records:
            continue
        methods = build_method_runs(instance, case, case_records)
        legacy = _load_legacy(instance, case, legacy_dir)
        problem = build_problem(instance, case, hyper, methods, legacy, job=case_records[0][0])
        rows = comparison_rows(problem, methods)
        pairwise = pairwise_matrices(problem, methods)
        lam = _lambda_arrays(problem, methods)
        suffix = _suffix(instance, case)

        atomic_write_text(out_dir / f'comparison_{suffix}.csv', rows_to_csv(rows))
        atomic_write_text(out_dir / f'pairwise_{suffix}_seed_mean.csv', matrix_to_csv(pairwise['labels'], pairwise['seed_mean']))
        atomic_write_text(out_dir / f'pairwise_{suffix}_seed0.csv', matrix_to_csv(pairwise['labels'], pairwise['seed0']))
        atomic_savez(out_dir / f'pairwise_{suffix}.npz', {
            'labels': np.array(pairwise['labels']), 'seed_mean': pairwise['seed_mean'], 'seed0': pairwise['seed0'],
        })
        atomic_savez(out_dir / f'lambda_{suffix}.npz', lam)

        extra = {}
        d_mean = next((m for m in methods if m.label == 'D'), None)
        for name, arr in _load_legacy_extra(instance, case, legacy_dir).items():
            if d_mean is not None:
                extra[f'legacy_{name}_vs_mean_D'] = frobenius(np.mean(np.stack(d_mean.y), axis=0), arr)
        report['cases'][case] = {
            'references': reference_metrics(problem),
            'rows': rows,
            'pairwise': {'labels': pairwise['labels'], 'seed_mean': pairwise['seed_mean'], 'seed0': pairwise['seed0']},
            'legacy_labels': sorted(legacy),
            'extra_legacy': extra,
        }
        latex_blocks.append(comparison_latex(case, rows, instance=instance, n_iter=int(hyper['n_iter'])))
    atomic_write_json(out_dir / f'metrics_{instance}.json', to_jsonable(report))
    return '\n'.join(latex_blocks)


def summarize(records: Sequence[tuple[dict[str, Any], dict[str, Any]]], cfg: dict[str, Any], out_dir: Path) -> None:
    '''
    Consolidated solutions, the job index and the comparison metrics (see the module docstring).
    '''

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for job, res in records:
        grouped[str(job['instance'])].append((job, res))
    index_rows = []
    for instance in INSTANCE_SIZES:
        if instance not in grouped:
            continue
        atomic_savez(out_dir / f'solutions_{instance}.npz', consolidate_solutions(grouped[instance]))
        index_rows += [_index_row(job, res) for job, res in sorted(grouped[instance], key=lambda r: (r[0]['case'], r[0]['method'], int(r[0]['seed'])))]
    atomic_write_text(out_dir / 'solutions_index.csv', _csv(index_rows, INDEX_COLUMNS))

    tables = []
    for instance in INSTANCE_SIZES:
        if instance in grouped:
            block = write_metrics(instance, grouped[instance], cfg, out_dir)
            if instance == 'K5T10' and block:
                tables.append(block)
    if tables:
        header = '% Manuscript table of the quantitative comparison (Section 8). Regenerate with `lplearn experiments summarize learners`.\n'
        atomic_write_text(out_dir / 'comparison_table.tex', header + '\n'.join(tables))
