'''
Deterministic reference solutions of the relaxed LP (stage `lp_solutions`).

A job solves one reference problem on one instance, case and partition:\n
- kind 'lp': the unregularized LP (method (A) on the detailed grid);\n
- kind 'lp_reg': the LP with Gamma_y/2 ||y||^2 and hard constraints, by interior point (method (B));\n
- kind 'saddle': the exact fixed point of the regularized operator G (the point that GDA and SGDA converge to).\n
With a partition the problem is the aggregate one, and the solution is lifted to the detailed grid.
The jobs are independent and fast, and their arrays feed the metrics of the `learners` stage.
'''


from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from lplearn.analysis.metrics import feasibility, fluid_evaluate
from lplearn.core import build_lp, queue_wcmdp
from lplearn.experiments.io import atomic_write_text
from lplearn.experiments.methods import INSTANCE_SIZES, LEARNER_DEFAULTS, NO_PARTITION, exact_solution, model_fields, params_from_job, partition_for
from lplearn.experiments.runner import ConfigError


NAME = 'lp_solutions'
VERSION = '3'  # 2: model parameters stored in every job; 3: aggregate regularization inherited (Section 6)
KINDS = ('lp', 'lp_reg', 'saddle')
CASES = ('CJS', 'SJS')
SMOKE: dict[str, Any] = {}


def make_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    '''
    Grid over instance, case, reference kind and partition. Configs name the instances as tables
    ([K5T10], [K11T30]) with `cases` and `partitions`; the top-level `kinds`, `k11_sjs` and `Gamma_*` apply to all.
    '''

    kinds = list(cfg.get('kinds', KINDS))
    model = model_fields(cfg)
    jobs = []
    for instance, (K, T) in INSTANCE_SIZES.items():
        if instance not in cfg:
            continue
        block = cfg[instance]
        defaults = LEARNER_DEFAULTS[instance]
        gamma = {
            'Gamma_y': float(cfg.get('Gamma_y', defaults['Gamma_y'])),
            'Gamma_lambda': float(cfg.get('Gamma_lambda', defaults['Gamma_lambda'])),
            'Gamma_mu': float(cfg.get('Gamma_mu', defaults['Gamma_mu'])),
        }
        cases = list(block.get('cases', ['CJS']))
        if instance == 'K11T30' and cfg.get('k11_sjs', False) and 'SJS' not in cases:
            cases.append('SJS')
        for case, kind, label in product(cases, kinds, block.get('partitions', [NO_PARTITION])):
            if case not in CASES or kind not in KINDS:
                raise ConfigError(f'lp_solutions: unknown case {case!r} or kind {kind!r}')
            try:
                partition_for(label, K)
            except KeyError as exc:
                raise ConfigError(f'lp_solutions: {exc.args[0]}') from exc
            jobs.append({'instance': instance, 'K': K, 'T': T, 'case': case, 'kind': kind, 'partition': label, **gamma, **model,
                         'aggregate_convention': str(cfg.get('aggregate_convention', 'consistent'))})
    if not jobs:
        raise ConfigError('lp_solutions: the config selects no jobs')
    return jobs


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    '''
    Solve one reference problem and report its solution on the detailed grid, with feasibility, fluid
    evaluation and solver information.
    '''

    K, T, case = int(job['K']), int(job['T']), str(job['case'])
    params = params_from_job(job)
    partition = partition_for(job['partition'], K)
    start = perf_counter()
    sol = exact_solution(
        params, job['kind'], partition,
        Gamma_y=job['Gamma_y'], Gamma_lambda=job['Gamma_lambda'], Gamma_mu=job['Gamma_mu'],
        convention=str(job.get('aggregate_convention', 'consistent')),
    )
    wall = perf_counter() - start

    model = queue_wcmdp(params)
    y = np.array(sol['y'], dtype=float)
    feas = feasibility(build_lp(model), y)
    fluid = fluid_evaluate(model, y)
    if job['kind'] == 'saddle':
        residual_kind, residual = 'fixed_point_residual', float(sol['fixed_point_residual'])
    else:
        residual_kind, residual = 'kkt_stationarity', float(sol['kkt_stationarity'])
    return {
        'y': y,
        'y_native': np.array(sol['y_native'], dtype=float),
        'lam_res': np.array(sol['lam_res'], dtype=float),
        'objective': float(sol['objective']),
        'mass': y.sum(axis=(1, 2)),
        'flow_violation': feas['flow_violation'],
        'budget_violation': feas['budget_violation'],
        'nonneg_violation': feas['nonneg_violation'],
        'fluid_cost': fluid['fluid_cost'],
        'fluid_excess': fluid['fluid_excess'],
        'residual': residual,
        'residual_kind': residual_kind,
        'converged': bool(sol.get('newton_converged', True)),  # LP solves raise unless the status is optimal
        'solver': str(sol['solver']),
        'solve_time': float(sol.get('solve_time', np.nan)),
        'wall_time': float(wall),
    }


def summarize(records: Sequence[tuple[dict[str, Any], dict[str, Any]]], cfg: dict[str, Any], out_dir: Path) -> None:
    '''
    Write lp_solutions_summary.csv: one line per solved problem with the scalar results.
    '''

    columns = [
        'instance', 'case', 'kind', 'partition', 'objective', 'mass_t0', 'mass_last', 'flow_violation',
        'budget_violation', 'nonneg_violation', 'fluid_cost', 'fluid_excess', 'residual_kind', 'residual',
        'converged', 'solver', 'solve_time', 'wall_time',
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    ordered = sorted(records, key=lambda record: (record[0]['instance'], record[0]['case'], record[0]['kind'], record[0]['partition']))
    for job, res in ordered:
        mass = np.asarray(res['mass'], dtype=float)
        writer.writerow([
            job['instance'], job['case'], job['kind'], job['partition'], repr(float(res['objective'])),
            repr(float(mass[0])), repr(float(mass[-1])), repr(float(res['flow_violation'])),
            repr(float(res['budget_violation'])), repr(float(res['nonneg_violation'])), repr(float(res['fluid_cost'])),
            repr(float(res['fluid_excess'])), res['residual_kind'], repr(float(res['residual'])), bool(res['converged']),
            res['solver'], repr(float(res['solve_time'])), repr(float(res['wall_time'])),
        ])
    atomic_write_text(out_dir / 'lp_solutions_summary.csv', buffer.getvalue())
