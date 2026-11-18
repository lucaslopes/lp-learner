'''
Methods (A)-(K) of Table `tab:methods`, their reference solutions and the learner entry points,
in the manuscript's regularization convention (Gamma/2 ||.||^2).\n
Legacy runs used Gamma_y = 0.5 and Gamma_lambda = Gamma_mu = 1e-5 in the legacy convention
(Gamma ||.||^2), i.e. Gamma_y = 1 and Gamma_lambda = Gamma_mu = 2e-5 here.\n
`LEGACY_FILES` names the stored legacy arrays (`policies_questa_*` directories under the legacy root, see LPL_LEGACY_DIR)
that the metrics compare against. Their layout is (S, 2, 2, T) indexed by [s, a, b, t].
'''


from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

from lplearn.core import (
    PARTITION_K11, PARTITIONS_K5, QueueParams, Regularization, aggregate, block_sizes, build_lp, lift_y, maps, queue_wcmdp,
    regularization_vector, regularized_saddle_point, run_gda, saddle_maps, singleton_partition, solve_lp,
)


METHOD_LABELS = tuple('ABCDEFGHIJK')
NO_PARTITION = 'none'  # partition label of the detailed grid in configs and job records

# solver: 'lp' (interior point, no regularization), 'lp_reg' (interior point, primal regularization),
# 'gda' (exact kernel) or 'sgda' (estimated kernel); partition: None = detailed.
METHODS_K5 = {
    'A': {'solver': 'lp', 'partition': None},
    'B': {'solver': 'lp_reg', 'partition': None},
    'C': {'solver': 'gda', 'partition': None},
    'D': {'solver': 'sgda', 'partition': None},
    **{label: {'solver': 'sgda', 'partition': part} for label, part in PARTITIONS_K5.items()},
}
# Pairs partition {0,1},{2,3},...,{10,11} of the legacy K11T30 figure (legacy array sgda_agg_c_1).
PARTITION_K11_PAIRS = tuple((i, i + 1) for i in range(0, 12, 2))

# K = 11 has no (A) or (B) entry in the learner stage; its aggregate methods are the boundary-aware K11 partition
# and the pairs partition P2 shown in the legacy figure.
METHODS_K11 = {
    'C': {'solver': 'gda', 'partition': None},
    'D': {'solver': 'sgda', 'partition': None},
    'K11': {'solver': 'sgda', 'partition': PARTITION_K11},
    'P2': {'solver': 'sgda', 'partition': PARTITION_K11_PAIRS},
}

INSTANCE_SIZES = {'K5T10': (5, 10), 'K11T30': (11, 30)}
INSTANCE_K5T10 = QueueParams(K=5, T=10)
INSTANCE_K11T30 = QueueParams(K=11, T=30)

# Hyperparameters of the legacy QUESTA runs (reports/policies_K5T10.py, policies_K11T30.py),
# translated to the manuscript convention. The legacy oracle stopping rule
# ||y - y_cvxpy||_F <= tol is not used: every run has a fixed iteration budget.
LEARNER_DEFAULTS = {
    'K5T10': {'Gamma_y': 1.0, 'Gamma_lambda': 2e-5, 'Gamma_mu': 2e-5, 'eta_y': 0.9, 'eta_dual': 0.03,
              'n_iter': 2_000_000, 'samples_per_pair': 10, 'avg_window': 100_000},
    'K11T30': {'Gamma_y': 1.0, 'Gamma_lambda': 2e-5, 'Gamma_mu': 2e-5, 'eta_y': 0.8, 'eta_dual': 0.02,
               'n_iter': 10_000_000, 'samples_per_pair': 10, 'avg_window': 100_000},
}

# Legacy file stems (without .npy) per instance, label and case (CJS = case 1, SJS = case 2).
# Legacy SGDA/GDA arrays are the last stored iterate of the legacy runs, i.e. for SGDA the average
# of the last 100,000 iterates (the legacy `ys_hist_size`), and they use the legacy aggregate costs.
LEGACY_FILES = {
    'K5T10': {
        'A': {'CJS': 'cvxpy_unregul_1', 'SJS': 'cvxpy_unregul_2'},
        'B': {'CJS': 'cvxpy_1', 'SJS': 'cvxpy_2'},
        'C': {'CJS': 'gda_agg_a_1', 'SJS': 'gda_agg_a_2'},
        'D': {'CJS': 'sgda_agg_a_1', 'SJS': 'sgda_agg_a_2'},
        'E': {'CJS': 'sgda_agg_b_1', 'SJS': 'sgda_agg_b_2'},
        'F': {'CJS': 'sgda_agg_c_1', 'SJS': 'sgda_agg_c_2'},
        'G': {'CJS': 'sgda_agg_d_1', 'SJS': 'sgda_agg_d_2'},
        'H': {'CJS': 'sgda_agg_e_1', 'SJS': 'sgda_agg_e_2'},
        'I': {'CJS': 'sgda_agg_f_1', 'SJS': 'sgda_agg_f_2'},
        'J': {'CJS': 'sgda_agg_g_1', 'SJS': 'sgda_agg_g_2'},
        'K': {'CJS': 'sgda_agg_h_1', 'SJS': 'sgda_agg_h_2'},
    },
    'K11T30': {
        'A': {'CJS': 'cvxpy_unregul_1', 'SJS': 'cvxpy_unregul_2'},
        'B': {'CJS': 'cvxpy_1', 'SJS': 'cvxpy_2'},
        'D': {'CJS': 'sgda_agg_a_1', 'SJS': 'sgda_agg_a_2'},
        'K11': {'CJS': 'sgda_agg_w_1'},  # [0], [1,2], [3,4,5], [6,7,8], [9,10], [11] = PARTITION_K11
        'P2': {'CJS': 'sgda_agg_c_1'},  # [0,1], [2,3], ..., [10,11] = PARTITION_K11_PAIRS
    },
}
# Extra legacy references that are not methods of the table (reported in the JSON only).
LEGACY_EXTRA = {
    'K11T30': {'D_10M': {'CJS': 'sgda_agg_a_1_10M'}},  # legacy SGDA on the detailed grid, 10M iterations
}


def instance_params(K: int, T: int, case: str, **overrides) -> QueueParams:
    return replace(QueueParams(K=K, T=T, case=case), **overrides)


MODEL_KEYS = ('p', 'b_low', 'b_high', 'alpha', 'beta', 'gamma', 'gamma_bar')


def model_fields(cfg: Mapping[str, Any]) -> dict[str, Any]:
    '''
    Model parameters to store in every job dict: the optional [model] table of a config over the QueueParams
    defaults, plus the initial distribution ('empty': all queues start empty). Storing them makes cached results
    depend on the model, not only on the code defaults.
    '''
    table = dict(cfg.get('model') or {})
    unknown = sorted(set(table) - set(MODEL_KEYS))
    if unknown:
        raise ValueError(f'[model] has unknown keys: {", ".join(unknown)}')
    base = QueueParams()
    fields = {key: float(table.get(key, getattr(base, key))) for key in MODEL_KEYS}
    fields['m0'] = 'empty'
    return fields


def params_from_job(job: Mapping[str, Any]) -> QueueParams:
    '''The QueueParams of a job dict built with `model_fields` (falls back to the defaults for older jobs).'''
    if job.get('m0', 'empty') != 'empty':
        raise ValueError(f"unsupported initial distribution {job.get('m0')!r}")
    base = QueueParams()
    return QueueParams(K=int(job['K']), T=int(job['T']), case=str(job['case']),
                       **{key: float(job.get(key, getattr(base, key))) for key in MODEL_KEYS})


def partition_for(label: str | None, K: int):
    '''Partition of the label used in configs and job records (None or "none" = detailed grid).'''
    if label is None or label == NO_PARTITION:
        return None
    if K == 5 and label in PARTITIONS_K5:
        return PARTITIONS_K5[label]
    if K == 11 and label == 'K11':
        return PARTITION_K11
    if K == 11 and label == 'P2':
        return PARTITION_K11_PAIRS
    raise KeyError(f'unknown partition {label!r} for K={K}')


def method_spec(instance: str, label: str) -> dict:
    '''Solver and partition label of method `label` for `instance` ("none" for the detailed grid).'''
    K = INSTANCE_SIZES[instance][0]
    table = METHODS_K5 if K == 5 else METHODS_K11
    if label not in table:
        raise KeyError(f'method {label!r} is not defined for {instance}')
    spec = table[label]
    return {'method': label, 'solver': spec['solver'], 'partition': NO_PARTITION if spec['partition'] is None else label}


def _model(params: QueueParams, partition):
    detailed = queue_wcmdp(params)
    if partition is None:
        return detailed, detailed, np.eye(detailed.S)
    U, _ = maps(partition, detailed.S)
    return detailed, aggregate(detailed, partition), U


def _lift(result: dict, U: np.ndarray) -> dict:
    '''Lift the occupation measures of a learner result (last, average and traced iterates) to the detailed grid.'''
    out = dict(result)
    for key in ('y_last', 'y_avg'):
        out[key] = lift_y(result[key], U)
    out['trace_y'] = np.einsum('sj,ntju->ntsu', U, result['trace_y'])
    return out


# Aggregate problems are regularized as in Section 6 of the manuscript: the regularization of the detailed problem
# restricted to block-constant vectors ('consistent' maps; Gamma_y/|P| on block mass, Gamma |P| on block prices).
AGGREGATE_CONVENTION = 'consistent'


@lru_cache(maxsize=256)
def exact_solution(params: QueueParams, kind: str, partition=None, Gamma_y: float = 1.0,
                   Gamma_lambda: float = 2e-5, Gamma_mu: float = 2e-5, convention: str = AGGREGATE_CONVENTION) -> dict:
    '''
    Deterministic reference solutions on the detailed grid (aggregate ones are lifted by U):\n
    - 'lp': unregularized LP (method A, or its aggregate analogue for a partition);\n
    - 'lp_reg': LP with Gamma_y/2 ||y||^2 and hard constraints (method B / aggregate analogue);\n
    - 'saddle': exact fixed point of G for (Gamma_y, Gamma_lambda, Gamma_mu), i.e. the point GDA/SGDA
      converge to (for a partition, the aggregate saddle point with the `convention` regularization).\n
    Returned arrays are read-only; copy before modifying.
    '''
    detailed, model, U = _model(params, partition)
    struct = build_lp(model)
    if kind in ('lp', 'lp_reg'):
        gamma_y = Gamma_y if kind == 'lp_reg' else 0.0
        if kind == 'lp_reg' and partition is not None and convention == 'consistent':
            sizes = block_sizes(partition)
            gamma_y = Gamma_y / np.broadcast_to(sizes[None, :, None], (model.T, model.S, model.U)).ravel()
        sol = solve_lp(struct, Gamma_y=gamma_y)
        y, lam_res = sol['y'], sol['lam_res']
        extra = {'solver': sol['solver'], 'solve_time': sol['solve_time'], 'kkt_stationarity': sol['kkt_stationarity']}
    elif kind == 'saddle':
        reg = Regularization(Gamma_y, Gamma_lambda, Gamma_mu)
        if partition is None:
            R = regularization_vector(struct, reg)
        else:
            R = saddle_maps(build_lp(detailed), struct, partition, convention).aggregate_regularization(reg)
        warm = solve_lp(struct, Gamma_y=Gamma_y)['y']
        sol = regularized_saddle_point(struct, R, y0=warm)
        parts = struct.split(sol['w'])
        y, lam_res = parts['y'], parts['lam_res']
        extra = {
            'solver': 'semismooth-newton (warm start: CLARABEL)', 'fixed_point_residual': sol['fixed_point_residual'],
            'newton_iterations': sol['iterations'], 'newton_converged': sol['converged'], 'w': sol['w'],
        }
    else:
        raise ValueError(f'unknown kind {kind!r}')
    y_lift = lift_y(y, U) if partition is not None else y
    out = {'y': y_lift, 'y_native': y, 'lam_res': lam_res, 'objective': float(build_lp(detailed).objective(y_lift)), **extra}
    for value in out.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return out


def run_learner(params: QueueParams, solver: str, partition=None, seed: int = 0,
                *, backend: str = 'numba', rust_timeout: float = 3600.0, **hyper) -> dict:
    '''
    Run GDA ('gda') or SGDA ('sgda') for a detailed or aggregate method and lift y to the detailed grid.\n
    `hyper` holds Gamma_y, Gamma_lambda, Gamma_mu, eta_y, eta_dual, n_iter and optionally samples_per_pair
    (SGDA only, default 10), avg_window, record_every and init (see run_gda).
    '''
    if solver not in ('gda', 'sgda'):
        raise ValueError(f"solver must be 'gda' or 'sgda', got {solver!r}")
    if backend not in ('numba', 'rust'):
        raise ValueError(f"backend must be 'numba' or 'rust', got {backend!r}")
    _, model, U = _model(params, partition)
    hp = dict(hyper)
    inherited = hp.pop('aggregate_regularization', 'inherited') == 'inherited'
    if partition is not None and inherited:
        hp['state_weights'] = block_sizes(partition)  # Section 6: regularization inherited from the detailed problem
    samples = int(hp.pop('samples_per_pair', 10))  # always consumed, so that a GDA call does not forward it twice
    R = samples if solver == 'sgda' else 0
    implementation = run_gda
    binary, binary_hash = None, None
    if backend == 'rust':
        from lplearn.rust import _executable, run_gda as rust_run_gda
        implementation = rust_run_gda
        binary = Path(_executable())
        binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
        hp.update(timeout=rust_timeout, executable=binary)
    res = implementation(model, hp.pop('Gamma_y'), hp.pop('Gamma_lambda'), hp.pop('Gamma_mu'), hp.pop('eta_y'),
                         hp.pop('eta_dual'), hp.pop('n_iter'), samples_per_pair=R, seed=seed, **hp)
    if binary is not None:
        if hashlib.sha256(binary.read_bytes()).hexdigest() != binary_hash:
            raise RuntimeError('Rust backend binary changed during the solve')
        res['backend_binary_sha256'] = binary_hash
    return _lift(res, U) if partition is not None else res


__all__ = [
    'METHOD_LABELS', 'METHODS_K5', 'METHODS_K11', 'INSTANCE_SIZES', 'INSTANCE_K5T10', 'INSTANCE_K11T30',
    'LEARNER_DEFAULTS', 'LEGACY_FILES', 'LEGACY_EXTRA', 'NO_PARTITION', 'instance_params', 'partition_for',
    'method_spec', 'exact_solution', 'run_learner', 'singleton_partition',
]
