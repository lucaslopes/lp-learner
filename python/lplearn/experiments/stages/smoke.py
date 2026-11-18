'''
Tiny demo stage, used by the test suite and to check the runner end to end.

Each job computes the eigenvalues of a random symmetric positive-definite matrix
``alpha * I + A A^T / n``. The config keys are ``sizes``, ``alphas``, ``seeds`` and
``sleep_s``. Fault injection, off by default, is set with ``fault_seed`` and
``fault_marker``: the job with that seed raises while the file named by
``fault_marker`` exists.
'''

import csv
import itertools
import os
import time
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from lplearn.experiments.io import atomic_write_text
from lplearn.experiments.runner import THREAD_ENV_VARS

NAME = 'smoke'
VERSION = '1'
SMOKE = {'sizes': [4], 'seeds': [0, 1]}


def make_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    '''
    Grid over sizes, alphas and seeds.
    '''

    fault_seed = cfg.get('fault_seed')
    fault_marker = cfg.get('fault_marker')
    jobs = []
    for size, alpha, seed in itertools.product(cfg.get('sizes', [4]), cfg.get('alphas', [1.0]), cfg.get('seeds', [0])):
        jobs.append({
            'size': int(size),
            'alpha': float(alpha),
            'seed': int(seed),
            'sleep_s': float(cfg.get('sleep_s', 0.0)),
            'fault_marker': fault_marker if fault_marker and seed == fault_seed else None,
        })
    return jobs


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    '''
    Eigenvalues of the matrix for one job, plus the thread caps the worker process sees.
    '''

    marker = job.get('fault_marker')
    if marker and Path(marker).exists():
        raise RuntimeError(f'injected fault: {marker} exists')
    n = job['size']
    rng = np.random.default_rng(job['seed'])
    a = rng.standard_normal((n, n))
    matrix = job['alpha'] * np.eye(n) + a @ a.T / n
    eigvals = np.linalg.eigvalsh(matrix)
    time.sleep(job['sleep_s'])
    return {
        'eigvals': eigvals,
        'condition': float(eigvals[-1] / eigvals[0]),
        'thread_env': ';'.join(f'{var}={os.environ.get(var, "")}' for var in THREAD_ENV_VARS),
    }


def summarize(records: list[tuple[dict[str, Any], dict[str, Any]]], cfg: dict[str, Any], out_dir: Path) -> None:
    '''
    Write smoke_summary.csv with one row per completed job.
    '''

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['size', 'alpha', 'seed', 'lambda_min', 'lambda_max', 'condition', 'thread_env'])
    for job, result in sorted(records, key=lambda record: (record[0]['size'], record[0]['alpha'], record[0]['seed'])):
        eigvals = result['eigvals']
        writer.writerow([
            job['size'],
            job['alpha'],
            job['seed'],
            f'{eigvals[0]:.6g}',
            f'{eigvals[-1]:.6g}',
            f'{result["condition"]:.6g}',
            result['thread_env'],
        ])
    atomic_write_text(out_dir / 'smoke_summary.csv', buffer.getvalue())
