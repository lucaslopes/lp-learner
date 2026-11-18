'''
Stage `bound`: numerical validation of the aggregation bound (Lemma 1, Lemma 2, Theorem 1, Proposition 1).

Jobs:
- `partition`: one partition of one instance (K5T10 CJS/SJS over (E)-(K) and the singleton; optionally
  K11T30 CJS with PARTITION_K11; optionally the exactly lumpable sanity instance). Each job covers every
  regime and both lifting conventions. Output: JSON records of the rows and of the Proposition 1 terms.
- `deltas`: the Proposition 1 terms of every partition for both cases at one (p, b_high) of the sweep.

Summaries (written to results/bound/summary/ by default): bound_rows.csv, bound_rank.json,
deltas_cjs_sjs.csv, deltas_sweep.csv, bound_table.tex, deltas_table.tex. Definitions: reports/bound/README.md.
'''

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lplearn.analysis import bound as study
from lplearn.core import ADMIT, HIGH, PARTITION_K11, PARTITIONS_K5, singleton_partition
from lplearn.experiments.io import atomic_write_text

NAME = 'bound'
VERSION = '2'  # 2: fixed points asserted at every eta of the grid
SMOKE = {'include_k11': False}

LABELS_K5 = tuple(PARTITIONS_K5)
LUMPABLE_PARTITION = study.LUMPABLE_PARTITION
DEFAULTS: dict[str, Any] = {
    'K': 5,
    'T': 10,
    'cases': ['CJS', 'SJS'],
    'partitions': ['E', 'F', 'G', 'H', 'I', 'J', 'K', 'singleton'],
    'p': 0.7,
    'b_low': 0.01,
    'b_high': 0.1,
    'alpha': 0.75,
    'beta': 0.25,
    'gamma': 100.0,
    'gamma_bar': 100.0,
    'lemma_factors': [2.0, 1.25],
    'gamma_y': 1.0,
    'gamma_grid': [1.0, 0.1, 0.01, 0.001, 0.0001, 2e-5],
    'conventions': ['paper', 'consistent'],
    'eta_grid_points': 13,
    'sweep_p': [0.3, 0.5, 0.7, 0.9],
    'sweep_b_high': [0.1, 0.3, 0.5],
    'include_lumpable': True,
    'include_k11': True,
    'k11_cases': ['CJS'],
    'k11_eta_grid_points': 5,
    'table_regime': 'lemma2',
    'table_convention': 'consistent',
    'experimental_regime': 'grid2e-05',
}
SWEEP_CASES = ('CJS', 'SJS')
ACTION_NAMES = ('(a,b)=(0,0)', '(a,b)=(0,1)', '(a,b)=(1,0)', '(a,b)=(1,1)')


def _lists(partition) -> list[list[int]]:
    return [[int(s) for s in block] for block in partition]


def _partition_for(label: str, S: int):
    if label == 'singleton':
        return singleton_partition(S)
    if label in PARTITIONS_K5:
        return PARTITIONS_K5[label]
    raise KeyError(f'unknown partition label {label!r}')


def make_jobs(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    '''
    The partition jobs of the K5T10 study (cases x partitions), the optional K11T30 and lumpable jobs, and one
    deltas job per (p, b_high) of the sweep.
    '''
    c = {**DEFAULTS, **dict(cfg)}
    K, T = int(c['K']), int(c['T'])
    shared = {
        'p': float(c['p']), 'b_low': float(c['b_low']), 'b_high': float(c['b_high']), 'alpha': float(c['alpha']),
        'beta': float(c['beta']), 'gamma': float(c['gamma']), 'gamma_bar': float(c['gamma_bar']),
    }
    regimes = {
        'lemma_factors': [float(x) for x in c['lemma_factors']],
        'gamma_y': float(c['gamma_y']),
        'gamma_grid': [float(x) for x in c['gamma_grid']],
        'conventions': [str(x) for x in c['conventions']],
    }
    group_k5 = [{'label': lab, 'partition': _lists(PARTITIONS_K5[lab])} for lab in LABELS_K5]
    jobs: list[dict[str, Any]] = []
    for case in c['cases']:
        for label in c['partitions']:
            jobs.append({
                'kind': 'partition', 'instance': 'queue', 'K': K, 'T': T, 'case': str(case), **shared,
                'label': str(label), 'partition': _lists(_partition_for(str(label), K + 1)), 'group': group_k5,
                **regimes, 'eta_grid_points': int(c['eta_grid_points']),
            })
    if c['include_k11']:
        group_k11 = [{'label': 'K11', 'partition': _lists(PARTITION_K11)}]
        for case in c['k11_cases']:
            jobs.append({
                'kind': 'partition', 'instance': 'queue', 'K': 11, 'T': 30, 'case': str(case),
                **{**shared}, 'label': 'K11', 'partition': _lists(PARTITION_K11), 'group': group_k11,
                **regimes, 'eta_grid_points': int(c['k11_eta_grid_points']),
            })
    if c['include_lumpable']:
        group_lum = [
            {'label': 'lumpable', 'partition': _lists(LUMPABLE_PARTITION)},
            {'label': 'lumpable_singleton', 'partition': _lists(singleton_partition(4))},
        ]
        for label, part in (('lumpable', LUMPABLE_PARTITION), ('lumpable_singleton', singleton_partition(4))):
            jobs.append({
                'kind': 'partition', 'instance': 'lumpable', 'label': label, 'partition': _lists(part),
                'group': group_lum, **regimes, 'eta_grid_points': int(c['eta_grid_points']),
            })
    sweep_partitions = [{'label': lab, 'partition': _lists(PARTITIONS_K5[lab])} for lab in LABELS_K5]
    sweep_partitions.append({'label': 'singleton', 'partition': _lists(singleton_partition(K + 1))})
    for p in c['sweep_p']:
        for b_high in c['sweep_b_high']:
            jobs.append({
                'kind': 'deltas', 'K': K, 'T': T, 'cases': list(SWEEP_CASES), **{**shared, 'p': float(p), 'b_high': float(b_high)},
                'partitions': sweep_partitions,
            })
    return jobs


def run_job(job: Mapping[str, Any]) -> dict[str, Any]:
    '''
    Compute one job. Raises (the job fails and is recorded) if a bound falls below the true distance.
    '''
    if job['kind'] == 'partition':
        out = study.run_partition_job(job)
        return {
            'rows_json': study.dumps_compact(out['rows']),
            'blocks_json': study.dumps_compact(out['blocks']),
            'delta_json': study.dumps_compact(out['delta_max']),
            'n_rows': len(out['rows']),
            'min_slack': study.check_rows(out['rows'])['min_slack'],
        }
    if job['kind'] == 'deltas':
        return {'deltas_json': study.dumps_compact(study.run_deltas_job(job))}
    raise ValueError(f'unknown job kind {job["kind"]!r}')


def _action(u: int) -> str:
    return f'a={int(ADMIT[u])},b={int(HIGH[u])}'


def _worst_text(worst) -> str:
    u, s, target = worst
    return f'{_action(u)};s={s};to=({target})'


def _deltas_rows(partition_records: list[tuple[Mapping, dict]]) -> list[dict]:
    out = []
    for job, data in partition_records:
        if job['instance'] != 'queue' or job['K'] != 5:
            continue
        blocks = data['blocks']
        delta = data['delta']
        top_p = max(blocks, key=lambda b: b['delta_P'])
        top_b = max(blocks, key=lambda b: b['delta_P_block'])
        partition_text = study.partition_text(job['partition'])
        out.append({
            'level': 'partition', 'case': job['case'], 'label': job['label'], 'partition': partition_text, 'block': '',
            'members': '', 'delta_c': delta['delta_c'], 'delta_P': delta['delta_P'],
            'delta_P_block': delta['delta_P_block'], 'delta_d': delta['delta_d'], 'delta_0': delta['delta_0'],
            'score_A': delta['score_A'], 'score_B': delta['score_B'],
            'worst_P': _worst_text(top_p['worst_P']), 'worst_P_block': _worst_text(top_b['worst_P_block']),
        })
        for b in blocks:
            out.append({
                'level': 'block', 'case': job['case'], 'label': job['label'], 'partition': partition_text,
                'block': b['block'], 'members': ','.join(str(s) for s in b['members']), 'delta_c': b['delta_c'],
                'delta_P': b['delta_P'], 'delta_P_block': b['delta_P_block'], 'delta_d': b['delta_d'],
                'delta_0': b['delta_0'], 'score_A': '', 'score_B': '',
                'worst_P': _worst_text(b['worst_P']), 'worst_P_block': _worst_text(b['worst_P_block']),
            })
    return out


def _check_defaults(partition_records, sweep_records, cfg) -> int:
    '''The sweep point at the default parameters must reproduce the partition jobs (same deltas).'''
    checked = 0
    for rec in sweep_records:
        if abs(rec['p'] - float(cfg['p'])) > 1e-12 or abs(rec['b_high'] - float(cfg['b_high'])) > 1e-12:
            continue
        for job, data in partition_records:
            if job['instance'] != 'queue' or job['K'] != 5 or job['label'] not in LABELS_K5 + ('singleton',):
                continue
            ref = rec['by_case'][job['case']][job['label']]
            for key in ('delta_c', 'delta_P', 'delta_P_block', 'delta_d', 'delta_0'):
                if abs(ref[key] - data['delta'][key]) > 1e-12:
                    raise RuntimeError(f'sweep and partition deltas disagree for {job["case"]} {job["label"]} {key}')
            checked += 1
    return checked


def summarize(records: list[tuple[dict, dict]], cfg: dict[str, Any], out_dir: Path) -> None:
    '''
    Collect the rows of the partition jobs and the sweep, check the bound inequality, write the summary files.
    '''
    c = {**DEFAULTS, **dict(cfg)}
    rows: list[dict] = []
    partition_records: list[tuple[dict, dict]] = []
    sweep_records: list[dict] = []
    for job, arrays in records:
        if job['kind'] == 'partition':
            rows.extend(study.loads_rows(arrays['rows_json']))
            partition_records.append((job, {
                'blocks': json.loads(arrays['blocks_json']),
                'delta': json.loads(arrays['delta_json']),
            }))
        elif job['kind'] == 'deltas':
            sweep_records.append(json.loads(arrays['deltas_json']))
    rows.sort(key=lambda r: (r['instance'], str(r['case']), r['K'], str(r['label']), r['regime'], r['convention']))
    check = study.check_rows(rows)
    deltas_check = _check_defaults(partition_records, sweep_records, c)
    report = study.rank_report(rows)
    sweep = study.sweep_rows(sweep_records)
    fractions = study.sweep_fractions(sweep)
    sweep_k5 = [r for r in sweep if r['label'] in LABELS_K5]
    report['sweep'] = {
        'definition': 'fraction of partitions (E)-(K) with delta_P(CJS) > delta_P(SJS) (lifted rows) and the block form',
        'fractions': fractions,
        'overall_frac_dP': (sum(r['CJS_gt_SJS'] for r in sweep_k5) / len(sweep_k5)) if sweep_k5 else None,
        'overall_frac_dP_block': (sum(r['CJS_gt_SJS_block'] for r in sweep_k5) / len(sweep_k5)) if sweep_k5 else None,
    }
    report['checks'] = {**check, 'default_sweep_points_matched': deltas_check, 'rows': len(rows)}
    report['regimes'] = sorted({r['regime'] for r in rows})
    report['config'] = {k: c[k] for k in ('cases', 'partitions', 'p', 'b_high', 'lemma_factors', 'gamma_y', 'gamma_grid',
                                           'conventions', 'eta_grid_points', 'sweep_p', 'sweep_b_high')}

    target = Path(out_dir)
    row_fields = list(study.ROW_FIELDS)
    atomic_write_text(target / 'bound_rows.csv', study.csv_text(rows, row_fields))
    atomic_write_text(target / 'bound_rank.json', study.dumps_pretty(report))
    delta_rows = _deltas_rows(partition_records)
    atomic_write_text(target / 'deltas_cjs_sjs.csv', study.csv_text(delta_rows, [
        'level', 'case', 'label', 'partition', 'block', 'members', 'delta_c', 'delta_P', 'delta_P_block', 'delta_d',
        'delta_0', 'score_A', 'score_B', 'worst_P', 'worst_P_block',
    ]))
    atomic_write_text(target / 'deltas_sweep.csv', study.csv_text(sweep, [
        'p', 'b_high', 'label', 'delta_c', 'delta_0', 'delta_d', 'dP_CJS', 'dP_SJS', 'dPb_CJS', 'dPb_SJS',
        'CJS_gt_SJS', 'CJS_gt_SJS_block', 'tie',
    ]))
    atomic_write_text(target / 'bound_table.tex', study.bound_tables_tex(
        rows, regime=str(c['table_regime']), convention=str(c['table_convention']),
        experimental=str(c['experimental_regime']),
    ))
    default_rows = [r for r in rows if r['instance'] == 'queue' and r['K'] == 5]
    atomic_write_text(target / 'deltas_table.tex', study.delta_tables_tex(default_rows, fractions))
