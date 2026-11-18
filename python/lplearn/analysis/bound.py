'''
Numerical validation of the aggregation bound of Section 6 of the SIGMETRICS manuscript
(Lemma 1, Lemma 2, Theorem 1 and Proposition 1). Definitions and findings: reports/bound/README.md.\n
For a detailed weakly coupled MDP and a partition of its states, `study_partition` computes, in every
regime (a regularization Gamma = (Gamma_y, Gamma_lambda, Gamma_mu)) and for both lifting conventions:\n
- the detailed regularized saddle point w* and the lifted aggregate saddle point w_bar*, with their
  fixed-point residuals, and the true distance ||w* - w_bar*|| split into primal, resource,
  nonnegativity and flow components;\n
- the one-step mismatch M = ||G(w_bar*) - G_bar(w_bar*)|| at the step eta = 1/Gamma^+;\n
- the Lemma 1 bound M/(1 - c_G), with c_G = 1 - eta (Gamma^- - kappa), defined only when Gamma^- > kappa;\n
- the spectral bound. For every eta with c(eta) = ||I - eta (diag R + S)||_2 < 1, Theorem 1 holds with
  c_G replaced by c(eta), so M(eta)/(1 - c(eta)) is a valid bound. The mismatch is linear in the step
  (M(eta) = eta m, up to projection effects at zero multipliers) and 1 - c(eta) = eta Gamma^- + o(eta),
  so the bound decreases to m/Gamma^- as eta -> 0. The minimum over a step grid and this limit are both
  reported, and the tightest valid one is used;\n
- the within-block variations of Proposition 1 (delta_c, delta_P, delta_P_block, delta_d, delta_0), the
  distances of the aggregate LP solutions to the detailed ones, and combined model scores.\n
The inequality bound >= ||w* - w_bar*|| is checked for every row; a violation raises (it indicates a bug).
'''


from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from lplearn.core import (
    CONVENTIONS, PARTITION_K11, PARTITIONS_K5, G_step, QueueParams, Regularization, WCMDP, aggregate, as_partition,
    build_lp, deltas, eta_grid, kappa, lemma1_contraction, lifted_G, lifted_kernel, maps, mismatch, queue_wcmdp,
    regularization_vector, regularized_saddle_point, saddle_maps, singleton_partition, skew_operator, spectral_contraction,
)
from lplearn.experiments.methods import exact_solution


FIXED_POINT_TOL = 1e-8  # relative residual for the fixed points checked at every grid eta
TOL = 1e-9                                  # absolute tolerance of the bound inequality
SLOPE_STEPS = (1e-3, 1e-6)                  # steps (relative to 1/Gamma^+) at which the slope m = lim M/eta is estimated
LABELS_K5 = ('E', 'F', 'G', 'H', 'I', 'J', 'K')
LUMPABLE_PARTITION = ((0,), (1, 2), (3,))
DELTA_KEYS = ('delta_c', 'delta_P', 'delta_P_block', 'delta_d', 'delta_0')
RANK_PREDICTORS = ('B_lemma1', 'B_spec', 'M', 'M_slope', 'delta_c', 'delta_P', 'delta_P_block', 'delta_0', 'delta_d', 'score_A', 'score_B')
RANK_PRIMAL = ('dist_y', 'dist_lp', 'dist_lp_reg')
ORDER_MEASURES = ('dist_total', 'B_spec', 'B_lemma1', 'M', 'score_A', 'delta_c', 'delta_P', 'delta_P_block', 'delta_0', 'dist_lp')
ROW_FIELDS = (
    'instance', 'K', 'T', 'case', 'label', 'partition', 'n_blocks', 'convention', 'regime', 'family',
    'Gamma_y', 'Gamma_lambda', 'Gamma_mu', 'kappa_det', 'kappa_hat', 'kappa_max', 'eta', 'lemma1_holds', 'lemma1_reason',
    'c_G', 'B_lemma1', 'lemma2_holds', 'Gamma_min_hat', 'Gamma_max_hat', 'M', 'M_slope', 'slope_spread', 'c_spec_eta',
    'eta_grid_best', 'c_grid_best', 'B_grid', 'B_spec_limit', 'B_spec', 'B_best', 'bound_source', 'ratio_spec',
    'ratio_lemma1', 'dist_total', 'dist_y', 'dist_lam_res', 'dist_lam_nn', 'dist_mu0', 'dist_muf', 'dist_lp', 'dist_lp_reg',
    'res_det', 'res_agg', 'res_lift', 'converged_det', 'converged_agg', 'ineq_star', 'eq_star', 'ineq_bar', 'eq_bar',
    'delta_c', 'delta_P', 'delta_P_block', 'delta_d', 'delta_0', 'score_A', 'score_B', 'check_ok',
)
NAN = math.nan


# ---------------------------------------------------------------------------
# Instances, partitions and regimes
# ---------------------------------------------------------------------------


def tuple_partition(partition) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(s) for s in block) for block in partition)


def partition_text(partition) -> str:
    return ''.join('{' + ','.join(str(s) for s in block) + '}' for block in partition)


def lumpable_model() -> WCMDP:
    '''
    The exactly lumpable instance of tests/test_core.py: states 1 and 2 are exact copies, so the partition
    {0}, {1,2}, {3} has zero delta terms (P[u, s, r] = P_hat[u, J(s), J(r)] / |J(r)|).
    '''
    rng = np.random.default_rng(0)
    blk, size = np.array([0, 1, 1, 2]), np.array([1, 2, 1])
    P_hat = rng.random((4, 3, 3))
    P_hat /= P_hat.sum(axis=2, keepdims=True)
    P = np.array([[[P_hat[u, blk[s], blk[r]] / size[blk[r]] for r in range(4)] for s in range(4)] for u in range(4)])
    d = np.zeros((2, 4, 4))
    d[0, :, 2:] = 1
    d[1][:, [1, 3]] = 1
    return WCMDP(P=P, c=(rng.random((3, 4)) * 10)[blk], d=d, budgets=np.array([0.6, 0.3]),
                 m0=np.array([0.4, 0.2, 0.2, 0.2]), T=6)


def queue_params(job: Mapping) -> QueueParams:
    return QueueParams(
        K=int(job['K']), T=int(job['T']), case=str(job['case']), p=float(job['p']), b_low=float(job['b_low']),
        b_high=float(job['b_high']), alpha=float(job['alpha']), beta=float(job['beta']), gamma=float(job['gamma']),
        gamma_bar=float(job['gamma_bar']),
    )


@dataclass(frozen=True)
class Regime:
    '''
    Regularization regime. Family 'lemma': Gamma_y = Gamma_lambda = Gamma_mu = value * kappa_max.
    Family 'grid': (Gamma_y, Gamma_lambda, Gamma_mu) = (gamma_y, value, value).
    '''

    name: str
    family: str
    value: float
    gamma_y: float = 1.0

    def regularization(self, kappa_max: float) -> Regularization:
        if self.family == 'lemma':
            g = self.value * kappa_max
            return Regularization(g, g, g)
        return Regularization(self.gamma_y, self.value, self.value)


def regimes_from_spec(lemma_factors: Sequence[float], gamma_y: float, gamma_grid: Sequence[float]) -> list[Regime]:
    regimes = [Regime(f'lemma{float(f):g}', 'lemma', float(f)) for f in lemma_factors]
    regimes += [Regime(f'grid{float(g):g}', 'grid', float(g), float(gamma_y)) for g in gamma_grid]
    return regimes


# ---------------------------------------------------------------------------
# Elementary quantities
# ---------------------------------------------------------------------------


def _norm(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float).ravel()))


def split_distance(struct, w_star: np.ndarray, w_bar: np.ndarray) -> dict[str, float]:
    '''||w* - w_bar*|| and its components: primal y, resource multipliers, nonnegativity multipliers, flow multipliers.'''
    a, b = struct.split(w_star), struct.split(w_bar)
    return {
        'dist_total': _norm(w_star - w_bar),
        'dist_y': _norm(a['y'] - b['y']),
        'dist_lam_res': _norm(a['lam_res'] - b['lam_res']),
        'dist_lam_nn': _norm(a['lam_nn'] - b['lam_nn']),
        'dist_mu0': _norm(a['mu0'] - b['mu0']),
        'dist_muf': _norm(a['muf'] - b['muf']),
    }


def feasibility(struct, w: np.ndarray) -> tuple[float, float]:
    '''Largest violation of the inequality constraints (resources and y >= 0) and of the equalities.'''
    y = struct.split(w)['y'].ravel()
    ineq = float(np.maximum(struct.D @ y - struct.r, 0.0).max())
    eq = float(np.abs(struct.A @ y - struct.b).max())
    return ineq, eq


def spearman(x, y) -> tuple[float, float]:
    '''Spearman rank correlation and its two-sided p-value over the pairs finite in both; NaN if undefined.'''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return NAN, NAN
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def lp_references(params: QueueParams, partition) -> dict[str, float]:
    '''
    Distances of the lifted aggregate LP solutions to the detailed ones: unregularized (method A, `lp`) and
    primal-regularized with Gamma_y = 1 (method B, `lp_reg`).
    '''
    y_a = exact_solution(params, 'lp', None)['y']
    y_b = exact_solution(params, 'lp_reg', None)['y']
    y_lp = exact_solution(params, 'lp', partition)['y']
    y_lp_reg = exact_solution(params, 'lp_reg', partition)['y']
    return {'dist_lp': _norm(y_lp - y_a), 'dist_lp_reg': _norm(y_lp_reg - y_b)}


def delta_details(model: WCMDP, partition) -> dict:
    '''
    The Proposition 1 terms of `partition`: maxima over the blocks (as in core `deltas`), the per-block values,
    the worst (action, state, target block) of delta_P and delta_P_block in each block, and the model scores
    score_A = delta_c / max|c| + delta_P + delta_0 and score_B = score_A + delta_d / max|d|.
    '''
    blocks = tuple_partition(partition)
    summary = deltas(model, blocks)
    V = maps(blocks, model.S)[1]
    P_lift = lifted_kernel(model, blocks)
    P_hat = aggregate(model, blocks).P
    per_block = []
    for j, block in enumerate(blocks):
        idx = list(block)
        lift = np.abs(model.P[:, idx, :] - P_lift[:, idx, :]).sum(axis=2)            # (U, |block|)
        u, i = np.unravel_index(int(np.argmax(lift)), lift.shape)
        s = idx[i]
        row = np.abs(model.P[u, s] - P_lift[u, s])
        target = int(np.argmax([row[list(b)].sum() for b in blocks]))
        to_blocks = np.einsum('usr,kr->usk', model.P[:, idx, :], V)                 # (U, |block|, J)
        blk = np.abs(to_blocks - P_hat[:, j][:, None, :]).sum(axis=2)               # (U, |block|)
        u2, i2 = np.unravel_index(int(np.argmax(blk)), blk.shape)
        s2 = idx[i2]
        target2 = int(np.argmax(np.abs(to_blocks[u2, i2] - P_hat[u2, j])))
        per_block.append({
            'block': j,
            'members': idx,
            'delta_c': float(np.max(model.c[idx].max(axis=0) - model.c[idx].min(axis=0))),
            'delta_P': float(lift.max()),
            'delta_P_block': float(blk.max()),
            'delta_d': float(np.max(model.d[:, idx].max(axis=1) - model.d[:, idx].min(axis=1))),
            'delta_0': float(model.m0[idx].max() - model.m0[idx].min()),
            'worst_P': [int(u), int(s), target],
            'worst_P_block': [int(u2), int(s2), target2],
        })
    out = {k: float(summary[k]) for k in DELTA_KEYS}
    cmax = float(np.abs(model.c).max()) or 1.0
    dmax = float(np.abs(model.d).max()) or 1.0
    out['score_A'] = out['delta_c'] / cmax + out['delta_P'] + out['delta_0']
    out['score_B'] = out['score_A'] + out['delta_d'] / dmax
    out['blocks'] = per_block
    return out


# ---------------------------------------------------------------------------
# The study of one partition
# ---------------------------------------------------------------------------


def study_partition(
        model: WCMDP,
        partition,
        group,
        regimes: Sequence[Regime],
        *,
        conventions: Sequence[str] = CONVENTIONS,
        eta_points: int = 13,
        params: QueueParams | None = None,
        meta: Mapping | None = None,
    ) -> dict:
    '''
    All quantities of Section 6 for one partition in every regime and convention.\n
    `group` lists the partitions that define kappa_max (the largest Lipschitz constant over the detailed
    problem and every aggregate problem of the group); `partition` is always included. With `params` the
    LP-level distances are computed. Returns {'rows': [...], 'blocks': {...}}. Raises RuntimeError if a
    valid bound is below the true distance by more than TOL.
    '''
    partition = tuple_partition(partition)
    group = [tuple_partition(g) for g in group]
    if partition not in group:
        group.append(partition)
    meta = dict(meta or {})
    sd = build_lp(model)
    skew = skew_operator(sd)
    kap_det = kappa(sd)
    kap_hat = {g: kappa(build_lp(aggregate(model, g))) for g in group}
    kappa_max = max([kap_det, *kap_hat.values()])
    details = delta_details(model, partition)
    sa = build_lp(aggregate(model, partition))
    refs = lp_references(params, partition) if params is not None else {'dist_lp': NAN, 'dist_lp_reg': NAN}
    rows = []
    for regime in regimes:
        reg = regime.regularization(kappa_max)
        R = regularization_vector(sd, reg)
        eta_c = 1.0 / reg.upper
        c_G = lemma1_contraction(kap_det, reg, eta_c)
        sol_d = regularized_saddle_point(sd, R)
        w_star = sol_d['w']
        res_det = _norm(G_step(sd, w_star, R, eta_c) - w_star)
        etas = np.unique(np.append(eta_grid(kap_det, R, n=eta_points), eta_c))
        spec = {float(e): spectral_contraction(sd, R, float(e), skew=skew) for e in etas}
        for conv in conventions:
            sm = saddle_maps(sd, sa, partition, conv)
            R_hat = sm.aggregate_regularization(reg)
            sol_a = regularized_saddle_point(sa, R_hat)
            w_hat = sol_a['w']
            w_bar = sm.to_detailed(w_hat)
            res_agg = _norm(G_step(sa, w_hat, R_hat, eta_c) - w_hat)
            res_lift = _norm(lifted_G(sm, w_bar, R_hat, eta_c, R=R) - w_bar)
            M_c = mismatch(sm, w_bar, R, R_hat, eta_c)
            slopes = [mismatch(sm, w_bar, R, R_hat, t * eta_c) / (t * eta_c) for t in SLOPE_STEPS]
            m_slope = max(slopes)
            bound_limit = m_slope / reg.lower
            best = {'eta': NAN, 'c': NAN, 'bound': math.inf}
            for e in etas:
                c = spec[float(e)]
                if c >= 1.0:
                    continue
                # Theorem 1 at this step size needs w* and w_bar* to be fixed points of G and G_bar for the same eta.
                for name, residual, scale in (
                        ('G', _norm(G_step(sd, w_star, R, float(e)) - w_star), _norm(w_star)),
                        ('G_bar', _norm(lifted_G(sm, w_bar, R_hat, float(e), R=R) - w_bar), _norm(w_bar))):
                    if residual > FIXED_POINT_TOL * max(1.0, scale) * max(1.0, float(e) / eta_c):
                        raise RuntimeError(f'{name} fixed point fails at eta={float(e):.3g}: residual {residual:.3g}')
                b = mismatch(sm, w_bar, R, R_hat, float(e)) / (1.0 - c)
                if b < best['bound']:
                    best = {'eta': float(e), 'c': c, 'bound': b}
            bound_grid = best['bound']
            bound_spec = min(bound_limit, bound_grid)
            bound_l1 = M_c / (1.0 - c_G) if c_G is not None else NAN
            dist = split_distance(sd, w_star, w_bar)
            dtot = dist['dist_total']
            for name, value in (('B_lemma1', bound_l1), ('B_spec', bound_spec)):
                if math.isfinite(value) and value < dtot - TOL:
                    raise RuntimeError(
                        f'bound violated ({name}): bound {value:.6g} < distance {dtot:.6g} for '
                        f'K={meta.get("K")} case={meta.get("case")} label={meta.get("label")} regime={regime.name} conv={conv}'
                    )
            ineq_s, eq_s = feasibility(sd, w_star)
            ineq_b, eq_b = feasibility(sd, w_bar)
            R_hat_min, R_hat_max = float(R_hat.min()), float(R_hat.max())
            lemma2 = bool(R_hat_min > kap_hat[partition] and eta_c <= 1.0 / R_hat_max)
            if c_G is not None and bound_l1 <= bound_spec:
                source = 'lemma1'
            else:
                source = 'spectral_grid' if bound_grid < bound_limit else 'spectral_limit'
            row = {
                **meta,
                'partition': partition_text(partition),
                'n_blocks': len(partition),
                'convention': conv,
                'regime': regime.name,
                'family': regime.family,
                'Gamma_y': reg.Gamma_y,
                'Gamma_lambda': reg.Gamma_lambda,
                'Gamma_mu': reg.Gamma_mu,
                'kappa_det': kap_det,
                'kappa_hat': kap_hat[partition],
                'kappa_max': kappa_max,
                'eta': eta_c,
                'lemma1_holds': c_G is not None,
                'lemma1_reason': 'ok' if c_G is not None else 'Gamma^- <= kappa',
                'c_G': c_G if c_G is not None else NAN,
                'B_lemma1': bound_l1,
                'lemma2_holds': lemma2,
                'Gamma_min_hat': R_hat_min,
                'Gamma_max_hat': R_hat_max,
                'M': M_c,
                'M_slope': m_slope,
                'slope_spread': abs(slopes[0] - slopes[1]) / max(abs(slopes[0]), 1e-300),
                'c_spec_eta': spec[float(eta_c)],
                'eta_grid_best': best['eta'],
                'c_grid_best': best['c'],
                'B_grid': bound_grid,
                'B_spec_limit': bound_limit,
                'B_spec': bound_spec,
                'B_best': min(bound_l1, bound_spec) if math.isfinite(bound_l1) else bound_spec,
                'bound_source': source,
                'ratio_spec': bound_spec / dtot if dtot > 0 else NAN,
                'ratio_lemma1': bound_l1 / dtot if (dtot > 0 and math.isfinite(bound_l1)) else NAN,
                **dist,
                **refs,
                'res_det': res_det,
                'res_agg': res_agg,
                'res_lift': res_lift,
                'converged_det': bool(sol_d['converged']),
                'converged_agg': bool(sol_a['converged']),
                'ineq_star': ineq_s,
                'eq_star': eq_s,
                'ineq_bar': ineq_b,
                'eq_bar': eq_b,
                'delta_c': details['delta_c'],
                'delta_P': details['delta_P'],
                'delta_P_block': details['delta_P_block'],
                'delta_d': details['delta_d'],
                'delta_0': details['delta_0'],
                'score_A': details['score_A'],
                'score_B': details['score_B'],
                'check_ok': True,
            }
            rows.append(row)
    return {'rows': rows, 'blocks': details['blocks'], 'delta_max': {k: details[k] for k in (*DELTA_KEYS, 'score_A', 'score_B')}}


def run_partition_job(job: Mapping) -> dict:
    '''One `partition` job of the stage: a partition of the K5T10, K11T30 or lumpable instance.'''
    if job['instance'] == 'queue':
        params = queue_params(job)
        model = queue_wcmdp(params)
        meta = {'instance': 'queue', 'K': params.K, 'T': params.T, 'case': params.case, 'label': job['label']}
    elif job['instance'] == 'lumpable':
        params, model = None, lumpable_model()
        meta = {'instance': 'lumpable', 'K': 3, 'T': 6, 'case': 'lumpable', 'label': job['label']}
    else:
        raise ValueError(f'unknown instance {job["instance"]!r}')
    regimes = regimes_from_spec(job['lemma_factors'], job['gamma_y'], job['gamma_grid'])
    group = [item['partition'] for item in job['group']]
    return study_partition(
        model, job['partition'], group, regimes, conventions=tuple(job['conventions']),
        eta_points=int(job['eta_grid_points']), params=params, meta=meta,
    )


def run_deltas_job(job: Mapping) -> dict:
    '''One `deltas` job: Proposition 1 terms of every partition for both cases at one (p, b_high).'''
    by_case = {}
    for case in job['cases']:
        model = queue_wcmdp(queue_params({**job, 'case': case}))
        by_case[case] = {item['label']: delta_details(model, item['partition']) for item in job['partitions']}
    return {'p': float(job['p']), 'b_high': float(job['b_high']), 'by_case': by_case}


def check_rows(rows: Sequence[Mapping]) -> dict:
    '''Re-check the bound inequality on stored rows. Returns the smallest slack and the number of rows checked.'''
    slack = math.inf
    for row in rows:
        for key in ('B_lemma1', 'B_spec'):
            value = row.get(key)
            if value is None or not math.isfinite(value):
                continue
            slack = min(slack, value - row['dist_total'])
            if value < row['dist_total'] - TOL:
                raise RuntimeError(f'bound violated in stored row: {key} for {row["label"]} {row["case"]} {row["regime"]}')
    return {'n_rows': len(rows), 'min_slack': slack}


# ---------------------------------------------------------------------------
# Summaries: rankings, sweep, tables
# ---------------------------------------------------------------------------


def rank_report(rows: Sequence[Mapping]) -> dict:
    '''
    Spearman correlations across the partitions (E)-(K) of each (case, regime, convention) and the exact
    orderings. The singleton and the lumpable sanity instances are excluded.
    '''
    groups: dict[tuple, dict[str, Mapping]] = {}
    for row in rows:
        if row['instance'] != 'queue' or row['label'] not in LABELS_K5:
            continue
        groups.setdefault((row['K'], row['T'], row['case'], row['regime'], row['convention']), {})[row['label']] = row
    out = []
    for key, by_label in sorted(groups.items()):
        labels = [lab for lab in LABELS_K5 if lab in by_label]
        measures = ('dist_total', *RANK_PREDICTORS, *RANK_PRIMAL)
        values = {name: [float(by_label[lab][name]) for lab in labels] for name in measures}
        correlations = {}
        for name in (*RANK_PREDICTORS, *RANK_PRIMAL):
            rho, p = spearman(values[name], values['dist_total'])
            correlations[name] = {'rho': rho, 'p': p}
        orders = {}
        for name in ORDER_MEASURES:
            pairs = []
            for lab in labels:
                value = float(by_label[lab][name])
                pairs.append((value if math.isfinite(value) else math.inf, lab))
            orders[name] = [lab for _, lab in sorted(pairs)]
        out.append({
            'K': key[0], 'T': key[1], 'case': key[2], 'regime': key[3], 'convention': key[4], 'labels': labels,
            'n': len(labels), 'values': values, 'spearman_vs_dist_total': correlations, 'orders': orders,
        })
    return {'groups': out, 'note': 'Seven partitions per group; Spearman p-values are indicative only.'}


def sweep_rows(deltas_records: Sequence[Mapping]) -> list[dict]:
    '''One row per (p, b_high, partition) with the lifted and block-form delta_P of CJS and SJS.'''
    out = []
    for rec in sorted(deltas_records, key=lambda r: (r['p'], r['b_high'])):
        cjs, sjs = rec['by_case'].get('CJS', {}), rec['by_case'].get('SJS', {})
        for label in LABELS_K5 + ('singleton',):
            if label not in cjs or label not in sjs:
                continue
            a, b = cjs[label], sjs[label]
            out.append({
                'p': rec['p'], 'b_high': rec['b_high'], 'label': label,
                'delta_c': a['delta_c'], 'delta_0': a['delta_0'], 'delta_d': a['delta_d'],
                'dP_CJS': a['delta_P'], 'dP_SJS': b['delta_P'],
                'dPb_CJS': a['delta_P_block'], 'dPb_SJS': b['delta_P_block'],
                'CJS_gt_SJS': int(a['delta_P'] > b['delta_P'] + 1e-12),
                'CJS_gt_SJS_block': int(a['delta_P_block'] > b['delta_P_block'] + 1e-12),
                'tie': int(abs(a['delta_P'] - b['delta_P']) <= 1e-12),
            })
    return out


def sweep_fractions(rows: Sequence[Mapping]) -> list[dict]:
    '''Fraction of partitions (E)-(K) with delta_P(CJS) > delta_P(SJS), per (p, b_high).'''
    groups: dict[tuple, list[Mapping]] = {}
    for row in rows:
        if row['label'] in LABELS_K5:
            groups.setdefault((row['p'], row['b_high']), []).append(row)
    out = []
    for (p, b_high), items in sorted(groups.items()):
        n = len(items)
        out.append({
            'p': p, 'b_high': b_high, 'n': n,
            'frac_dP_CJS_gt_SJS': sum(r['CJS_gt_SJS'] for r in items) / n,
            'frac_dP_block_CJS_gt_SJS': sum(r['CJS_gt_SJS_block'] for r in items) / n,
            'n_ties_dP': sum(r['tie'] for r in items),
        })
    return out


def _fmt_plain(x: float, spec: str) -> str:
    return '--' if x is None or not math.isfinite(x) else format(x, spec)


def tex_number(x: float, digits: int = 3) -> str:
    '''Number for a table cell: plain for moderate magnitudes, mantissa and power of ten otherwise.'''
    if x is None or not math.isfinite(x):
        return '--'
    if x == 0:
        return '0'
    plain = format(x, f'.{digits}g')
    if 'e' not in plain and abs(x) >= 1e-2:
        return plain
    exponent = int(math.floor(math.log10(abs(x))))
    mantissa = round(x / 10 ** exponent, digits - 1)
    if abs(mantissa) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return f'${mantissa:.{digits - 1}f}\\times10^{{{exponent}}}$'


def _pick(rows: Sequence[Mapping], **match) -> Mapping | None:
    for row in rows:
        if all(row.get(k) == v for k, v in match.items()):
            return row
    return None


def bound_tables_tex(rows: Sequence[Mapping], regime: str = 'lemma2', convention: str = 'consistent',
                     experimental: str = 'grid2e-05') -> str:
    '''
    Two booktabs tables for the manuscript: (1) the Lemma 1 regime, both cases, with the four Proposition 1
    terms, M, the Lemma 1 and spectral bounds and the true distance; (2) the experimental regime with M,
    the spectral bound and the true distance.
    '''
    queue = [r for r in rows if r['instance'] == 'queue' and r['K'] == 5 and r['convention'] == convention]
    head = (
        '\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n'
        f'\\caption{{Aggregation bound for partitions (E)--(K) in the Lemma~\\ref{{lem:contraction}} regime '
        f'($\\Gamma_y=\\Gamma_\\lambda=\\Gamma_\\mu=2\\kappa_{{\\max}}$, {convention} lifting). '
        '$M=\\|G(\\overline w^*)-\\overline G(\\overline w^*)\\|$ at $\\eta=1/\\Gamma^+$; $B_{\\rm L1}=M/(1-c_G)$ '
        '(Lemma~\\ref{lem:contraction}); $B_{\\rm sp}=\\inf_\\eta M(\\eta)/(1-c(\\eta))$ with $c(\\eta)=\\|I-\\eta(\\mathrm{diag}\\,R+S)\\|_2$, '
        'the spectral form of Theorem~\\ref{thm:main-bound}; $\\|\\cdot\\|$ is the true distance $\\|w^*-\\overline w^*\\|$.}\n'
        '\\label{tab:bound-lemma1}\n'
        '\\begin{tabular}{lrrrrrrr}\n\\toprule\n'
        'Partition & $\\delta_c$ & $\\delta_P$ & $\\delta_0$ & $M$ & $B_{\\rm L1}$ & $B_{\\rm sp}$ & $\\|w^*-\\bar w^*\\|$ \\\\\n'
        '\\midrule\n'
    )
    body = []
    for case in ('CJS', 'SJS'):
        body.append(f'\\multicolumn{{8}}{{l}}{{\\textit{{{case}}}}} \\\\\n')
        for lab in LABELS_K5:
            r = _pick(queue, case=case, label=lab, regime=regime)
            if r is None:
                continue
            body.append(
                f'({lab}) & {_fmt_plain(r["delta_c"], ".1f")} & {_fmt_plain(r["delta_P"], ".2f")} & '
                f'{_fmt_plain(r["delta_0"], ".0f")} & {tex_number(r["M"])} & {tex_number(r["B_lemma1"])} & '
                f'{tex_number(r["B_spec"])} & {tex_number(r["dist_total"])} \\\\\n'
            )
        if case == 'CJS':
            body.append('\\midrule\n')
    table1 = head + ''.join(body) + '\\bottomrule\n\\end{tabular}\n\\end{table}\n'

    head2 = (
        '\n\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n'
        f'\\caption{{Experimental regime ($\\Gamma_y=1$, $\\Gamma_\\lambda=\\Gamma_\\mu=2\\cdot10^{{-5}}$, '
        f'{convention} lifting): Lemma~\\ref{{lem:contraction}} does not apply ($\\Gamma^-\\le\\kappa$), so only the spectral form '
        f'$B_{{\\rm sp}}$ is shown.}}\n'
        '\\label{tab:bound-experimental}\n'
        '\\begin{tabular}{lrrrrrr}\n\\toprule\n'
        ' & \\multicolumn{3}{c}{CJS} & \\multicolumn{3}{c}{SJS} \\\\\n'
        '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n'
        'Partition & $M$ & $B_{\\rm sp}$ & $\\|w^*-\\bar w^*\\|$ & $M$ & $B_{\\rm sp}$ & $\\|w^*-\\bar w^*\\|$ \\\\\n'
        '\\midrule\n'
    )
    body2 = []
    for lab in LABELS_K5:
        cells = []
        for case in ('CJS', 'SJS'):
            r = _pick(queue, case=case, label=lab, regime=experimental)
            if r is None:
                cells += ['--', '--', '--']
            else:
                cells += [tex_number(r['M']), tex_number(r['B_spec']), tex_number(r['dist_total'])]
        body2.append(f'({lab}) & ' + ' & '.join(cells) + ' \\\\\n')
    table2 = head2 + ''.join(body2) + '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    return table1 + table2


def delta_tables_tex(default_rows: Sequence[Mapping], fractions: Sequence[Mapping]) -> str:
    '''Table of the Proposition 1 terms at the default parameters, and the sweep fractions delta_P(CJS) > delta_P(SJS).'''
    # The Proposition 1 terms do not depend on the regime or the convention, so any row of a (case, label) serves.
    queue = [r for r in default_rows if r['instance'] == 'queue' and r['K'] == 5]
    head = (
        '\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n'
        '\\caption{Proposition~\\ref{prop:mismatch} terms of partitions (E)--(K) at the default parameters '
        '($p=0.7$, $\\overline b=0.1$). $\\delta_P$ uses the lifted rows and $\\delta_P^{\\rm blk}$ compares only the block-level '
        'transition probabilities. $\\delta_d=0$ for all partitions because consumption depends only on the action.}\n'
        '\\label{tab:deltas}\n'
        '\\begin{tabular}{lrrrrrr}\n\\toprule\n'
        ' & & & \\multicolumn{2}{c}{$\\delta_P$} & \\multicolumn{2}{c}{$\\delta_P^{\\rm blk}$} \\\\\n'
        '\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n'
        'Partition & $\\delta_c$ & $\\delta_0$ & CJS & SJS & CJS & SJS \\\\\n'
        '\\midrule\n'
    )
    body = []
    for lab in LABELS_K5:
        cjs = _pick(queue, case='CJS', label=lab)
        sjs = _pick(queue, case='SJS', label=lab)
        if cjs is None or sjs is None:
            continue
        body.append(
            f'({lab}) & {_fmt_plain(cjs["delta_c"], ".1f")} & {_fmt_plain(cjs["delta_0"], ".0f")} & '
            f'{_fmt_plain(cjs["delta_P"], ".3f")} & {_fmt_plain(sjs["delta_P"], ".3f")} & '
            f'{_fmt_plain(cjs["delta_P_block"], ".3f")} & {_fmt_plain(sjs["delta_P_block"], ".3f")} \\\\\n'
        )
    table1 = head + ''.join(body) + '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    if not fractions:
        return table1
    ps = sorted({f['p'] for f in fractions})
    bs = sorted({f['b_high'] for f in fractions})
    lookup = {(f['p'], f['b_high']): f for f in fractions}
    lines = []
    for p in ps:
        cells = []
        for b in bs:
            f = lookup.get((p, b))
            cells.append('--' if f is None else f'{f["frac_dP_CJS_gt_SJS"]:.2f} / {f["frac_dP_block_CJS_gt_SJS"]:.2f}')
        lines.append(f'{p:g} & ' + ' & '.join(cells) + ' \\\\\n')
    head2 = (
        '\n\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n'
        '\\caption{Fraction of partitions (E)--(K) with $\\delta_P^{\\rm CJS}>\\delta_P^{\\rm SJS}$ '
        '(lifted rows / block form) over the parameter sweep ($K=5$).}\n'
        '\\label{tab:deltas-sweep}\n'
        f'\\begin{{tabular}}{{l{"r" * len(bs)}}}\n\\toprule\n'
        '$p$ & ' + ' & '.join(f'$\\overline b={b:g}$' for b in bs) + ' \\\\\n'
        '\\midrule\n'
    )
    return table1 + head2 + ''.join(lines) + '\\bottomrule\n\\end{tabular}\n\\end{table}\n'


def csv_text(rows: Sequence[Mapping], fields: Sequence[str]) -> str:
    '''CSV with the given columns; NaN is written as "nan" and booleans as 0/1.'''
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(list(fields))
    for row in rows:
        cells = []
        for key in fields:
            value = row.get(key, '')
            if isinstance(value, bool):
                cells.append(int(value))
            elif isinstance(value, float):
                cells.append(f'{value:.10g}')
            elif isinstance(value, (list, tuple)):
                cells.append(json.dumps(value))
            else:
                cells.append('' if value is None else value)
        writer.writerow(cells)
    return buffer.getvalue()


def _plain(obj):
    '''JSON-ready copy: NumPy values to Python, NaN and infinities to None, tuples to lists.'''
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def dumps_compact(obj) -> str:
    return json.dumps(_plain(obj), sort_keys=True, separators=(',', ':'), allow_nan=False)


def dumps_pretty(obj) -> str:
    return json.dumps(_plain(obj), sort_keys=True, indent=2, allow_nan=False) + '\n'


def loads_rows(text: str) -> list[dict]:
    '''Parse stored rows; JSON null becomes NaN in numeric fields.'''
    out = []
    for row in json.loads(text):
        out.append({k: (NAN if v is None else v) for k, v in row.items()})
    return out
