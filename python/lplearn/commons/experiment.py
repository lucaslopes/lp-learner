'''
Finite-N simulation of the no-coordinator comparison, with common random numbers (CRN).

Populations (controllers) run on the same per-replication random streams, so their differences are paired:

- ``coord``    LP projection of the unregularized LP (method A): ``finite_n.LPProjectionController``, which rounds
               and repairs the conditional policy to the sample-path budgets (feasible by construction).
- ``indep``    every queue samples its action independently from the LP conditional policy
               (``finite_n.IndependentRandomizedController``).
- ``selfish``  every queue follows the private DP policy (tie-break as a parameter).
- ``price``    every queue follows the DP on c + lam * d with the LP shadow prices.
- ``devnf``, ``devlp``, ``devna``  one deviating queue (index 0) in an otherwise selfish population: it never
               requests the fast mode, follows the LP conditional policy, or never requests admission.

For replication r the streams are keyed by [seed, r, k]: 0 transition uniforms (T, N), 1 controller, 2 mechanism
(priorities), 3 initial states, exactly as in lplearn.finite_n.simulate; with the same controller and an unenforced
mechanism this loop reproduces that simulator's per-replication costs to floating-point precision (checked in tests).
The same keys serve every population, so differences are paired by replication.

Per replication the module records the per-queue cost (average over arms of the sum over stages of c),
its shadow-priced accounting (cost + sum_t lam_l(t) (usage_l(t) - budget_l)_+), per-stage requests, usage, overload
indicators, excesses, denials and degradations, and the queue statistics. Requests and usage are fractions of N.
'''


from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from lplearn.core.instance import ADMIT, HIGH, WCMDP, QueueParams, queue_wcmdp, transition_kernel
from lplearn.core.lp import build_lp, solve_lp
from lplearn.core.policy import conditional_policy
from lplearn.workflow import occupation_feasibility

from lplearn.finite_n import IndependentRandomizedController, LPProjectionController

from .mechanisms import capacities, fluid_mechanism, make_mechanism
from .selfish import best_response_value, deterministic_policy, policy_tables, priced_solution, selfish_solution


POPULATIONS = ('coord', 'indep', 'selfish', 'price', 'devnf', 'devlp', 'devna')
DETAILED = ('coord', 'indep', 'selfish', 'price')
TOTALS = ('cost', 'shadow', 'nonadmit', 'blocked', 'depart', 'qlen', 'full', 'repairs', 'arm0', 'rest')
STAGED = ('cost_t', 'req_a', 'req_b', 'use_a', 'use_b', 'over_a', 'over_b', 'exc_a', 'exc_b', 'den_a', 'den_b', 'deg_b')
COST_TERMS = ('storage', 'processing', 'rejection', 'blocked')


def _sample_actions(pi_states: np.ndarray, x: np.ndarray) -> np.ndarray:
    '''
    Inverse-CDF sampling of one action per row of `pi_states` (N, U) with uniforms `x` (N,).
    '''

    cum = np.cumsum(pi_states, axis=1)
    return np.minimum((x[:, None] >= cum).sum(axis=1), pi_states.shape[1] - 1)


class TableController:
    '''
    Every queue follows the same deterministic table of requests, indexed by stage and queue length.
    '''

    def __init__(self, a_tab: np.ndarray, b_tab: np.ndarray):
        self.a_tab = np.asarray(a_tab, dtype=np.int8)
        self.b_tab = np.asarray(b_tab, dtype=np.int8)

    def reset(self, N: int, rng: np.random.Generator) -> None:
        pass

    def act(self, t: int, states: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        return self.a_tab[t][states], self.b_tab[t][states]


class DeviatorController:
    '''
    Queue 0 deviates from the selfish table; every other queue stays selfish.
    '''

    MODES = ('never_fast', 'never_admit', 'lp_conditional')

    def __init__(self, a_tab: np.ndarray, b_tab: np.ndarray, mode: str, pi: np.ndarray | None = None):
        if mode not in self.MODES:
            raise ValueError(f'mode must be one of {self.MODES}')
        if mode == 'lp_conditional' and pi is None:
            raise ValueError('lp_conditional needs the LP conditional policy')
        self.base = TableController(a_tab, b_tab)
        self.mode = mode
        self.pi = None if pi is None else np.asarray(pi, dtype=float)

    def reset(self, N: int, rng: np.random.Generator) -> None:
        pass

    def act(self, t: int, states: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.base.act(t, states, rng)
        a = a.copy()
        b = b.copy()
        if self.mode == 'never_fast':
            b[0] = 0
        elif self.mode == 'never_admit':
            a[0] = 0
        else:
            u = _sample_actions(self.pi[t][states[:1]], rng.random(1))[0]
            a[0] = ADMIT[u]
            b[0] = HIGH[u]
        return a, b


@dataclass
class Game:
    '''
    Model data shared by the replications of one job.
    '''

    params: QueueParams
    model: WCMDP
    lam: np.ndarray  # (2, T) LP shadow prices of the admission and fast-mode budgets
    _cdf: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        p = self.params
        self._low = np.cumsum(transition_kernel(p.case, p.K, p.p, p.b_low), axis=-1)

    def cdf(self, q_high: float) -> tuple[np.ndarray, np.ndarray]:
        '''
        Cumulative kernels (2, S, S) for low-rate and fast-mode arms at the given fast-mode service probability.
        '''

        key = float(q_high)
        if key not in self._cdf:
            p = self.params
            self._cdf[key] = np.cumsum(transition_kernel(p.case, p.K, p.p, key), axis=-1)
        return self._low, self._cdf[key]


def _repairs(controller) -> int:
    counters = getattr(controller, 'counters', None)
    return int(counters.get('repairs', 0)) if counters else 0


def _run_replication(game: Game, controller, mechanism, r: int, seed: int, N: int) -> dict[str, float | np.ndarray]:
    '''
    One replication of one population. Returns the per-replication totals and, when requested by the caller, the stage arrays.
    '''

    m = game.model
    p = game.params
    T, S, K = m.T, m.S, p.K
    alpha, beta = p.alpha, p.beta
    caps = capacities(N, alpha, beta)
    uniforms = np.random.default_rng([seed, r, 0]).random((T, N))
    cdf_init = np.cumsum(m.m0)
    cdf_init[-1] = 1.0
    x_init = np.random.default_rng([seed, r, 3]).random(N)
    states = np.minimum(np.searchsorted(cdf_init, x_init, side='right'), S - 1).astype(np.int64)
    rng_c = np.random.default_rng([seed, r, 1])
    controller.reset(N, rng_c)
    rng_m = np.random.default_rng([seed, r, 2])

    arm_cost = np.zeros(N)
    staged = {k: np.zeros(T) for k in STAGED}
    tot = {'nonadmit': 0.0, 'blocked': 0.0, 'depart': 0.0, 'qlen': 0.0, 'full': 0.0, 'repairs': 0.0}
    for t in range(T):
        repairs_before = _repairs(controller)
        a_req, b_req = controller.act(t, states, rng_c)
        tot['repairs'] += (_repairs(controller) - repairs_before) / N
        a_req = np.asarray(a_req, dtype=np.int8)
        b_req = np.asarray(b_req, dtype=np.int8)
        a, b, info = mechanism.apply(t, states, a_req, b_req, caps, rng_m)
        u = 2 * a.astype(np.int64) + b.astype(np.int64)
        step = m.c[states, u]
        arm_cost += step
        full = states == K
        D_a, D_b = int(a_req.sum()), int(b_req.sum())
        staged['cost_t'][t] = step.mean()
        staged['req_a'][t] = D_a / N
        staged['req_b'][t] = D_b / N
        staged['use_a'][t] = int(a.sum()) / N
        staged['use_b'][t] = int(b.sum()) / N
        staged['over_a'][t] = D_a > caps[0]
        staged['over_b'][t] = D_b > caps[1]
        staged['exc_a'][t] = max(D_a / N - alpha, 0.0)
        staged['exc_b'][t] = max(D_b / N - beta, 0.0)
        staged['den_a'][t] = info['denied_a'] / N
        staged['den_b'][t] = info['denied_b'] / N
        staged['deg_b'][t] = info['degraded_b'] / N
        tot['qlen'] += states.mean() / T
        tot['full'] += full.mean() / T
        tot['nonadmit'] += float((a == 0).mean())
        tot['blocked'] += float((full & (a == 1)).mean())
        if t < T - 1:
            low, high = game.cdf(info['q_high'])
            rows = np.where(b[:, None] == 1, high[a, states], low[a, states])
            nxt = np.minimum((uniforms[t][:, None] >= rows).sum(axis=1), K).astype(np.int64)
            admitted = (a == 1) & ~full
            tot['depart'] += float(np.mean(states + admitted - nxt))
            states = nxt

    penalty = float(np.sum(game.lam[0] * np.maximum(staged['use_a'] - alpha, 0.0))
                    + np.sum(game.lam[1] * np.maximum(staged['use_b'] - beta, 0.0)))
    cost = float(arm_cost.mean())
    rec = {
        'cost': cost,
        'shadow': cost + penalty,
        'arm0': float(arm_cost[0]),
        'rest': float(arm_cost[1:].mean()) if N > 1 else math.nan,
        **tot,
    }
    rec.update(staged)
    return rec


def _storage_dtype(key: str):
    '''
    Per-stage flags are stored as uint8 and other per-stage arrays as float32, to keep the result archives small.
    '''

    if key.startswith('over_'):
        return np.uint8
    return np.float32 if key in STAGED else np.float64


def _simulate(game: Game, controller, mechanism, N: int, reps: Iterable[int], seed: int, detailed: bool) -> dict[str, np.ndarray]:
    keys = TOTALS + (STAGED if detailed else ())
    buf: dict[str, list] = {k: [] for k in keys}
    for r in reps:
        rec = _run_replication(game, controller, mechanism, r, seed, N)
        for k in keys:
            buf[k].append(rec[k])
    return {k: np.asarray(v, dtype=_storage_dtype(k)) for k, v in buf.items()}


LP_TOLERANCES = {'tol_gap_abs': 1e-10, 'tol_gap_rel': 1e-10, 'tol_feas': 1e-10, 'max_iter': 500}


def solve_method_a(model: WCMDP) -> dict:
    '''
    The unregularized LP (method A) with tight interior-point tolerances. Its shadow prices are then accurate to about
    1e-8, so the exact ties that the LP induces in the price-taking DP are separated from genuine differences.
    '''

    result = solve_lp(build_lp(model), Gamma_y=0.0, **LP_TOLERANCES)
    diagnostics = occupation_feasibility(result['y'], model)
    if not diagnostics['feasible']:
        raise RuntimeError(f'LP occupation is infeasible before policy conversion: {diagnostics}')
    result['occupation_feasibility'] = diagnostics
    return result


def job_params(job: dict) -> QueueParams:
    '''
    The model parameters of a job.
    '''

    return QueueParams(
        K=int(job['K']),
        T=int(job['T']),
        case=str(job['case']),
        p=float(job['p']),
        b_low=float(job['b_low']),
        b_high=float(job['b_high']),
        alpha=float(job['alpha']),
        beta=float(job['beta']),
        gamma=float(job['gamma']),
        gamma_bar=float(job['gamma_bar']),
    )


def run_job(job: dict) -> dict[str, np.ndarray]:
    '''
    All populations for one chunk of replications, plus the fluid (N -> infinity) values of the same settings.
    '''

    params = job_params(job)
    model = queue_wcmdp(params)
    lp = solve_method_a(model)
    pi_lp = conditional_policy(lp['y'])
    lam = np.asarray(lp['lam_res'], dtype=float)
    game = Game(params, model, lam)
    N = int(job['N'])
    mechanism = make_mechanism(job['mechanism'], params.b_low, params.b_high)
    caps = capacities(N, params.alpha, params.beta)
    tie = str(job['tie_break'])

    sol_s = selfish_solution(model, tie)
    sol_p = priced_solution(model, lam, tie)
    a_s, b_s = policy_tables(sol_s)
    a_p, b_p = policy_tables(sol_p)
    controllers = {
        'coord': LPProjectionController(pi_lp, model.budgets, rounding='randomized'),
        'indep': IndependentRandomizedController(pi_lp),
        'selfish': TableController(a_s, b_s),
        'price': TableController(a_p, b_p),
        'devnf': DeviatorController(a_s, b_s, 'never_fast'),
        'devlp': DeviatorController(a_s, b_s, 'lp_conditional', pi=pi_lp),
        'devna': DeviatorController(a_s, b_s, 'never_admit'),
    }
    reps = range(int(job['rep_start']), int(job['rep_start']) + int(job['n_reps']))
    out: dict[str, np.ndarray] = {}
    for name in POPULATIONS:
        rec = _simulate(game, controllers[name], mechanism, N, reps, int(job['seed']), detailed=name in DETAILED)
        for key, value in rec.items():
            out[f'{name}__{key}'] = value

    mech = str(job['mechanism'])
    fl_s = fluid_mechanism(params, model, deterministic_policy(sol_s.u), mech)
    fl_p = fluid_mechanism(params, model, deterministic_policy(sol_p.u), mech)
    fl_lp = fluid_mechanism(params, model, pi_lp, mech)
    br_value, br_policy = best_response_value(params, model, fl_s['grant_a'], fl_s['grant_b'], mech, fl_s['q_high'])
    out.update({
        'lp_value': float(lp['objective']),
        'lp_min_occupation': lp['occupation_feasibility']['min_occupation'],
        'lp_nonnegativity_violation': lp['occupation_feasibility']['nonnegativity_violation'],
        'lp_initial_residual': lp['occupation_feasibility']['initial_residual'],
        'lp_flow_residual': lp['occupation_feasibility']['flow_residual'],
        'lp_stage_mass_residual': lp['occupation_feasibility']['stage_mass_residual'],
        'lp_resource_violation': lp['occupation_feasibility']['resource_violation'],
        'lp_feasibility_tolerance': lp['occupation_feasibility']['tolerance'],
        'lp_lam_res': lam,
        'fluid_lp': fl_lp['cost'],
        'fluid_selfish': fl_s['cost'],
        'fluid_price': fl_p['cost'],
        'fluid_br': br_value,
        'fluid_selfish_components': np.array([fl_s['components'][k] for k in COST_TERMS]),
        'fluid_price_components': np.array([fl_p['components'][k] for k in COST_TERMS]),
        'fluid_lp_components': np.array([fl_lp['components'][k] for k in COST_TERMS]),
        'fluid_selfish_usage': fl_s['usage'],
        'fluid_selfish_grants': np.vstack([fl_s['grant_a'], fl_s['grant_b'], fl_s['q_high']]),
        'fluid_price_usage': fl_p['usage'],
        'fluid_lp_usage': fl_lp['usage'],
        'policy_selfish_u': sol_s.u.astype(np.int8),
        'policy_price_u': sol_p.u.astype(np.int8),
        'br_policy_selfish': br_policy.astype(np.int8),
    })
    return out
