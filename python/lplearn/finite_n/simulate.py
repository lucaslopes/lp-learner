'''
Replicated simulation of the N-arm queue system under a controller and a mechanism, with common random numbers.

For replication r (global index rep_offset + i) the random streams are

- ``default_rng([seed, r, 0]).random((T, N))``: transition uniforms, one row per stage;
- ``default_rng([seed, r, 1])``: controller randomness (passed to ``reset`` and ``act``);
- ``default_rng([seed, r, 2])``: mechanism randomness;
- ``default_rng([seed, r, 3]).random(N)``: initial states, drawn from m0 by inverse CDF.

Next states are drawn by inverse CDF of P[u, s, :] at the transition uniform, where u is the REALIZED
action, so two controllers that realize the same actions see the same trajectories.

Stages t = 0, ..., T-1 are decision epochs. Costs and occupation are recorded for these stages only. The
transition out of stage T-1 is drawn as well, so that departures can be reported for every stage; it does
not enter any cost.

All per-stage quantities are per arm (divided by N), so they can be compared across N. The keys are
documented in ``simulate``.
'''


from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lplearn.core import WCMDP

from .mechanisms import integer_budgets


def _counter(controller, key: str) -> int:
    counters = getattr(controller, 'counters', None)
    return 0 if not counters else int(counters.get(key, 0))


def _checked_actions(a, b, N: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != (N,) or b.shape != (N,):
        raise ValueError(f'actions must have shape ({N},), got {a.shape} and {b.shape}')
    if not (np.all((a == 0) | (a == 1)) and np.all((b == 0) | (b == 1))):
        raise ValueError('actions must be binary')
    return a.astype(np.int8), b.astype(np.int8)


def _next_states(cdf_next: np.ndarray, u: np.ndarray, states: np.ndarray, x: np.ndarray, S: int) -> np.ndarray:
    cdf = cdf_next[u, states]  # (N, S), rows of the cumulative kernel
    return np.minimum((cdf < x[:, None]).sum(axis=1), S - 1)


_PER_STAGE = (
    'cost_t', 'req_adm_t', 'adm_t', 'req_high_t', 'high_t', 'viol_adm_t', 'viol_high_t', 'over_adm_t',
    'over_high_t', 'queue_t', 'full_t', 'nonadmit_t', 'blocked_t', 'depart_t', 'repairs_t', 'repairs_adm_t',
    'repairs_high_t', 'fallbacks_t', 'denied_adm_t', 'denied_high_t',
)
_COUNTERS = ('repairs', 'repairs_adm', 'repairs_high', 'fallbacks')


def simulate(
        model: WCMDP,
        make_controller: Callable[[], object],
        N: int,
        n_reps: int,
        seed: int,
        mechanism,
        rep_offset: int = 0,
    ) -> dict[str, np.ndarray]:
    '''
    Simulate `n_reps` replications of the N-arm system and return per-replication arrays.

    `make_controller` is a zero-argument factory; one controller is built per call and reset for every
    replication. `mechanism` maps requests to realized actions (see ``lplearn.finite_n.mechanisms``).
    Budgets are the integer budgets floor(budget N + 1e-9) of the model.

    Per replication (leading axis n_reps):
    - ``rep_index``: global replication index;
    - ``cost``: total cost per arm, the sum over stages of ``cost_t``;
    - ``viol_max``, ``viol_any``: largest realized excess over an integer budget (fraction of N) and whether any
      realized excess occurred;
    - ``over_max``, ``over_any``: the same for requested actions (request overload);

    Per stage (shape (n_reps, T)), all per arm:
    - ``cost_t``: stage cost; ``req_adm_t``, ``adm_t``, ``req_high_t``, ``high_t``: requested and realized usage;
    - ``viol_adm_t``, ``viol_high_t``: positive part of realized usage minus the integer budget (fraction of N);
    - ``over_adm_t``, ``over_high_t``: positive part of requested usage minus the integer budget;
    - ``queue_t``: mean queue length; ``full_t``: fraction of full queues (s = K);
    - ``nonadmit_t``: fraction with realized a = 0; ``blocked_t``: fraction with realized a = 1 at s = K;
    - ``depart_t``: mean of s + a_eff - s' with a_eff = a for s < K and 0 otherwise;
    - ``repairs_t``, ``repairs_adm_t``, ``repairs_high_t``: arms whose requested action the controller moved to
      meet the admission budget, the high-rate budget, or either (``counters``), per arm;
    - ``fallbacks_t``: LP fallbacks in the stage (a count, not a fraction);
    - ``denied_adm_t``, ``denied_high_t``: requests denied by the mechanism;
    - ``state_hist`` (n_reps, T, S): fraction of arms in each queue length before the actions of each stage.
    '''
    N, n_reps, rep_offset, seed = int(N), int(n_reps), int(rep_offset), int(seed)
    if N < 1 or n_reps < 1 or rep_offset < 0:
        raise ValueError('N and n_reps must be positive and rep_offset nonnegative')
    S, T, K = model.S, model.T, model.S - 1
    budgets_int = integer_budgets(model.budgets, N)
    B_adm, B_high = int(budgets_int[0]), int(budgets_int[1])
    cdf_next = np.cumsum(model.P, axis=2)
    cdf_next[:, :, -1] = 1.0
    cdf_init = np.cumsum(model.m0)
    cdf_init[-1] = 1.0
    controller = make_controller()

    out = {
        'rep_index': np.arange(rep_offset, rep_offset + n_reps, dtype=np.int64),
        'cost': np.zeros(n_reps),
        'viol_max': np.zeros(n_reps),
        'viol_any': np.zeros(n_reps, dtype=bool),
        'over_max': np.zeros(n_reps),
        'over_any': np.zeros(n_reps, dtype=bool),
        'state_hist': np.zeros((n_reps, T, S)),
    }
    for name in _PER_STAGE:
        out[name] = np.zeros((n_reps, T), dtype=np.int64 if name == 'fallbacks_t' else float)

    for i in range(n_reps):
        r = rep_offset + i
        x_trans = np.random.default_rng([seed, r, 0]).random((T, N))
        rng_ctrl = np.random.default_rng([seed, r, 1])
        rng_mech = np.random.default_rng([seed, r, 2])
        x_init = np.random.default_rng([seed, r, 3]).random(N)
        states = np.minimum(np.searchsorted(cdf_init, x_init, side='right'), S - 1)
        controller.reset(N, rng_ctrl)
        for t in range(T):
            out['state_hist'][i, t] = np.bincount(states, minlength=S) / N
            before = {key: _counter(controller, key) for key in _COUNTERS}
            a_req, b_req = _checked_actions(*controller.act(t, states, rng_ctrl), N)
            a, b, info = mechanism.apply(t, states, a_req, b_req, budgets_int, rng_mech)
            a, b = _checked_actions(a, b, N)
            u = 2 * a.astype(np.int64) + b
            nxt = _next_states(cdf_next, u, states, x_trans[t], S)
            departures = states + a.astype(np.int64) * (states < K) - nxt

            adm_req, adm, high_req, high = int(a_req.sum()), int(a.sum()), int(b_req.sum()), int(b.sum())
            out['cost_t'][i, t] = model.c[states, u].sum() / N
            out['req_adm_t'][i, t] = adm_req / N
            out['adm_t'][i, t] = adm / N
            out['req_high_t'][i, t] = high_req / N
            out['high_t'][i, t] = high / N
            out['viol_adm_t'][i, t] = max(adm - B_adm, 0) / N
            out['viol_high_t'][i, t] = max(high - B_high, 0) / N
            out['over_adm_t'][i, t] = max(adm_req - B_adm, 0) / N
            out['over_high_t'][i, t] = max(high_req - B_high, 0) / N
            out['queue_t'][i, t] = states.mean()
            out['full_t'][i, t] = np.mean(states == K)
            out['nonadmit_t'][i, t] = np.mean(a == 0)
            out['blocked_t'][i, t] = np.mean((states == K) & (a == 1))
            out['depart_t'][i, t] = departures.mean()
            now = {key: _counter(controller, key) for key in _COUNTERS}
            out['repairs_t'][i, t] = (now['repairs'] - before['repairs']) / N
            out['repairs_adm_t'][i, t] = (now['repairs_adm'] - before['repairs_adm']) / N
            out['repairs_high_t'][i, t] = (now['repairs_high'] - before['repairs_high']) / N
            out['fallbacks_t'][i, t] = now['fallbacks'] - before['fallbacks']
            out['denied_adm_t'][i, t] = int(info.get('denied_adm', 0)) / N
            out['denied_high_t'][i, t] = int(info.get('denied_high', 0)) / N
            states = nxt

        out['cost'][i] = out['cost_t'][i].sum()
        viol = np.maximum(out['viol_adm_t'][i], out['viol_high_t'][i]).max()
        over = np.maximum(out['over_adm_t'][i], out['over_high_t'][i]).max()
        out['viol_max'][i], out['viol_any'][i] = viol, viol > 0
        out['over_max'][i], out['over_any'][i] = over, over > 0
    return out
