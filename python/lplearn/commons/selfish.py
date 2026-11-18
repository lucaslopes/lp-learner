'''
Private objective of a queue and the policies that optimise it.

A queue that ignores the shared budgets minimises its own expected sum of the model's one-stage
cost over the horizon. That is the single-arm finite-horizon MDP with kernels P[u], costs c and
no resource constraint, solved by backward induction. The same recursion with stage-dependent
costs c + lam * d, using the LP shadow prices lam, gives the price-taking policy.

Exact ties are resolved by an explicit parameter. At s = K, admitting and rejecting cost the same
immediate amount when gamma = gamma_bar, and both lead to the same next-state law (the admission is
blocked), so they tie exactly. ``no_request`` (default) picks the minimiser that requests fewer
resources (ties broken toward no admission, then no fast mode); ``request`` picks the one that
requests more. The tie-break changes only which requests the mechanism sees, never the model cost.
'''


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lplearn.core.instance import ADMIT, HIGH, WCMDP, QueueParams, transition_kernel


TIE_BREAKS = ('no_request', 'request')
TIE_TOL = 1e-6  # absolute, in cost units: LP-induced ties are below 1e-6 when the LP is solved tightly (see experiment.solve_method_a)
_REQUESTS = ADMIT + HIGH


@dataclass(frozen=True, eq=False)
class PrivateSolution:
    '''
    Backward-induction output for one private objective. Actions are indexed u = 2a + b.
    '''

    V: np.ndarray            # (T+1, S) cost-to-go, V[T] = 0
    Q: np.ndarray            # (T, S, U) action values
    u: np.ndarray            # (T, S) greedy action index after the tie-break
    n_minimisers: np.ndarray  # (T, S) number of actions within tolerance of the minimum
    tie_break: str

    @property
    def a(self) -> np.ndarray:
        return ADMIT[self.u]

    @property
    def b(self) -> np.ndarray:
        return HIGH[self.u]

    def admit_advantage(self) -> np.ndarray:
        '''
        (T, S, 2) indexed by b: Q_t(s, a=0, b) - Q_t(s, a=1, b). Positive means the queue gains privately from admitting.
        '''
        Q = self.Q
        return np.stack([Q[:, :, 0] - Q[:, :, 2], Q[:, :, 1] - Q[:, :, 3]], axis=-1)

    def fast_advantage(self) -> np.ndarray:
        '''
        (T, S, 2) indexed by a: Q_t(s, a, b=0) - Q_t(s, a, b=1). Positive means the queue gains privately from the fast mode.
        '''
        Q = self.Q
        return np.stack([Q[:, :, 0] - Q[:, :, 1], Q[:, :, 2] - Q[:, :, 3]], axis=-1)


def _greedy(q_row: np.ndarray, tie_break: str, tol: float) -> tuple[int, int]:
    '''
    Greedy action for one state, and the number of actions that tie with the minimum.
    '''

    best = q_row.min()
    cands = np.flatnonzero(q_row <= best + tol)
    if tie_break == 'no_request':  # fewest requests, then no admission, then no fast mode
        order = np.lexsort((HIGH[cands], ADMIT[cands], _REQUESTS[cands]))
    else:  # most requests, then admission, then fast mode
        order = np.lexsort((-HIGH[cands], -ADMIT[cands], -_REQUESTS[cands]))
    return int(cands[order[0]]), int(cands.size)


def private_dp(P: np.ndarray, cost: np.ndarray, tie_break: str = 'no_request', tol: float = TIE_TOL) -> PrivateSolution:
    '''
    Minimise E[sum_t cost_t(S_t, U_t)] over deterministic Markov policies, with zero terminal value.\n
    `P` is (U, S, S) row-stochastic and `cost` is (T, S, U). The stage index t = 0..T-1 follows the LP convention.
    '''

    if tie_break not in TIE_BREAKS:
        raise ValueError(f'tie_break must be one of {TIE_BREAKS}, got {tie_break!r}')
    P = np.asarray(P, dtype=float)
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 3:
        raise ValueError('cost must have shape (T, S, U)')
    T, S, U = cost.shape
    if P.shape != (U, S, S):
        raise ValueError(f'P must have shape {(U, S, S)}, got {P.shape}')
    V = np.zeros((T + 1, S))
    Q = np.zeros((T, S, U))
    pick = np.zeros((T, S), dtype=np.int64)
    nmin = np.zeros((T, S), dtype=np.int64)
    for t in range(T - 1, -1, -1):
        Q[t] = cost[t] + np.einsum('usr,r->su', P, V[t + 1])
        for s in range(S):
            pick[t, s], nmin[t, s] = _greedy(Q[t, s], tie_break, tol)
        V[t] = Q[t].min(axis=1)
    return PrivateSolution(V=V, Q=Q, u=pick, n_minimisers=nmin, tie_break=tie_break)


def stage_costs(model: WCMDP) -> np.ndarray:
    '''
    The model's one-stage cost as a (T, S, U) array.
    '''

    return np.broadcast_to(model.c, (model.T, model.S, model.U)).astype(float).copy()


def selfish_solution(model: WCMDP, tie_break: str = 'no_request') -> PrivateSolution:
    '''
    Private optimum of a queue that ignores the budgets: minimise its own cost c.
    '''

    return private_dp(model.P, stage_costs(model), tie_break)


def priced_solution(model: WCMDP, lam_res: np.ndarray, tie_break: str = 'no_request') -> PrivateSolution:
    '''
    Price-taking optimum: minimise c + lam_res[0, t] a + lam_res[1, t] b, with the LP shadow prices lam_res (L, T).
    '''

    lam = np.asarray(lam_res, dtype=float)
    cost = stage_costs(model)
    cost = cost + lam[0][:, None, None] * ADMIT[None, None, :] + lam[1][:, None, None] * HIGH[None, None, :]
    return private_dp(model.P, cost, tie_break)


def policy_tables(sol: PrivateSolution) -> tuple[np.ndarray, np.ndarray]:
    '''
    Requested admission and fast-mode tables (T, S) as int8 arrays, for a deterministic policy.
    '''

    return sol.a.astype(np.int8), sol.b.astype(np.int8)


def deterministic_policy(u: np.ndarray, n_actions: int = 4) -> np.ndarray:
    '''
    One-hot Markov policy pi[t, s, u] from a (T, S) table of action indices.
    '''

    T, S = u.shape
    pi = np.zeros((T, S, n_actions))
    pi[np.arange(T)[:, None], np.arange(S)[None, :], u] = 1.0
    return pi


def incentive_rows(sol: PrivateSolution) -> list[dict]:
    '''
    One row per (t, s) with the policy choice, its ties and the private advantages.
    '''

    adm = sol.admit_advantage()
    fast = sol.fast_advantage()
    rows = []
    T, S = sol.u.shape
    for t in range(T):
        for s in range(S):
            rows.append({
                't': t,
                's': s,
                'action_a': int(sol.a[t, s]),
                'action_b': int(sol.b[t, s]),
                'n_minimisers': int(sol.n_minimisers[t, s]),
                'V_next': float(sol.V[t, s]),
                'adv_admit_b0': float(adm[t, s, 0]),
                'adv_admit_b1': float(adm[t, s, 1]),
                'adv_fast_a0': float(fast[t, s, 0]),
                'adv_fast_a1': float(fast[t, s, 1]),
            })
    return rows


def best_response_value(
        params: QueueParams,
        model: WCMDP,
        grant_a: np.ndarray,
        grant_b: np.ndarray | None = None,
        mechanism: str = 'M1',
        q_high: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
    '''
    Best response of ONE queue (N -> infinity) to exogenous grant probabilities.\n
    A queue that requests admission (fast mode) is served with probability grant_a[t] (grant_b[t], M1 only):
    a denied request leaves the queue with a = 0 (b = 0), exactly as not requesting. Under M2 fast requests
    are always served, at the congested per-slot probability q_high[t]. Returns the expected cost from m0 and
    the optimal request policy (T, S, 2) as (request_a, request_b).
    '''

    T, S, U = model.T, model.S, model.U
    grant_a = np.asarray(grant_a, dtype=float)
    if mechanism == 'M1' and grant_b is not None:
        gb = np.asarray(grant_b, dtype=float)
    else:
        gb = np.ones(T)
    qh = np.full(T, params.b_high) if q_high is None else np.asarray(q_high, dtype=float)
    V = np.zeros(S)
    policy = np.zeros((T, S, 2), dtype=np.int64)
    for t in range(T - 1, -1, -1):
        P_t = model.P.copy()
        if mechanism == 'M2' and qh[t] < params.b_high:
            congested = transition_kernel(params.case, params.K, params.p, qh[t])
            for u in range(U):
                if HIGH[u] == 1:
                    P_t[u] = congested[ADMIT[u]]
        W = np.zeros((S, 4))
        for ra in (0, 1):
            for rb in (0, 1):
                a_opts = ((1, grant_a[t]), (0, 1.0 - grant_a[t])) if ra else ((0, 1.0),)
                b_opts = ((1, gb[t]), (0, 1.0 - gb[t])) if rb else ((0, 1.0),)
                value = np.zeros(S)
                for a_real, wa in a_opts:
                    for b_real, wb in b_opts:
                        if wa * wb == 0.0:
                            continue
                        u = 2 * a_real + b_real
                        value += wa * wb * (model.c[:, u] + P_t[u] @ V)
                W[:, 2 * ra + rb] = value
        pick = W.argmin(axis=1)
        policy[t, :, 0] = pick // 2
        policy[t, :, 1] = pick % 2
        V = W.min(axis=1)
    return float(model.m0 @ V), policy
