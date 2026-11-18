'''
What happens when the requests of the queues exceed the shared budgets.

The model has no such mechanism. Its LP enforces the budgets only in expectation, and a population
of selfish queues may request more than the budgets allow. The three mechanisms below are modelling
choices, not properties of the paper's model, and must be reported as such.

M0 'unenforced': budgets are ignored and realised actions equal the requests (the literal model).
M1 'rationing' (primary): budgets are hard capacities. If the requests for a resource exceed the integer
   capacity floor(budget * N), a uniformly random subset of requesters of that size is served. A denied
   admission becomes a = 0 (the job is not admitted, cost gamma); a denied fast mode becomes b = 0 (cost of
   the low rate).
M2 'congestion' (variant): admission is rationed as in M1. The fast-mode capacity is shared instead: if the
   D_b fast requesters exceed floor(beta N), each keeps b = 1 (and pays the high-rate cost) but is served with
   the effective per-slot probability q_eff = b_low + (b_high - b_low) floor(beta N) / D_b.

Each mechanism takes the requests of one stage and returns the realised (a, b) and an info dict. The
lotteries are decided by priority draws made for every arm at every stage, whatever the requests, so the
mechanism's random stream does not depend on the controller (common random numbers).

The module also has the N -> infinity (fluid) evaluation of a requesting policy under each mechanism,
with the same grant rules: the grant probabilities are the mean-field ratios budget / request.
'''


from __future__ import annotations

import math

import numpy as np

from lplearn.core.instance import ADMIT, HIGH, WCMDP, QueueParams, transition_kernel


MECHANISM_NAMES = ('M0', 'M1', 'M2')
MECHANISM_LABELS = {'M0': 'unenforced', 'M1': 'rationing', 'M2': 'congestion'}


def capacities(N: int, alpha: float, beta: float) -> tuple[int, int]:
    '''
    Integer capacities floor(alpha N) and floor(beta N) (a small tolerance guards against float error).
    '''

    return int(math.floor(alpha * N + 1e-9)), int(math.floor(beta * N + 1e-9))


def ration(requests: np.ndarray, priority: np.ndarray, capacity: int) -> np.ndarray:
    '''
    Serve min(#requests, capacity) of the requesters: those with the smallest priority draws.\n
    With i.i.d. uniform priorities the served set is a uniformly random subset of the requesters.
    '''

    idx = np.flatnonzero(requests)
    served = np.zeros(np.shape(requests), dtype=np.int8)
    if idx.size <= capacity:
        served[idx] = 1
    elif capacity > 0:
        served[idx[np.argsort(priority[idx], kind='stable')[:capacity]]] = 1
    return served


def congested_service(demand: float, capacity: float, b_low: float, b_high: float) -> float:
    '''
    Effective per-slot service probability of each fast-mode requester under M2 (counts or fractions).
    '''

    if demand <= capacity or demand == 0:
        return float(b_high)
    return float(b_low + (b_high - b_low) * capacity / demand)


class Unenforced:
    '''
    M0: requests are realised as they are.
    '''

    name = 'M0'

    def __init__(self, b_low: float, b_high: float):
        self.b_low = float(b_low)
        self.b_high = float(b_high)

    def apply(self, t, states, a_req, b_req, budgets_int, rng):
        a = np.asarray(a_req, dtype=np.int8)
        b = np.asarray(b_req, dtype=np.int8)
        return a, b, {'denied_a': 0, 'denied_b': 0, 'degraded_b': 0, 'q_high': self.b_high}


class RandomRationing:
    '''
    M1: both budgets are hard capacities, rationed by independent uniform lotteries.
    '''

    name = 'M1'

    def __init__(self, b_low: float, b_high: float):
        self.b_low = float(b_low)
        self.b_high = float(b_high)

    def apply(self, t, states, a_req, b_req, budgets_int, rng):
        n = np.shape(a_req)[0]
        prio_a = rng.random(n)
        prio_b = rng.random(n)  # drawn even when unused, so the stream does not depend on the requests
        a = ration(a_req, prio_a, budgets_int[0])
        b = ration(b_req, prio_b, budgets_int[1])
        info = {
            'denied_a': int(np.sum(a_req) - np.sum(a)),
            'denied_b': int(np.sum(b_req) - np.sum(b)),
            'degraded_b': 0,
            'q_high': self.b_high,
        }
        return a, b, info


class CongestedHighRate:
    '''
    M2: admission is rationed; fast-mode requests are all honoured but share the capacity floor(beta N).
    '''

    name = 'M2'

    def __init__(self, b_low: float, b_high: float):
        self.b_low = float(b_low)
        self.b_high = float(b_high)

    def apply(self, t, states, a_req, b_req, budgets_int, rng):
        n = np.shape(a_req)[0]
        prio_a = rng.random(n)
        a = ration(a_req, prio_a, budgets_int[0])
        b = np.asarray(b_req, dtype=np.int8)
        demand = int(np.sum(b))
        capacity = int(budgets_int[1])
        info = {
            'denied_a': int(np.sum(a_req) - np.sum(a)),
            'denied_b': 0,
            'degraded_b': demand if demand > capacity else 0,
            'q_high': congested_service(demand, capacity, self.b_low, self.b_high),
        }
        return a, b, info


def make_mechanism(name: str, b_low: float, b_high: float):
    '''
    Mechanism object for 'M0', 'M1' or 'M2'.
    '''

    if name == 'M0':
        return Unenforced(b_low, b_high)
    if name == 'M1':
        return RandomRationing(b_low, b_high)
    if name == 'M2':
        return CongestedHighRate(b_low, b_high)
    raise ValueError(f'unknown mechanism {name!r}; expected one of {MECHANISM_NAMES}')


# ---------------------------------------------------------------------------
# Fluid (N -> infinity) evaluation of a requesting policy
# ---------------------------------------------------------------------------


def realized_policy(pi_t: np.ndarray, grant_a: float, grant_b: float, mechanism: str) -> np.ndarray:
    '''
    Distribution of the realised action (S, U) for requests pi_t (S, U) at one stage. A requested
    admission is granted with probability grant_a; a requested fast mode with grant_b under M1 only.
    '''

    S, U = pi_t.shape
    out = np.zeros((S, U))
    for u in range(U):
        a_req, b_req = int(ADMIT[u]), int(HIGH[u])
        a_opts = ((1, grant_a), (0, 1.0 - grant_a)) if a_req else ((0, 1.0),)
        if b_req and mechanism == 'M1':
            b_opts = ((1, grant_b), (0, 1.0 - grant_b))
        else:
            b_opts = ((b_req, 1.0),)
        for a_real, wa in a_opts:
            for b_real, wb in b_opts:
                w = wa * wb
                if w > 0.0:
                    out[:, 2 * a_real + b_real] += pi_t[:, u] * w
    return out


def fluid_mechanism(params: QueueParams, model: WCMDP, pi_req: np.ndarray, mechanism: str = 'M1') -> dict:
    '''
    Mean-field evaluation of requests pi_req[t, s, u] under a mechanism.\n
    Grants are the fluid ratios budget / request when the request exceeds the budget (M1: admission and fast
    mode; M2: admission only, with the congested fast rate). The realised policy and the state law are propagated
    exactly through the kernels. The cost is split into storage, processing, rejection and blocked terms, which
    add up to the model cost. `usage` is the realised usage per arm, `requested` the requested usage.
    '''

    if mechanism not in MECHANISM_NAMES:
        raise ValueError(f'unknown mechanism {mechanism!r}')
    T, S, U = pi_req.shape
    alpha, beta = (float(x) for x in model.budgets)
    K = params.K
    storage = np.arange(S) / K
    proc_u = 2.0 * (1.0 + np.where(HIGH == 1, params.b_high, params.b_low))
    rej_u = params.gamma * (ADMIT == 0).astype(float)
    blk_su = params.gamma_bar * ((np.arange(S)[:, None] == K) & (ADMIT[None, :] == 1))

    m = model.m0.astype(float).copy()
    usage = np.zeros((2, T))
    requested = np.zeros((2, T))
    mass = np.zeros((T, S))
    grant_a = np.ones(T)
    grant_b = np.ones(T)
    q_high = np.full(T, params.b_high)
    comp = {'storage': 0.0, 'processing': 0.0, 'rejection': 0.0, 'blocked': 0.0}
    for t in range(T):
        mass[t] = m
        req_a = float(m @ (pi_req[t] @ ADMIT))
        req_b = float(m @ (pi_req[t] @ HIGH))
        requested[:, t] = (req_a, req_b)
        if mechanism in ('M1', 'M2') and req_a > alpha:
            grant_a[t] = alpha / req_a
        if mechanism == 'M1' and req_b > beta:
            grant_b[t] = beta / req_b
        if mechanism == 'M2' and req_b > beta:
            q_high[t] = params.b_low + (params.b_high - params.b_low) * beta / req_b
        pir = realized_policy(pi_req[t], grant_a[t], grant_b[t], mechanism)
        usage[:, t] = (m @ (pir @ ADMIT), m @ (pir @ HIGH))
        comp['storage'] += float(m @ storage)
        comp['processing'] += float(m @ (pir @ proc_u))
        comp['rejection'] += float(m @ (pir @ rej_u))
        comp['blocked'] += float(np.sum(m[:, None] * pir * blk_su))
        if t + 1 < T:
            P_t = model.P
            if mechanism == 'M2' and q_high[t] < params.b_high:
                P_t = model.P.copy()
                congested = transition_kernel(params.case, K, params.p, q_high[t])
                for u in range(U):
                    if HIGH[u] == 1:
                        P_t[u] = congested[ADMIT[u]]
            m = np.einsum('su,usr->r', pir * m[:, None], P_t)
    return {
        'cost': float(sum(comp.values())),
        'components': comp,
        'usage': usage,
        'requested': requested,
        'mass': mass,
        'grant_a': grant_a,
        'grant_b': grant_b,
        'q_high': q_high,
    }
