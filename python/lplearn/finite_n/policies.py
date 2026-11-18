'''
Controllers that turn LP occupation measures into REQUESTED actions for the N-arm queue system.

- ``LPProjectionController``: open loop. At stage t with queue counts N_s, the targets are
  z[s, u] = N_s pi[t, s, u]. They are rounded to integer counts n[s, u] (sum_u n[s, u] = N_s and
  E[n] = z) by systematic randomized rounding, then repaired greedily onto the integer budgets,
  then the counts are assigned to arms uniformly at random within each state.
- ``LPUpdateController``: closed loop. At stage t it re-solves the relaxed LP over the remaining
  horizon T - t, started from the current empirical distribution (detailed, or aggregate through the
  map V and lifted with U), and applies the same round-and-repair step to its first-stage policy.
  If a solve fails it uses the open-loop policy of the full-horizon LP for that stage and counts a fallback.
- ``IndependentRandomizedController``: every arm draws its own action from pi[t, s, :]. Nothing is
  rounded or repaired, so the budgets only hold in expectation.

All three follow the controller protocol documented in ``lplearn.finite_n``. The coordinated controllers
keep a ``counters`` dict with the number of arms moved by the repair ('repairs', split into
'repairs_adm' and 'repairs_high') and the number of LP fallbacks ('fallbacks').
'''


from __future__ import annotations

import hashlib
from dataclasses import replace

import cvxpy as cp
import numpy as np
from cvxpy.error import SolverError

from lplearn.core import ADMIT, HIGH, WCMDP, aggregate, as_partition, build_lp, conditional_policy, lift_y, maps

from .mechanisms import integer_budgets


_SOLVER = 'CLARABEL'
_OK_STATUSES = ('optimal', 'optimal_inaccurate')
_COUNTER_KEYS = ('repairs', 'repairs_adm', 'repairs_high', 'fallbacks')


def _zero_counters() -> dict[str, int]:
    return dict.fromkeys(_COUNTER_KEYS, 0)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    '''Rows over the last axis as probability vectors. A row without mass becomes the action (0, 0).'''
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    total = x.sum(axis=-1, keepdims=True)
    positive = total > 1e-12
    out = np.zeros_like(x)
    np.divide(x, total, out=out, where=positive)
    out[~positive[..., 0], 0] = 1.0
    return out


def normalize_policy(pi) -> np.ndarray:
    '''Validate a (T, S, U) Markov policy and make every conditional row a probability vector.'''
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 3:
        raise ValueError(f'pi must have shape (T, S, U), got {pi.shape}')
    return _normalize_rows(pi)


def systematic_round(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    '''
    Randomized rounding of nonnegative rows z[s, :] whose sums are integers N_s (up to round-off).

    Each row is rounded by systematic sampling: take the floors, then place the R_s = N_s - sum floor
    round-ups on the cumulative sums of the fractional parts, using one uniform shift per row. This gives
    exactly R_s round-ups per row, P(round up u) = frac(z[s, u]) and therefore E[n] = z. The result is an
    integer array with n.sum(axis=1) = round(z.sum(axis=1)).
    '''
    z = np.asarray(z, dtype=float)
    S, U = z.shape
    base = np.floor(z).astype(np.int64)
    frac = z - base
    round_ups = np.rint(z.sum(axis=1)).astype(np.int64) - base.sum(axis=1)
    counts = base.copy()
    shifts = rng.random(S)
    for s in range(S):
        k = int(round_ups[s])
        if k <= 0:
            continue
        cum = np.cumsum(frac[s])
        points = shifts[s] + np.arange(k)
        idx = np.minimum(np.searchsorted(cum, points, side='right'), U - 1)
        counts[s] += np.bincount(idx, minlength=U)
    return counts


def repair_counts(n: np.ndarray, pi_t: np.ndarray, budget_adm: int, budget_high: int) -> tuple[np.ndarray, int, int]:
    '''
    Greedy sample-path repair of integer counts n[s, u] (u = 2a + b, a = admit, b = high rate).

    While the admissions exceed budget_adm, one arm moves from (1, b) to (0, b): the state with the smallest
    pi[t, s, a=1] is chosen among states that hold an admitted arm (ties: largest s), and within it the
    variant b with the smaller pi[t, s, (1, b)] (ties: b = 0). Then, while the high-rate requests exceed
    budget_high, one arm moves from (a, 1) to (a, 0): the state with the smallest pi[t, s, b=1] (ties: smallest
    s), and within it the variant a with the smaller pi[t, s, (a, 1)] (ties: a = 0). The two repairs do not
    interact, and unused budget is never filled. Returns (repaired counts, admission moves, high-rate moves).
    '''
    n = np.array(n, dtype=np.int64)
    S = n.shape[0]
    moves_adm = 0
    pa = pi_t[:, 2] + pi_t[:, 3]
    while int(n[:, 2:].sum()) > budget_adm:
        s = min((s for s in range(S) if n[s, 2] + n[s, 3] > 0), key=lambda s: (pa[s], -s))
        u = min((u for u in (2, 3) if n[s, u] > 0), key=lambda u: (pi_t[s, u], u))
        n[s, u] -= 1
        n[s, u - 2] += 1
        moves_adm += 1
    moves_high = 0
    pb = pi_t[:, 1] + pi_t[:, 3]
    while int(n[:, 1::2].sum()) > budget_high:
        s = min((s for s in range(S) if n[s, 1] + n[s, 3] > 0), key=lambda s: (pb[s], s))
        u = min((u for u in (1, 3) if n[s, u] > 0), key=lambda u: (pi_t[s, u], u))
        n[s, u] -= 1
        n[s, u - 1] += 1
        moves_high += 1
    return n, moves_adm, moves_high


def assign_arms(states: np.ndarray, n: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    '''
    Give each arm the action prescribed by the counts n[s, u] of its state, uniformly at random within the state.
    Returns the requested (a, b) as int8 arrays.
    '''
    N = states.size
    S, U = n.shape
    keys = rng.random(N)
    order = np.lexsort((keys, states))  # arms grouped by state, in random order within each state
    sequence = np.concatenate([np.repeat(np.arange(U), n[s]) for s in range(S)])
    if sequence.size != N:
        raise RuntimeError(f'counts sum to {sequence.size}, expected {N} arms')
    actions = np.empty(N, dtype=np.int64)
    actions[order] = sequence
    return ADMIT[actions].astype(np.int8), HIGH[actions].astype(np.int8)


class _Coordinated:
    '''Shared round, repair and assignment step of the coordinated controllers.'''

    def _set_budgets(self, budgets) -> None:
        self.budgets = np.asarray(budgets, dtype=float)
        if self.budgets.shape != (2,):
            raise ValueError(f'budgets must be (alpha, beta), got {self.budgets.shape}')

    def reset(self, N: int, rng: np.random.Generator) -> None:
        self.N = int(N)
        self.budget_int = integer_budgets(self.budgets, self.N)
        self.counters = _zero_counters()

    def _decide(self, pi_t: np.ndarray, states: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, 'budget_int'):
            raise RuntimeError('call reset(N, rng) before act')
        if states.shape != (self.N,):
            raise ValueError(f'states must have shape ({self.N},), got {states.shape}')
        S = pi_t.shape[0]
        counts = np.bincount(states, minlength=S)
        n = systematic_round(counts[:, None] * pi_t, rng)
        n, moves_adm, moves_high = repair_counts(n, pi_t, int(self.budget_int[0]), int(self.budget_int[1]))
        self.counters['repairs'] += moves_adm + moves_high
        self.counters['repairs_adm'] += moves_adm
        self.counters['repairs_high'] += moves_high
        return assign_arms(states, n, rng)


class LPProjectionController(_Coordinated):
    '''
    Open-loop LP projection: round and repair the stage-t conditional policy pi[t] to sample-path budgets.

    `pi` is a (T, S, U) conditional policy (for example ``conditional_policy(y)``), `budgets` is
    ``(alpha, beta)`` as fractions of N. Integer budgets are floor(budget N + 1e-9) at reset.
    '''

    def __init__(self, pi, budgets, rounding: str = 'randomized'):
        if rounding != 'randomized':
            raise ValueError("rounding must be 'randomized'")
        self.rounding = rounding
        self.pi = normalize_policy(pi)
        self._set_budgets(budgets)
        self.T, self.S, self.U = self.pi.shape
        self.counters = _zero_counters()

    def act(self, t: int, states: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        return self._decide(self.pi[t], states, rng)


class _HorizonLP:
    '''
    Relaxed LP of Section 3 over `model.T` stages with the initial distribution as a cvxpy Parameter.
    The problem is DPP, so only the parameter value changes between solves. Clarabel is the solver.
    '''

    def __init__(self, model: WCMDP, Gamma_y: float):
        struct = build_lp(model)
        T, S, U = model.T, model.S, model.U
        n_res = struct.L * T
        self.shape = (T, S, U)
        self.y = cp.Variable(struct.n_y, nonneg=True)
        self.m0 = cp.Parameter(S, nonneg=True)
        initial_map = np.zeros((struct.A.shape[0], S))
        initial_map[:S] = np.eye(S)
        constraints = [
            struct.A @ self.y == cp.Constant(initial_map) @ self.m0,
            struct.D[:n_res] @ self.y <= struct.r[:n_res],
        ]
        objective = struct.c @ self.y
        if Gamma_y > 0:
            objective = objective + 0.5 * Gamma_y * cp.sum_squares(self.y)
        self.problem = cp.Problem(cp.Minimize(objective), constraints)

    def solve(self, m0: np.ndarray) -> np.ndarray:
        self.m0.value = np.asarray(m0, dtype=float)
        self.problem.solve(solver=_SOLVER)
        if self.problem.status not in _OK_STATUSES or self.y.value is None:
            raise RuntimeError(f'horizon LP failed with status {self.problem.status}')
        return np.asarray(self.y.value, dtype=float).reshape(self.shape)


_HORIZON_LPS: dict[tuple, _HorizonLP] = {}


def _model_digest(model: WCMDP) -> str:
    digest = hashlib.sha256()
    for arr in (model.P, model.c, model.d, model.budgets):
        arr = np.ascontiguousarray(arr, dtype=float)
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _horizon_lp(model: WCMDP, T: int, Gamma_y: float) -> _HorizonLP:
    '''One parametrized problem per (model, remaining horizon, Gamma_y), cached for the life of the process.'''
    key = (_model_digest(model), int(T), float(Gamma_y))
    lp = _HORIZON_LPS.get(key)
    if lp is None:
        lp = _HORIZON_LPS[key] = _HorizonLP(replace(model, T=int(T)), float(Gamma_y))
    return lp


class LPUpdateController(_Coordinated):
    '''
    Closed-loop LP update. `model` is the detailed WCMDP (``queue_wcmdp``). With `partition`, the aggregate
    LP of ``aggregate(model, partition)`` is solved instead: the empirical distribution is aggregated with V,
    and the stage policy of each state is the lifted block policy of its block.
    '''

    def __init__(self, model: WCMDP, partition=None, Gamma_y: float = 0.0):
        self.model = model
        self.Gamma_y = float(Gamma_y)
        self._set_budgets(model.budgets)
        self.T, self.S, self.U = model.T, model.S, model.U
        if partition is None:
            self.solve_model = model
            self.V = None
            self.lift = None
            self.block_of_state = None
        else:
            blocks = as_partition(partition, self.S)
            self.lift, self.V = maps(blocks, self.S)
            self.block_of_state = np.argmax(self.V, axis=0)
            self.solve_model = aggregate(model, blocks)
        self._open_loop: np.ndarray | None = None
        self.counters = _zero_counters()

    def act(self, t: int, states: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        pi_t = self._stage_policy(t, states)
        return self._decide(pi_t, states, rng)

    def _stage_policy(self, t: int, states: np.ndarray) -> np.ndarray:
        m = np.bincount(states, minlength=self.S) / states.size
        try:
            return self._solved_policy(t, m)
        except (RuntimeError, SolverError):
            self.counters['fallbacks'] += 1
            return self._open_loop_policy()[t]

    def _solved_policy(self, t: int, m: np.ndarray) -> np.ndarray:
        lp = _horizon_lp(self.solve_model, self.T - t, self.Gamma_y)
        if self.V is None:
            return _normalize_rows(lp.solve(m)[0])
        block_policy = _normalize_rows(lp.solve(self.V @ m)[0])
        return block_policy[self.block_of_state]

    def _open_loop_policy(self) -> np.ndarray:
        if self._open_loop is None:
            lp = _horizon_lp(self.solve_model, self.T, self.Gamma_y)
            if self.V is None:
                y = lp.solve(self.model.m0)
            else:
                y = lift_y(lp.solve(self.V @ self.model.m0), self.lift)
            self._open_loop = conditional_policy(y)
        return self._open_loop


class IndependentRandomizedController:
    '''Decentralized execution of a conditional policy: arm n draws its own action from pi[t, S_n, :].'''

    def __init__(self, pi):
        self.pi = normalize_policy(pi)
        self.T, self.S, self.U = self.pi.shape
        self.counters = _zero_counters()

    def reset(self, N: int, rng: np.random.Generator) -> None:
        self.N = int(N)
        self.counters = _zero_counters()

    def act(self, t: int, states: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        cdf = np.cumsum(self.pi[t][states], axis=1)
        draws = rng.random(states.size)
        u = np.minimum((cdf < draws[:, None]).sum(axis=1), self.U - 1)
        return ADMIT[u].astype(np.int8), HIGH[u].astype(np.int8)
