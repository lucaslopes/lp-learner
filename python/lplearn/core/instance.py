'''
Load-balancing and auto-scaling instance of a finite-horizon weakly coupled MDP.

Conventions shared by the whole ``lplearn.core`` package:

- states ``s = 0, ..., K`` (queue length), so ``S = K + 1``;
- actions ``u = 2a + b`` with ``a`` = admit and ``b`` = high service rate,
  i.e. ``u0 = (0,0)``, ``u1 = (0,1)``, ``u2 = (1,0)``, ``u3 = (1,1)``;
- ``P[u, s, s']`` is row-stochastic, ``c[s, u]`` is the one-stage cost,
  ``d[l, s, u]`` the consumption of resource ``l`` (0 = admission, 1 = high rate)
  and ``budgets[l]`` the per-arm budget;
- the horizon has stages ``t = 0, ..., T-1`` and ``m0`` is the initial distribution.
'''


from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import binom


ACTIONS = ((0, 0), (0, 1), (1, 0), (1, 1))
ADMIT = np.array([0, 0, 1, 1])
HIGH = np.array([0, 1, 0, 1])
CASES = ('CJS', 'SJS')
RESOURCES = ('admission', 'high_rate')


@dataclass(frozen=True)
class QueueParams:
    '''
    Parameters of the queueing instance (Section 7 of the manuscript).\n
    `case`: 'CJS' (each pending job completes independently per slot) or 'SJS' (one job per slot).\n
    `p`: per-slot arrival probability, so interarrival epochs are Geometric(p) on {1, 2, ...}.\n
    `b_low`, `b_high`: per-slot service probabilities q(0) and q(1).\n
    `alpha`, `beta`: admission and high-rate budgets (fractions of queues per stage).\n
    `gamma`, `gamma_bar`: rejection and full-buffer admission penalties.\n
    `m0`: initial distribution over 0..K (default: all queues empty).\n
    '''

    K: int = 5
    T: int = 10
    case: str = 'CJS'
    p: float = 0.7
    b_low: float = 0.01
    b_high: float = 0.1
    alpha: float = 0.75
    beta: float = 0.25
    gamma: float = 100.0
    gamma_bar: float = 100.0
    m0: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.case not in CASES:
            raise ValueError(f"case must be one of {CASES}, got {self.case!r}")
        if self.K < 1 or self.T < 1:
            raise ValueError('K and T must be positive')
        if not (0 < self.p <= 1 and 0 <= self.b_low < self.b_high <= 1):
            raise ValueError('need 0 < p <= 1 and 0 <= b_low < b_high <= 1')
        if self.m0 is not None and len(self.m0) != self.K + 1:
            raise ValueError('m0 must have K+1 entries')

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, eq=False)
class WCMDP:
    '''Finite-horizon weakly coupled MDP data for one representative arm.'''

    P: np.ndarray        # (U, S, S) row-stochastic transition kernels
    c: np.ndarray        # (S, U) one-stage cost
    d: np.ndarray        # (L, S, U) resource consumption
    budgets: np.ndarray  # (L,) per-arm budgets
    m0: np.ndarray       # (S,) initial distribution
    T: int

    @property
    def S(self) -> int:
        return self.c.shape[0]

    @property
    def U(self) -> int:
        return self.c.shape[1]

    @property
    def L(self) -> int:
        return self.d.shape[0]


def storage_cost(K: int) -> np.ndarray:
    '''C_s(s) = s / K for s = 0..K.'''
    return np.arange(K + 1) / K


def processing_cost(q: float) -> float:
    '''C_p(b) = 2 (1 + q(b)), the legacy linear service-capacity cost.'''
    return 2.0 * (1.0 + q)


def service_probability(b: int, b_low: float, b_high: float) -> float:
    '''q(b) = b_low + b (b_high - b_low).'''
    return b_low + b * (b_high - b_low)


@lru_cache(maxsize=4096)
def _kernel_cached(case: str, K: int, p: float, q: float, tol: float) -> np.ndarray:
    S = K + 1
    if p >= 1.0:
        taus, weights = np.array([1]), np.array([1.0])
    else:
        tau_max = int(np.ceil(np.log(tol) / np.log(1.0 - p))) + 1
        taus = np.arange(1, max(tau_max, 2) + 1)
        weights = p * (1.0 - p) ** (taus - 1)
    kernel = np.zeros((2, S, S))
    for a in range(2):
        for s in range(S):
            n = s + a if s < K else s  # an admission to a full queue is blocked
            if case == 'CJS':
                remain = (1.0 - q) ** taus  # each pending job is still present after tau slots
                left = np.arange(n + 1)
                probs = binom.pmf(left[None, :], n, remain[:, None])  # (tau, n+1)
                row = weights @ probs
                kernel[a, s, : n + 1] = row
            else:  # SJS: at most one completion per slot, Bin(tau, q) completions
                done = np.arange(n)
                probs = binom.pmf(done[None, :], taus[:, None], q) if n > 0 else np.zeros((len(taus), 0))
                tail = binom.sf(n - 1, taus, q)  # P(Bin(tau, q) >= n)
                kernel[a, s, n - done] = weights @ probs
                kernel[a, s, 0] += weights @ tail
            kernel[a, s] /= kernel[a, s].sum()
    kernel.setflags(write=False)
    return kernel


def transition_kernel(case: str, K: int, p: float, q: float, tol: float = 1e-15) -> np.ndarray:
    '''
    Embedded transition kernel at arrival epochs for a fixed service probability q.\n
    Returns an array of shape (2, K+1, K+1) indexed by [a, s, s'].\n
    The interarrival time tau is Geometric(p); the mixture over tau is truncated when the
    remaining geometric tail mass is below `tol` and each row is renormalized exactly.\n
    '''
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    return np.array(_kernel_cached(case, int(K), float(p), float(q), float(tol)))


def queue_wcmdp(params: QueueParams) -> WCMDP:
    '''Build the detailed WC-MDP of the load-balancing / auto-scaling instance.'''
    K, S = params.K, params.K + 1
    P = np.empty((4, S, S))
    c = np.empty((S, 4))
    d = np.zeros((2, S, 4))
    storage = storage_cost(K)
    for u, (a, b) in enumerate(ACTIONS):
        q = service_probability(b, params.b_low, params.b_high)
        P[u] = transition_kernel(params.case, K, params.p, q)[a]
        c[:, u] = storage + processing_cost(q) + params.gamma * (a == 0)
        c[K, u] += params.gamma_bar * (a == 1)
        d[0, :, u] = a
        d[1, :, u] = b
    if params.m0 is None:
        m0 = np.zeros(S)
        m0[0] = 1.0
    else:
        m0 = np.asarray(params.m0, dtype=float)
        m0 = m0 / m0.sum()
    budgets = np.array([params.alpha, params.beta], dtype=float)
    return WCMDP(P=P, c=c, d=d, budgets=budgets, m0=m0, T=params.T)
