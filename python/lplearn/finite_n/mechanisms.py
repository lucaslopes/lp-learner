'''
Mechanisms map the REQUESTED actions of a controller to the REALIZED actions of the N-arm system.

A mechanism is any object with a ``name`` string and the method

    apply(t, states, a_req, b_req, budgets_int, rng) -> (a, b, info)

where ``a_req``, ``b_req`` are int8 arrays of shape (N,), ``budgets_int`` is the integer budget vector
``(floor(alpha N + 1e-9), floor(beta N + 1e-9))`` and ``info`` is a dict with the int keys
``denied_adm`` and ``denied_high`` (requests that were turned into 0 by the mechanism).

Three mechanisms are provided: ``Unenforced`` (realized = requested, budgets may be violated),
``StrictBudget`` (realized = requested, but raises if a budget is exceeded, so that an infeasible
coordinated controller cannot go unnoticed) and ``RandomRationing`` (uniform random rationing).
'''


from __future__ import annotations

import numpy as np


ADMISSION, HIGH_RATE = 0, 1


def integer_budgets(budgets, N: int) -> np.ndarray:
    '''Integer per-stage budgets floor(budget N + 1e-9) for N arms; `budgets` = (alpha, beta).'''
    return np.floor(np.asarray(budgets, dtype=float) * int(N) + 1e-9).astype(np.int64)


def _copy_actions(a_req: np.ndarray, b_req: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.array(a_req, dtype=np.int8), np.array(b_req, dtype=np.int8)


def _ration(requests: np.ndarray, budget: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    '''Keep a uniformly random subset of exactly `budget` requesters when there are more requests than budget.'''
    granted = np.zeros(requests.shape, dtype=np.int8)
    requesters = np.flatnonzero(requests)
    if requesters.size <= budget:
        granted[requesters] = 1
        return granted, 0
    kept = rng.choice(requesters, size=int(budget), replace=False)
    granted[kept] = 1
    return granted, int(requesters.size - budget)


class Unenforced:
    '''Realized actions equal the requested actions; the budgets are not enforced.'''

    name = 'Unenforced'

    def apply(self, t, states, a_req, b_req, budgets_int, rng):
        a, b = _copy_actions(a_req, b_req)
        return a, b, {'denied_adm': 0, 'denied_high': 0}


class StrictBudget:
    '''
    Realized actions equal the requested actions. Raises ValueError when the requests exceed an integer
    budget, so that a coordinated controller that is infeasible on some sample path fails loudly.
    '''

    name = 'StrictBudget'

    def apply(self, t, states, a_req, b_req, budgets_int, rng):
        admitted = int(np.sum(a_req))
        highs = int(np.sum(b_req))
        if admitted > int(budgets_int[ADMISSION]) or highs > int(budgets_int[HIGH_RATE]):
            raise ValueError(
                f'stage {t}: requested admissions {admitted} (budget {int(budgets_int[ADMISSION])}) and '
                f'high-rate requests {highs} (budget {int(budgets_int[HIGH_RATE])}) violate the sample-path budgets'
            )
        a, b = _copy_actions(a_req, b_req)
        return a, b, {'denied_adm': 0, 'denied_high': 0}


class RandomRationing:
    '''
    If the requests of a resource exceed its integer budget, grant a uniformly random subset of requesters of
    exactly the budget size. Denied requests become 0. The two resources are rationed independently; the
    admission draw is made first, then the high-rate draw.
    '''

    name = 'RandomRationing'

    def apply(self, t, states, a_req, b_req, budgets_int, rng):
        a, denied_adm = _ration(np.asarray(a_req, dtype=np.int8), int(budgets_int[ADMISSION]), rng)
        b, denied_high = _ration(np.asarray(b_req, dtype=np.int8), int(budgets_int[HIGH_RATE]), rng)
        return a, b, {'denied_adm': denied_adm, 'denied_high': denied_high}


MECHANISMS = {cls.name: cls for cls in (Unenforced, StrictBudget, RandomRationing)}
