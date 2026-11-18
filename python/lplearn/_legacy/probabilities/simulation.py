'''
This module contains the function that simulates the next state given the current state and the actions.
'''


import numpy as np
from lplearn._legacy.utils import numba_njit


@numba_njit
def simulate_next_state(
        P: np.ndarray,
        s: int,
        a: int,
        b: int,
        rng: int,
    ) -> int:

    '''
    Simulate the next state given the current state and the actions.\n
    `P`: Normalized transition probabilities.\n
    `s`: Current state.\n
    `a`: Load balancing action (0 to reject) or (1 to accept).\n
    `b`: Service rate action (0 for low service rate) or (1 for high service rate).\n
    '''

    U = rng.uniform(0, 1)

    K = P.shape[0] - 1
    indicator = np.zeros(K + 1)
    pxa = P[:,s,a,b]
    indicator[0] = 1 if pxa[0] > U else 0
    for y in range(1, K + 1):
        if np.sum(pxa[:y]) <= U < np.sum(pxa[:y+1]):
            indicator[y] = 1
        else:
            indicator[y] = 0
    next_state = np.argmax(indicator)  # Picking the state for which the indicator function is equal to 1

    return next_state
