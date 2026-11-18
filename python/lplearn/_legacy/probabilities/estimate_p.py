'''
This module contains the function to estimate the transition probabilities.
'''


import numpy as np
from lplearn._legacy.utils import numba_njit
from lplearn._legacy.probabilities.simulation import simulate_next_state


@numba_njit
def estimate_transition_probabilities(
        P: np.ndarray,
        R: int,
        rng: int,
    ) -> np.ndarray:

    '''
    Estimate the transition probabilities using the simulation method.\n
    `P`: Normalized transition probabilities.\n
    `R`: Number of simulations of the next state to estimate the transition probabilities.\n
    '''

    K = P.shape[0] - 1
    P_hat = np.zeros((K+1, K+1, 2, 2))

    for s in range(K+1):
        for a in range(2):
            for b in range(2):
                temp_s_ = np.zeros(K+1)
                for _ in range(R):
                    s_ = simulate_next_state(P, s, a, b, rng)
                    temp_s_[s_] += 1
                P_hat[:,s,a,b] = temp_s_ / R

                # Check that probabilities sum to 1
                if not (0.98 < np.sum(P_hat[:, s, a, b]) < 1.02):
                    raise ValueError(f'Sum of probabilities is not close to 1.0, it is {np.sum(P_hat[:, s, a, b])}')

    return P_hat
