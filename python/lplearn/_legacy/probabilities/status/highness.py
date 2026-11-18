'''
This module contains functions to calculate the probability of using the high service rate.
'''


import numpy as np
from lplearn._legacy.utils import numba_njit


@numba_njit
def probability_of_use_high_service_rate_numpy(
        y: np.ndarray,
        beta: float,
        to_round: bool = False
    ) -> float:

    '''
    Probability of using the high service rate.\n
    `y`: Policy matrix.\n
    `beta`: High Rate Usage Ratio Limit.\n
    `to_round`: If True, the result is rounded to 4 decimal places.\n
    $pi_H(t) &= sum_{s=0}^{K} sum_{a=0}^1 y_{s,a,1}(t)$\n
    $hat{pi}_H = frac{1}{\beta T} sum_{t=0}^{T-1} pi_H(t)$\n
    return $hat{pi}_H$.\n
    '''

    T = y.shape[-1]
    sum = y[:,:,1,:].sum()
    prob = sum / (beta * T)
    prob = round(prob, 4) if to_round else prob

    return prob


@numba_njit
def probability_of_use_high_service_rate_loop(
        y: np.ndarray,
        beta: float,
        to_round: bool = False
    ) -> float:

    '''
    Probability of using the high service rate.\n
    `y`: Policy matrix.\n
    `beta`: High Rate Usage Ratio Limit.\n
    `to_round`: If True, the result is rounded to 4 decimal places.\n
    $pi_H(t) &= sum_{s=0}^{K} sum_{a=0}^1 y_{s,a,1}(t)$.\n
    $hat{pi}_H = frac{1}{\beta T} sum_{t=0}^{T-1} pi_H(t)$.\n
    return $hat{pi}_H$.\n
    '''

    T = y.shape[-1]
    K = y.shape[0] - 1
    sum = 0
    for t in range(T):
        for s in range(K):
            for a in range(y.shape[1]):
                sum += y[s,a,1,t]
    prob = sum / (beta * T)
    prob = round(prob, 4) if to_round else prob

    return prob
