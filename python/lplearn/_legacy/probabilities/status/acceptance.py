'''
This module contains functions to calculate the probability of accepting a job.
'''


import numpy as np
from lplearn._legacy.utils import numba_njit


@numba_njit
def probability_of_accepting_a_job_numpy(
        y: np.ndarray,
        alpha: float,
        to_round: bool = False
    ) -> float:

    '''
    Probability of accepting a job.\n
    `y`: Policy matrix.\n
    `alpha`: Acceptance limit.\n
    `to_round`: Whether to round the result to 4 decimal places or not.\n
    $pi_A(t) &= sum_{s=0}^{K-1} sum_{b=0}^1 y_{s,1,b}(t) = 1-sum_{s=0}^{K} sum_{b=0}^1 y_{s,0,b}$.\n
    $hat{pi}_A = frac{1}{alpha T} sum_{t=0}^{T-1} pi_A(t)$.\n
    return $hat{pi}_A$.\n
    '''

    T = y.shape[-1]
    K = y.shape[0] - 1
    sum = y[:K,1,:,:].sum()
    prob = sum / (alpha * T)
    prob = round(prob, 4) if to_round else prob


    return prob


@numba_njit
def probability_of_accepting_a_job_loop(
        y: np.ndarray,
        alpha: float,
        to_round: bool = False
    ) -> float:

    '''
    Probability of accepting a job.\n
    `y`: Policy matrix.\n
    `alpha`: Acceptance limit.\n
    `to_round`: Whether to round the result to 4 decimal places or not.\n
    $pi_A(t) &= sum_{s=0}^{K-1} sum_{b=0}^1 y_{s,1,b}(t) = 1-sum_{s=0}^{K} sum_{b=0}^1 y_{s,0,b}$.\n
    $hat{pi}_A = frac{1}{alpha T} sum_{t=0}^{T-1} pi_A(t)$.\n
    return $hat{pi}_A$.\n
    '''

    T = y.shape[-1]
    K = y.shape[0] - 1

    sum = 0
    for t in range(T):
        for s in range(K-1):
            for b in range(y.shape[2]):
                sum += y[s,1,b,t]
    prob = sum / (alpha * T)
    prob = round(prob, 4) if to_round else prob

    return prob
